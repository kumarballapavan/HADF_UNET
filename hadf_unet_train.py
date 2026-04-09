
#!/usr/bin/env python3
"""
HADF-UNet training script based on the manuscript:
"HazeSat-Net-Dual: A Prior-Guided Dual-Frequency Haze-Aware Dehazing Framework
 on a Sentinel-2 MSI Multi-regional Hazy Dataset for Enhanced Land Cover"

What this script implements from the manuscript:
- 9-channel input = RGB + 6 physics priors
  (DCP, luminance, MMD, local contrast, edge magnitude, color saturation)
- Dual-frequency decomposition using non-trainable box blur
- Low-frequency and high-frequency branches
- Residual U-Net encoder/decoder
- Haze-level head from E1
- Haze-gated fusion: I_out = g * I_hat + (1 - g) * I_h
- Hybrid loss:
    lambda1 * Charbonnier(I_hat, I_clear)
  + lambda2 * (1 - SSIM(I_hat, I_clear))
  + lambda3 * L1(g, g_target)

Implementation notes:
- The manuscript specifies the architecture and loss clearly, but not the exact
  optimizer, learning-rate schedule, batch size, or epoch count.
  Reasonable defaults are used here and can be edited in CONFIG.
- For evaluation, PSNR/SSIM are computed on the final fused output I_out.
- The haze proxy target g_target is spatially averaged to match scalar gate g.
"""

import os
import math
import json
import time
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision import transforms


# ============================================================
# CONFIG
# ============================================================
@dataclass
class Config:
    # Dataset root structure:
    # dataset_root/
    #   train/
    #     hazy/
    #     clear/
    #   val/
    #     hazy/
    #     clear/
    #   test/
    #     hazy/
    #     clear/
    #
    # Pairing rule: same filename in hazy and clear folders.
    dataset_root: str = "./dataset"
    out_dir: str = "./hadf_unet_runs/exp1"

    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 0
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42

    base_channels: int = 32
    blur_kernel: int = 5
    prior_window: int = 15

    lambda_char: float = 1.0
    lambda_ssim: float = 0.5
    lambda_gate: float = 0.1
    charbonnier_eps: float = 1e-3

    use_amp: bool = True
    save_every: int = 10
    early_stop_patience: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = Config()


# ============================================================
# UTILS
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def list_common_filenames(hazy_dir: Path, clear_dir: Path) -> List[str]:
    hazy_files = {p.name for p in hazy_dir.iterdir() if p.is_file()}
    clear_files = {p.name for p in clear_dir.iterdir() if p.is_file()}
    common = sorted(hazy_files.intersection(clear_files))
    if not common:
        raise FileNotFoundError(f"No matching filenames found in {hazy_dir} and {clear_dir}")
    return common


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def tensor_to_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10((data_range ** 2) / mse)


# ============================================================
# DATASET
# ============================================================
class PairedDehazeDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int = 256, train: bool = False):
        self.root = Path(root)
        self.split = split
        self.hazy_dir = self.root / split / "hazy"
        self.clear_dir = self.root / split / "clear"

        if not self.hazy_dir.exists() or not self.clear_dir.exists():
            raise FileNotFoundError(f"Missing split folders: {self.hazy_dir} or {self.clear_dir}")

        self.names = list_common_filenames(self.hazy_dir, self.clear_dir)

        if train:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
            ])

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        hazy = Image.open(self.hazy_dir / name).convert("RGB")
        clear = Image.open(self.clear_dir / name).convert("RGB")

        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        random.seed(seed)
        torch.manual_seed(seed)
        hazy = self.tf(hazy)
        random.seed(seed)
        torch.manual_seed(seed)
        clear = self.tf(clear)

        hazy = to_tensor(hazy)   # [0,1], CxHxW
        clear = to_tensor(clear)

        return hazy, clear, name


# ============================================================
# PRIORS
# ============================================================
class HazePriors(nn.Module):
    def __init__(self, window_size: int = 15, eps: float = 1e-6):
        super().__init__()
        self.window_size = window_size
        self.pad = window_size // 2
        self.eps = eps

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def minmax_normalize_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x: Bx1xHxW
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        # Dark channel prior
        p_dc = torch.min(rgb, dim=1, keepdim=True).values

        # Luminance
        p_y = 0.299 * r + 0.587 * g + 0.114 * b

        # Maximum-minimum intensity difference on local window
        local_max = F.max_pool2d(p_y, kernel_size=self.window_size, stride=1, padding=self.pad)
        local_min = -F.max_pool2d(-p_y, kernel_size=self.window_size, stride=1, padding=self.pad)
        p_mmd = local_max - local_min

        # Local contrast = sqrt(var + eps)
        mean = F.avg_pool2d(p_y, kernel_size=self.window_size, stride=1, padding=self.pad)
        mean_sq = F.avg_pool2d(p_y * p_y, kernel_size=self.window_size, stride=1, padding=self.pad)
        var = (mean_sq - mean * mean).clamp_min(0.0)
        p_lc = torch.sqrt(var + self.eps)

        # Edge magnitude on luminance
        gx = F.conv2d(p_y, self.sobel_x, padding=1)
        gy = F.conv2d(p_y, self.sobel_y, padding=1)
        p_edge = torch.sqrt(gx * gx + gy * gy + self.eps)

        # Color saturation
        mean_rgb = rgb.mean(dim=1, keepdim=True)
        p_sat = torch.sqrt(((rgb - mean_rgb) ** 2).mean(dim=1, keepdim=True) + self.eps)

        # Normalize priors to [0,1] for stable learning and g_target construction
        priors = {
            "dc": self.minmax_normalize_per_image(p_dc, self.eps),
            "y": self.minmax_normalize_per_image(p_y, self.eps),
            "mmd": self.minmax_normalize_per_image(p_mmd, self.eps),
            "lc": self.minmax_normalize_per_image(p_lc, self.eps),
            "edge": self.minmax_normalize_per_image(p_edge, self.eps),
            "sat": self.minmax_normalize_per_image(p_sat, self.eps),
        }
        return priors


# ============================================================
# BUILDING BLOCKS
# ============================================================
def conv_bn_gelu(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_gelu(ch, ch, 3, 1, 1),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DualFreqBranch(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.stem = conv_bn_gelu(in_ch, out_ch, 3, 1, 1)
        self.res1 = ResidualBlock(out_ch)
        self.res2 = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool):
        super().__init__()
        if downsample:
            self.proj = conv_bn_gelu(in_ch, out_ch, 3, 2, 1)
        else:
            self.proj = conv_bn_gelu(in_ch, out_ch, 3, 1, 1)
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.res(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_bn_gelu(in_ch, out_ch, 3, 1, 1),
        )
        self.match_skip = nn.Identity() if skip_ch == out_ch else nn.Conv2d(skip_ch, out_ch, kernel_size=1)
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.match_skip(skip)
        x = x + skip
        x = self.res(x)
        return x


# ============================================================
# MODEL
# ============================================================
class HADFUNet(nn.Module):
    def __init__(self, base_channels: int = 32, blur_kernel: int = 5, prior_window: int = 15):
        super().__init__()
        self.priors = HazePriors(window_size=prior_window)
        self.blur_kernel = blur_kernel

        # 9-channel input = RGB + 6 priors
        self.low_branch = DualFreqBranch(9, base_channels)
        self.high_branch = DualFreqBranch(9, base_channels)

        self.fuse = nn.Sequential(
            conv_bn_gelu(base_channels * 2, base_channels, 3, 1, 1),
            ResidualBlock(base_channels),
        )

        # Encoder: E1, E2, E3
        self.enc1 = EncoderStage(base_channels, base_channels, downsample=False)
        self.enc2 = EncoderStage(base_channels, base_channels * 2, downsample=True)
        self.enc3 = EncoderStage(base_channels * 2, base_channels * 4, downsample=True)

        # Haze-level head from E1
        self.haze_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

        # Decoder
        self.dec2 = DecoderStage(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderStage(base_channels * 2, base_channels, base_channels)

        self.out_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def box_blur(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.blur_kernel // 2
        return F.avg_pool2d(x, kernel_size=self.blur_kernel, stride=1, padding=pad)

    def build_x0(self, hazy: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        priors = self.priors(hazy)
        x0 = torch.cat([
            hazy,
            priors["dc"],
            priors["y"],
            priors["mmd"],
            priors["lc"],
            priors["edge"],
            priors["sat"],
        ], dim=1)
        return x0, priors

    def forward(self, hazy: torch.Tensor):
        x0, priors = self.build_x0(hazy)

        low = self.box_blur(x0)
        high = x0 - low

        f_low = self.low_branch(low)
        f_high = self.high_branch(high)
        f0 = self.fuse(torch.cat([f_low, f_high], dim=1))

        e1 = self.enc1(f0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        haze_map = self.haze_head(e1)                  # Bx1xHxW
        g = torch.sigmoid(haze_map.mean(dim=(2, 3), keepdim=True))  # Bx1x1x1 scalar gate per image

        d2 = self.dec2(e3, e2)
        d1 = self.dec1(d2, e1)

        i_hat = torch.sigmoid(self.out_head(d1))
        i_out = g * i_hat + (1.0 - g) * hazy

        return {
            "i_hat": i_hat,
            "i_out": i_out,
            "g": g,
            "haze_map": haze_map,
            "priors": priors,
        }


# ============================================================
# LOSSES / METRICS
# ============================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def create_gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_1d = g.view(1, 1, 1, -1)
    window_2d = (window_1d.transpose(-1, -2) @ window_1d).squeeze(0).squeeze(0)
    window_2d = window_2d / window_2d.sum()
    return window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
               data_range: float = 1.0, size_average: bool = True) -> torch.Tensor:
    channels = x.size(1)
    device = x.device
    window = create_gaussian_window(window_size, sigma, channels, device)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=(1, 2, 3))


def build_gate_target(priors: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Manuscript proxy:
    # g_target = 1/3 * (P_DC + (1 - P_LC) + (1 - P_Edge))
    proxy_map = (priors["dc"] + (1.0 - priors["lc"]) + (1.0 - priors["edge"])) / 3.0
    proxy_map = proxy_map.clamp(0.0, 1.0)
    return proxy_map.mean(dim=(2, 3), keepdim=True)  # Bx1x1x1


class HybridDehazeLoss(nn.Module):
    def __init__(self, lambda_char: float = 1.0, lambda_ssim: float = 0.5,
                 lambda_gate: float = 0.1, charbonnier_eps: float = 1e-3):
        super().__init__()
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_gate = lambda_gate
        self.charb = CharbonnierLoss(charbonnier_eps)

    def forward(self, outputs: Dict[str, torch.Tensor], clear: torch.Tensor):
        i_hat = outputs["i_hat"]
        g = outputs["g"]
        priors = outputs["priors"]

        l_char = self.charb(i_hat, clear)
        l_ssim = 1.0 - ssim_torch(i_hat, clear, size_average=True)

        g_target = build_gate_target(priors)
        l_gate = F.l1_loss(g, g_target)

        total = self.lambda_char * l_char + self.lambda_ssim * l_ssim + self.lambda_gate * l_gate

        return total, {
            "loss_total": total.detach().item(),
            "loss_char": l_char.detach().item(),
            "loss_ssim": l_ssim.detach().item(),
            "loss_gate": l_gate.detach().item(),
        }


# ============================================================
# TRAIN / EVAL
# ============================================================
def build_loaders(cfg: Config):
    train_ds = PairedDehazeDataset(cfg.dataset_root, "train", cfg.image_size, train=True)
    val_ds = PairedDehazeDataset(cfg.dataset_root, "val", cfg.image_size, train=False)
    test_ds = PairedDehazeDataset(cfg.dataset_root, "test", cfg.image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"))
    return train_loader, val_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for hazy, clear, _ in loader:
            hazy = hazy.to(device, non_blocking=True)
            clear = clear.to(device, non_blocking=True)

            outputs = model(hazy)
            pred = outputs["i_out"]

            psnr_vals = tensor_to_psnr(pred, clear)
            ssim_vals = ssim_torch(pred, clear, size_average=False)

            psnr_list.extend(psnr_vals.detach().cpu().tolist())
            ssim_list.extend(ssim_vals.detach().cpu().tolist())

    mean_psnr = sum(psnr_list) / max(1, len(psnr_list))
    mean_ssim = sum(ssim_list) / max(1, len(ssim_list))
    return {"psnr": mean_psnr, "ssim": mean_ssim}


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, scaler,
                    device: torch.device, epoch: int, cfg: Config):
    model.train()
    running = {"loss_total": 0.0, "loss_char": 0.0, "loss_ssim": 0.0, "loss_gate": 0.0}
    count = 0

    for hazy, clear, _ in loader:
        hazy = hazy.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_amp = cfg.use_amp and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(hazy)
            loss, loss_dict = criterion(outputs, clear)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = hazy.size(0)
        count += bs
        for k in running:
            running[k] += loss_dict[k] * bs

    for k in running:
        running[k] /= max(1, count)
    return running


def run(cfg: Config):
    seed_everything(cfg.seed)
    ensure_dir(cfg.out_dir)

    device = torch.device(cfg.device)
    train_loader, val_loader, test_loader = build_loaders(cfg)

    model = HADFUNet(
        base_channels=cfg.base_channels,
        blur_kernel=cfg.blur_kernel,
        prior_window=cfg.prior_window,
    ).to(device)

    criterion = HybridDehazeLoss(
        lambda_char=cfg.lambda_char,
        lambda_ssim=cfg.lambda_ssim,
        lambda_gate=cfg.lambda_gate,
        charbonnier_eps=cfg.charbonnier_eps,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    history = []
    best_score = -1e9
    best_epoch = -1
    patience = 0

    print("Starting HADF-UNet training...")
    print(json.dumps(asdict(cfg), indent=2))

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, cfg)
        val_stats = evaluate(model, val_loader, device)
        scheduler.step()

        score = val_stats["psnr"] + 10.0 * val_stats["ssim"]

        epoch_record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_stats,
            "val_psnr": val_stats["psnr"],
            "val_ssim": val_stats["ssim"],
            "time_sec": round(time.time() - t0, 2),
        }
        history.append(epoch_record)
        save_json(history, Path(cfg.out_dir) / "history.json")

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_stats['loss_total']:.4f} | "
            f"char={train_stats['loss_char']:.4f} | "
            f"ssim_loss={train_stats['loss_ssim']:.4f} | "
            f"gate={train_stats['loss_gate']:.4f} | "
            f"val_psnr={val_stats['psnr']:.4f} | "
            f"val_ssim={val_stats['ssim']:.4f} | "
            f"{epoch_record['time_sec']:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": asdict(cfg),
            "history": history,
        }

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience = 0
            torch.save(ckpt, Path(cfg.out_dir) / "best.pt")
            print(f"  -> saved best checkpoint at epoch {epoch}")
        else:
            patience += 1

        if epoch % cfg.save_every == 0:
            torch.save(ckpt, Path(cfg.out_dir) / f"epoch_{epoch:03d}.pt")

        if patience >= cfg.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # Final test with best checkpoint
    best_path = Path(cfg.out_dir) / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])

    test_stats = evaluate(model, test_loader, device)
    print("\nFinal test results:")
    print(f"PSNR: {test_stats['psnr']:.4f}")
    print(f"SSIM: {test_stats['ssim']:.4f}")

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "test_psnr": test_stats["psnr"],
        "test_ssim": test_stats["ssim"],
        "config": asdict(cfg),
    }
    save_json(summary, Path(cfg.out_dir) / "summary.json")
    print(f"\nSaved outputs to: {cfg.out_dir}")


if __name__ == "__main__":
    run(CONFIG)
