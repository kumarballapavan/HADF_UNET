A sample of the dataset is included in the data folder. 

This work is currently under consideration for a journal publication. 

The details of this work will be provided post the publication process.



# HADF-UNet Training

## Required Packages

Install the required packages using:

```bash
pip install torch torchvision pillow 


Dataset Structure

Arrange the dataset in the following format:

data/
  train/
    hazy/
    clear/
  val/
    hazy/
    clear/
  test/
    hazy/
    clear/


How to Train

Open hadf_unet_train.py and set the dataset path and output path in the Config section:

dataset_root = "./dataset"
out_dir = "./hadf_unet_runs/exp1"

Then run:

python hadf_unet_train.py


Output

The training script saves:

best.pt — best model checkpoint
history.json — training history
summary.json — final test summary

