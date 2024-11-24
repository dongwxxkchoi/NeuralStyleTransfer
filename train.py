# import 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image

from model import StyleTransfer
from loss import StyleLoss, ContentLoss

# load data
## preprocessing
def pre_processing():
    pass

## postprocessing
def post_processing():
    pass

# load model

# load loss

# setting optimizer

def train_main():
# train loop
## loss print
## image gen output save
    pass

if __name__ == "__main__":
    train_main()