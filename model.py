import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Todo Extend
class Model(nn.Module):

    def __init__(self, model_name='deepshadow'):
        super().__init__()
        self.model_name = model_name
        # Define layers (unet? resnet? optional?)

    def forward(self, x):
        return x
