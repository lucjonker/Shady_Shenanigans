import rasterio
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Todo figure it out
# https://medium.com/@ns_geoai/custom-geospatial-dataloader-with-pytorch-and-rasterio-4f6d896ef441
class ShadePredictionDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        self.file_paths = data_dirs
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.file_paths[idx], 'r') as src:
            data = src.read()
            if self.transform:
                data = self.transform(data)
            return data