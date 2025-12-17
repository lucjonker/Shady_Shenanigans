import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchgeo.datasets import stack_samples, unbind_samples, IntersectionDataset
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler

from dataset import ShadowDataset, DSMDataset


def run():
    dsm_filepath = "/Users/luc/Geomatics/Thesis/Data/input/271110_Ams/test_tiles/271110_p_175_2022_06_02_dsm.tif"
    shadow_filepath = "/Users/luc/Geomatics/Thesis/Data/output/building_shade/175/"

    # dsm_ds = DSMDataset(dsm_filepath)
    # shadow_ds = ShadowDataset(shadow_filepath)
    #
    # paired = IntersectionDataset(dsm_ds, shadow_ds, spatial_only=True)
    #
    # sampler = GridGeoSampler(dsm_ds, size=512, stride=256)
    # dataloader = DataLoader(paired, sampler=sampler, collate_fn=stack_samples)
    #
    # for batch in dataloader:
    #     print(batch)

if __name__ == '__main__':
    run()
