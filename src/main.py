import csv
import random
import time

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms

from src.dataset import DSMShadeDataset
from src.loss_functions import ssim_loss, sobel_loss, composite_loss
from src.utils import write_dataset_csv, Sobel

# Define resources for training
TEST_DATA_PATH = "/Users/luc/Geomatics/Thesis/test_data/"
CSV_PATH = "../resources/dataset.csv"


def run():
    print("Generating dataset CSV")
    # write_dataset_csv(TEST_DATA_PATH, CSV_PATH)
    #
    # with open('/Users/luc/Geomatics/Thesis/ShadyShenanigans/results/results/loss_logs/version_0/metrics.csv', mode='r') as infile:
    #     reader = csv.reader(infile)
    #     mydict = dict((rows[0], rows[1]) for rows in reader)
    #     print(mydict["19"])


if __name__ == '__main__':
    run()