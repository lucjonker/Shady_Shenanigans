import random

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.dataset import DSMShadeDataset
from src.utils import write_dataset_csv

# Define resources for training
TEST_TILE_PATH = "/Users/luc/Geomatics/Thesis/Data/input/271110_Ams/test_tiles/dsms/combined/"
TEST_SHADE_PATH = "/Users/luc/Geomatics/Thesis/Data/output/tree_shade/"
CSV_PATH = "/Users/luc/Geomatics/Thesis/ShadyShenanigans/resources/dataset.csv"


def run():
    print("Generating dataset CSV")
    write_dataset_csv(TEST_TILE_PATH, TEST_SHADE_PATH, CSV_PATH)

    # dataset = DSMShadeDataset("/Users/luc/Geomatics/Thesis/ShadyShenanigans/resources/dataset.csv", TEST_TILE_PATH,
    #                           TEST_SHADE_PATH, tile_size=512)
    #
    # # # Test that tiling works
    # # for i in range(150):
    # #     data = dataset.__getitem__(i)
    # #     print(f"Tile: {data["dsm_id"]} and {data["shade_id"]}, subtile: {data['subtile']}")
    # #     print("-------------------")
    #
    # # show inputs and targets
    # plt.figure(figsize=(16, 6))
    # for i in range(5):
    #     plt.subplot(2, 5, i + 1)
    #     data = dataset.__getitem__(i)
    #     source = data["source"]
    #     dsm_image = source[0]
    #     plt.imshow(dsm_image.squeeze().numpy())
    #     plt.axis('off')
    #
    #     plt.subplot(2, 5, i + 6)
    #     target = data["target"]
    #     shade_image = target[0]
    #     plt.imshow(shade_image.squeeze().numpy())
    #     plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    run()