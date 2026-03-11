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
    write_dataset_csv(TEST_DATA_PATH, CSV_PATH)
    # dataset = DSMShadeDataset("/Users/luc/Geomatics/Thesis/ShadyShenanigans/resources/dataset.csv", TEST_TILE_PATH,
    #                           TEST_SHADE_PATH, tile_size=512)
    #
    # # Test that tiling works
    # time_before = time.time()
    # sobel = Sobel("cpu")
    # for i in range(5):
    #     data = dataset.__getitem__(i)
    #     source = data["source"][0].unsqueeze(0).unsqueeze(0)
    #     target = data["target"].unsqueeze(0)
    #
    #     print(composite_loss(sobel, source, target))
    #
    #     # print(f"Tile: {data["dsm_id"]} and {data["shade_id"]}, subtile: {data['subtile']}")
    #     # print("-------------------")
    # time_after = time.time()
    # total_time = time_after - time_before
    # print("Total time: ", total_time)
    #
    # # show inputs and targets
    # plt.figure(figsize=(16, 6))
    # for i in range(5):
    #     plt.subplot(2, 5, i + 1)
    #     data = dataset.__getitem__(i)
    #     source = data["source"]
    #     # dsm_image = source[0]
    #     dsm_image = sobel(source[0].unsqueeze(0).unsqueeze(0))
    #     plt.imshow(dsm_image.squeeze().numpy())
    #     plt.axis('off')
    #
    #     plt.subplot(2, 5, i + 6)
    #     target = data["target"]
    #     shade_image = sobel(target[0].unsqueeze(0).unsqueeze(0))
    #     plt.imshow(shade_image.squeeze().numpy())
    #     plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    run()