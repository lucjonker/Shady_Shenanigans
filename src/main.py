from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import DSMShadeDataset
from src.utils import write_dataset_csv

# Define resources for training
TEST_TILE_PATH = "/Users/luc/Geomatics/Thesis/Data/input/271110_Ams/test_tiles/dsms/combined/"
TEST_SHADE_PATH = "/Users/luc/Geomatics/Thesis/Data/output/tree_shade/"
CSV_PATH = "/Users/luc/Geomatics/Thesis/ShadyShenanigans/resources/dataset.csv"

DSM_REGEX = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_(?P<date>\d{4}_\d{2}_\d{2})_dsm.tif$"
SHADE_REGEX = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_Shadow_(?P<date>\d{8}_\d{4})_LST.tif$"


def run():
    # print("Generating dataset CSV")
    # write_dataset_csv(TEST_TILE_PATH, TEST_SHADE_PATH, DSM_REGEX, SHADE_REGEX, CSV_PATH)

    dataset = DSMShadeDataset("/Users/luc/Geomatics/Thesis/ShadyShenanigans/resources/dataset.csv", TEST_TILE_PATH, TEST_SHADE_PATH)
    # Create a DataLoader for batching and parallel data loading
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    len_dataset = len(dataset)
    num_batches = len(dataloader)
    print(len_dataset, num_batches)

    # show inputs and targets
    plt.figure(figsize=(16, 6))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        data = dataloader.dataset.__getitem__(i)
        input = data["input"]
        dsm_image = input[0]
        plt.imshow(dsm_image.squeeze().numpy())
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        target = data["target"]
        shade_image = target[0]
        plt.imshow(shade_image.squeeze().numpy())
        plt.axis('off')

        print(f"Pair {i + 1}: {data["dsm"]} and {data["shade"]}")

    plt.show()


if __name__ == '__main__':
    run()
