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

    # for batch in dataloader:
    #     d_in = batch["input"]
    #     target = batch["target"]
    #     sun_feat = batch["sun_feat"]
    #
    #     print(d_in.shape)
    #     print(target.shape)
    #     print(sun_feat)


if __name__ == '__main__':
    run()
