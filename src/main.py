import os
from pathlib import Path as P
from utils import write_dataset_csv

# Define resources for training todo modify to daic
TEST_DATA_PATH = "/Volumes/UDSstorage/msc_ljonker/training_data"
CSV_PATH = "/tudelft.net/UDSstorage/msc_ljonker/ShadyShenanigans/resources/"


def run():
    print("Generating dataset CSV")
    data_path = P(os.getenv('DATASETS_ROOT', default=TEST_DATA_PATH))
    write_dataset_csv(data_path, CSV_PATH)

    # dataset = DSMShadeDataset("/Users/luc/Geomatics/Thesis/ShadyShenanigans/resources/dataset.csv", TEST_DATA_PATH, max_cache=150)
    # print(len(dataset))


if __name__ == '__main__':
    run()