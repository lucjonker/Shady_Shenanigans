from datetime import datetime, timezone, timedelta

import pytz

from src.sun_position import sun_position
from suncalc import get_position, get_times
from src.utils import write_dataset_csv

#Define resources for training
TEST_TILE_PATH = "/Users/luc/Geomatics/Thesis/Data/input/271110_Ams/test_tiles"
TEST_SHADE_PATH = "/Users/luc/Geomatics/Thesis/Data/output/tree_shade/"
CSV_PATH = "/Users/luc/Geomatics/Thesis/ShadyShenanigans/resources/dataset.csv"

DSM_REGEX = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_(?P<date>\d{4}_\d{2}_\d{2})_dsm.tif$"
SHADE_REGEX = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_Shadow_(?P<date>\d{8}_\d{4})_LST.tif$"

def run():
    print("Generating dataset CSV")
    # write_dataset_csv(TEST_TILE_PATH, TEST_SHADE_PATH, DSM_REGEX, SHADE_REGEX, CSV_PATH)

    lon = 4.89944
    lat = 52.37565

    location = {
        'longitude': lon,
        'latitude' : lat,
        'altitude': 0
    }

    dt = datetime(2024, 8, 15, 7, 0, 0, tzinfo=timezone.utc)
    a = pytz.timezone("Europe/Amsterdam")
    b = dt.astimezone(a)
    print(dt, b)

    position = sun_position(b, location) #from sun_position.py
    print(position)

    pos = get_position(b, lon, lat) #from suncalc
    print(pos)

if __name__ == '__main__':
    run()
