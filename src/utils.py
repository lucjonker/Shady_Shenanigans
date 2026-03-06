import math
import os
import re
from datetime import datetime

import pandas as pd
import pytz
import rasterio
import torch
import torch.nn.functional as F
from pysolar.solar import get_altitude, get_azimuth
from rasterio.crs import CRS
from rasterio.warp import transform
from timezonefinder import TimezoneFinder
from torch import nn

DSM_REGEX = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_(?P<date>\d{4}_\d{2}_\d{2})_dsm.tif$"
SHADE_REGEX = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_Shadow_(?P<date>\d{8}_\d{4})_LST.tif$"


def get_location(path):
    with rasterio.open(path) as src:
        # center of raster in src crs
        x_center = (src.bounds.left + src.bounds.right) / 2.0
        y_center = (src.bounds.bottom + src.bounds.top) / 2.0

        # reproject that point to EPSG:4326
        lon, lat = transform(src.crs, CRS.from_string("EPSG:4326"), [x_center], [y_center])

        lat_center = float(lat[0])
        lon_center = float(lon[0])

        # Todo: do we need the altitude as well?
        return lat_center, lon_center


def get_regex_group(match, group_name):
    if match:
        group_value = match.group(group_name)
        return group_value
    return None


# Return 4D tensor [sin_az, cos_az, sin_el, cos_el]. (removing circular encoding of angles)
def compute_sun_features(zenith, azimuth):
    return torch.tensor([
        math.sin(azimuth), math.cos(azimuth),
        math.sin(zenith), math.cos(zenith)
    ], dtype=torch.float32)  # (4,)


def get_tile_coordinates(H, W, col: int, row: int, tile_size: int) -> tuple[int, int]:
    y0 = row
    x0 = col
    # Prevent losing data on extremes of tile
    if row + tile_size > H:
        y0 = H - tile_size
    if col + tile_size > W:
        x0 = W - tile_size
    return x0, y0


def write_dataset_csv(dsm_path, shade_map_path, csv_path, dsm_regex=DSM_REGEX, shade_regex=SHADE_REGEX):
    d = {'tile': [], 'dsm': [], 'shade_map': [], 'zenith': [], 'azimuth': []}
    tf = TimezoneFinder(in_memory=True)
    # For each tile DSM
    for dsm_filename in os.listdir(dsm_path):
        match = re.search(dsm_regex, dsm_filename)
        if not match:
            continue

        dsm_osmid = get_regex_group(match, 'osmid')
        dsm_tile_num = get_regex_group(match, 'tile')
        print(f"Writing entries for osmid: {dsm_osmid}", f"tile: {dsm_tile_num}...")

        # For each shade map corresponding to the same tile
        for shade_filename in os.listdir(shade_map_path + dsm_tile_num):
            match = re.search(shade_regex, shade_filename)
            if not match:
                continue

            # Get latitude and longitude
            lat, lon = get_location(shade_map_path + dsm_tile_num + "/" + shade_filename)

            # Get localized, daylight savings aware timezone
            tile_date = get_regex_group(match, 'date')
            dt = datetime.strptime(tile_date, '%Y%m%d_%H%M')
            tz = tf.timezone_at(lng=lon, lat=lat)
            a = pytz.timezone(tz)
            time = a.localize(dt, is_dst=False)

            # Calculate solar angles
            zenith = get_altitude(lat, lon, time)
            azimuth = get_azimuth(lat, lon, time)

            if zenith > 15:
                # Append row
                d['tile'].append(dsm_tile_num)
                d['dsm'].append(dsm_filename)
                d['shade_map'].append(shade_filename)
                d['zenith'].append(zenith)
                d['azimuth'].append(azimuth)
            else:
                print(f"Zenith {zenith} out of range")

    # Write data to csv
    df = pd.DataFrame(data=d)
    df.to_csv(csv_path, index=False)


# SOURCE https://github.com/chaddy1004/sobel-operator-pytorch
class Sobel(nn.Module):
    def __init__(self, device):
        super(Sobel, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.Tensor(kernel_h).unsqueeze(0).unsqueeze(0).to(device)
        kernel_v = torch.Tensor(kernel_v).unsqueeze(0).unsqueeze(0).to(device)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, img):
        x_v = F.conv2d(img, self.weight_v, padding=1)
        x_h = F.conv2d(img, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        return x
