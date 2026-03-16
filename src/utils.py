import math
import os
import re
import sys
from datetime import datetime
from os.path import dirname

import pandas as pd
import pytz
import rasterio
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
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


def write_dataset_csv(data_path, csv_path, dsm_regex=DSM_REGEX, shade_regex=SHADE_REGEX):
    d = {'osmid': [], 'tile': [], 'dsm': [], 'shade_map': [], 'zenith': [], 'azimuth': []}
    tf = TimezoneFinder(in_memory=True)
    # For each cities' data
    for city_filename in os.listdir(data_path):
        # Skip mac ds store
        if city_filename == ".DS_Store":
            continue
        print(f"Processing city with osmid: {city_filename}")
        # For each dsm within the city
        for dsm_filename in os.listdir(f"{data_path}{city_filename}/input"):
            match = re.search(dsm_regex, dsm_filename)
            if not match:
                continue

            dsm_osmid = get_regex_group(match, 'osmid')
            dsm_tile_num = get_regex_group(match, 'tile')
            print(f"Writing for tile: {dsm_tile_num}...")

            # For each shade map corresponding to the same tile
            for shade_filename in os.listdir(f"{data_path}{city_filename}/targets/{dsm_tile_num}"):
                match = re.search(shade_regex, shade_filename)
                if not match:
                    continue

                # Get latitude and longitude
                lat, lon = get_location(f"{data_path}{city_filename}/targets/{dsm_tile_num}/{shade_filename}")

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
                    d["osmid"].append(dsm_osmid)
                    d['tile'].append(dsm_tile_num)
                    d['dsm'].append(dsm_filename)
                    d['shade_map'].append(shade_filename)
                    d['zenith'].append(zenith)
                    d['azimuth'].append(azimuth)
                else:
                    print(f"Zenith {zenith} out of range")

    df_to_csv(csv_path, d)


def df_to_csv(csv_root: str, csv_name: str, d: dict):
    # Write data to csv
    df = pd.DataFrame(data=d)
    csv_path = os.path.join(csv_root, csv_name)
    df.to_csv(csv_path, index=False)


def plot_losses(g_losses, d_losses, title="Loss Analysis:"):
    plt.figure(figsize=(10, 5))
    plt.suptitle(title, fontsize=16)

    plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator Loss")
    plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss Over Epochs")
    plt.legend()
    plt.show()


# Plots result and target images assuming they come from the gpu
def display_res(source, result, target):
    source_cpu = source.cpu()
    result_cpu = result.cpu().detach()
    target_cpu = target.cpu().detach()

    plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(1, 3, 1)
    result = source_cpu[0][0]
    plt.imshow(result.numpy())
    ax1.title.set_text("Source")
    plt.axis('off')

    ax2 = plt.subplot(1, 3, 2)
    result = result_cpu[0]
    plt.imshow(result.squeeze().numpy())
    ax2.title.set_text("Model Result")
    plt.axis('off')

    ax3 = plt.subplot(1, 3, 3)
    result = target_cpu[0]
    plt.imshow(result.squeeze().numpy())
    ax3.title.set_text("Target Result")
    plt.axis('off')

    plt.set_cmap("viridis")
    plt.show()


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
