import os
import re

import pytz
import rasterio
import pandas as pd
from rasterio.crs import CRS
from rasterio.warp import transform
from pysolar.solar import get_altitude, get_azimuth
from timezonefinder import TimezoneFinder
from datetime import datetime


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


def write_dataset_csv(dsm_path, shade_map_path, dsm_regex, shade_regex, csv_path):
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

            # Append row
            d['tile'].append(dsm_tile_num)
            d['dsm'].append(dsm_filename)
            d['shade_map'].append(shade_filename)
            d['zenith'].append(zenith)
            d['azimuth'].append(azimuth)

    # Write data to csv
    df = pd.DataFrame(data=d)
    df.to_csv(csv_path, index=False)
