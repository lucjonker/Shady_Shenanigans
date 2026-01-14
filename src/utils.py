import os
import re
from dataclasses import dataclass

import pytz
import rasterio
import pandas as pd
from rasterio.crs import CRS
from rasterio.warp import transform
from datetime import datetime, timezone

from src.sun_position import sun_position

@dataclass
class TimeStruct:
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    UTC: float  # offset in hours, can be fractional


def datetime_to_time_struct(dt: datetime) -> TimeStruct:
    if dt.tzinfo is None:
        # Interpret naive datetime as UTC (or change this to your local rule)
        offset_hours = 0
    else:
        offset = dt.utcoffset() or 0
        offset_hours = offset.total_seconds() / 3600.0

    return TimeStruct(
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        second=dt.second,
        UTC=offset_hours,
    )



def get_location(path):
    with rasterio.open(path) as src:
        # center of raster in src crs
        x_center = (src.bounds.left + src.bounds.right) / 2.0
        y_center = (src.bounds.bottom + src.bounds.top) / 2.0

        # reproject that point to EPSG:4326
        lon, lat = transform(src.crs, CRS.from_string("EPSG:4326"), [x_center], [y_center])

        lat_center = float(lat[0])
        lon_center = float(lon[0])

        return {'latitude': lat_center, 'longitude': lon_center, 'altitude': 0}


def get_regex_group(match, group_name):
    if match:
        group_value = match.group(group_name)
        return group_value
    return None


def write_dataset_csv(dsm_path, shade_map_path, dsm_regex, shade_regex, csv_path):
    d = {'dsm': [], 'shade_map': [], 'zenith': [], 'azimuth': []}
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

            tile_date = get_regex_group(match, 'date')
            dt = datetime.strptime(tile_date, '%Y%m%d_%H%M')
            dt.replace(tzinfo=timezone.utc)
            tz = pytz.timezone('Europe/Amsterdam')
            tz_dt = dt.astimezone(tz)
            time = datetime_to_time_struct(tz_dt)
            location = get_location(shade_map_path + dsm_tile_num + "/" + shade_filename)

            # Todo: Investigate inconsistent results
            sun = sun_position(time, location)

            # Append row
            d['dsm'].append(dsm_filename)
            d['shade_map'].append(shade_filename)
            d['zenith'].append(sun['zenith'][0])
            d['azimuth'].append(sun['azimuth'][0])

    # Write data to csv
    df = pd.DataFrame(data=d)
    df.to_csv(csv_path, index=False)
