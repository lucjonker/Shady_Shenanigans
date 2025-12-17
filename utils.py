import re

import rasterio
from rasterio.warp import transform

# Todo: integrate as feature?
def get_latitude(path):
    with rasterio.open(path) as src:
        # 1) center of raster in src crs
        x_center = (src.bounds.left + src.bounds.right) / 2.0
        y_center = (src.bounds.bottom + src.bounds.top) / 2.0

        # 2) reproject that point to EPSG:4326
        lon, lat = transform(src.crs, "EPSG:4326", [x_center], [y_center])

    lat_center = float(lat[0])
    return lat_center


def get_group(string, regex, group_name):
    match = re.search(regex, string)
    match.group(group_name)

# Todo: JSON writer for dataset loading?
