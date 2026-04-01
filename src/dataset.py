import math

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from utils import compute_sun_features, get_tile_coordinates


# Center crop a 2D numpy array to (out_h, out_w).
def center_crop_to(arr, out_h, out_w):
    h, w = arr.shape
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return arr[y0:y0 + out_h, x0:x0 + out_w]


class DSMShadeDataset(Dataset):
    def __init__(self, index_csv, data_path, max_cache=200, tile_size=512, transforms=None):
        self.df = pd.read_csv(index_csv)
        self.data_path = data_path
        # Note tile size is flexible with this class but the associated model assumes 512x512 tiles
        self.tile_size = tile_size
        # Todo: if extra time, create sample index to avoid hardcoded subtile numbers
        self.sub_tiles_x, self.sub_tiles_y = 7, 7
        self.training_max = np.max(self.df['maximum'])
        self.transforms = transforms

        self.dsm_cache = {}
        self.shade_cache = {}
        self.max_cache = max_cache

    def __len__(self):
        # When sub-tiling the larger tile reflect in the length of the dataset
        return int(len(self.df) * (self.sub_tiles_x * self.sub_tiles_y))

    def read_raster_data(self, path, cache):
        if cache.get(path) is not None:
            return cache[path]

        with rasterio.open(path) as src:
            arr = src.read(1)  # (H, W)
        fl32arr = arr.astype(np.float32)

        if len(cache) > self.max_cache:
            cache.popitem()
        cache[path] = fl32arr
        return fl32arr

    def __getitem__(self, idx):
        # Divide by sub-tile to get main tile from dataframe
        row = self.df.iloc[int(idx // (self.sub_tiles_x * self.sub_tiles_y))]
        root = f"{self.data_path}/{row['osmid']}"
        dsm_path = f"{root}/input/{row['dsm']}"
        shade_path = f"{root}/targets/{row['tile']}/{row['shade_map']}"

        dsm_arr = self.read_raster_data(dsm_path, self.dsm_cache)  # (W, H)
        shade_arr = self.read_raster_data(shade_path, self.shade_cache)  # (W-, H-)
        cropto = shade_arr.shape

        # center-crop DSM to shade size
        dsm_arr = center_crop_to(dsm_arr, cropto[0], cropto[1])  # (W-, H-)

        # to tensors with channel dim
        dsm = torch.from_numpy(dsm_arr).unsqueeze(0)  # (W-, H-)
        shade = torch.from_numpy(shade_arr).unsqueeze(0)  # (W-, H-)

        # Normalize dsm to the highest point in the dataset
        dsmin = dsm.min()
        dsm = dsm - dsmin
        dsm = dsm / self.training_max

        # Sub-tile from data
        if self.tile_size is not None:
            tile_id = idx % (self.sub_tiles_x * self.sub_tiles_y)
            row_offset = int(tile_id // self.sub_tiles_y)
            column_offset = int(tile_id % self.sub_tiles_x)
            offset = int(self.tile_size / 2)

            _, H, W = dsm.shape
            x0, y0 = get_tile_coordinates(H, W, int(column_offset * offset), int(row_offset * offset), self.tile_size)

            dsm = dsm[:, y0:y0 + self.tile_size, x0:x0 + self.tile_size]
            shade = shade[:, y0:y0 + self.tile_size, x0:x0 + self.tile_size]
            # print(x0, y0, dsm.shape)

        # sun features from datetime
        sun_feat = compute_sun_features(row["zenith"], row["azimuth"])  # (4,)

        # broadcast to constant maps and concatenate with DSM
        sun_shape = sun_feat.shape[0]
        _, H, W = dsm.shape
        sun_maps = sun_feat.view(sun_shape, 1, 1).expand(sun_shape, H, W)  # (4, H, W)
        x = torch.cat([dsm, sun_maps], dim=0)  # (1 + 4, H, W)

        # Transforms optional
        if self.transforms is not None:
            x, shade = self.transforms(x, shade)

        return {
            "source": x,  # DSM + sun channels
            "target": shade,  # shade map
            # Todo: testing metadata, remove if not used
            # "sun_feat": sun_feat,
            # "dsm_id": row['dsm'],
            # "shade_id": row['shade_map'],
            # "subtile": idx % (self.sub_tiles_x * self.sub_tiles_y)
        }
