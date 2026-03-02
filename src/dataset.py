import math

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from src.utils import compute_sun_features, get_tile_coordinates

TARGET_W, TARGET_H = 1892, 1903


# Center crop a 2D numpy array to (out_h, out_w).
def center_crop_to(arr, out_h, out_w):
    h, w = arr.shape
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return arr[y0:y0 + out_h, x0:x0 + out_w]


class DSMShadeDataset(Dataset):
    def __init__(self, index_csv, dsm_path, shade_path, tile_size=None, transforms=None):
        """
        index_csv: CSV with columns [dsm, shade, zenith, azimuth]
        tile_size: optional patch size (e.g. 512); if None, use full 1892x1903.
        transforms: optional extra transforms on tensors.
        """
        self.df = pd.read_csv(index_csv)
        self.dsm_path = dsm_path
        self.shade_path = shade_path
        self.tile_size = tile_size
        self.transforms = transforms

        self.sub_tiles_x = int(TARGET_W // (tile_size / 2)) if tile_size is not None else 1
        self.sub_tiles_y = int(TARGET_H // (tile_size / 2)) if tile_size is not None else 1

    def __len__(self):
        # When sub-tiling the larger tile, we want to ensure we reflect that in the length of the dataset
        return int(len(self.df) * (self.sub_tiles_x * self.sub_tiles_y))

    def _read_band(self, path):
        with rasterio.open(path) as src:
            arr = src.read(1)  # (H, W)
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        # Divide by sub-tile to get main tile from dataframe
        row = self.df.iloc[int(idx // (self.sub_tiles_x * self.sub_tiles_y))]
        dsm_path = f"{self.dsm_path}{row['dsm']}"
        shade_path = f"{self.shade_path}{row['tile']}/{row['shade_map']}"

        dsm_arr = self._read_band(dsm_path)  # (1992, 2003)
        shade_arr = self._read_band(shade_path)  # (1892, 1903)

        # center-crop DSM to shade size
        dsm_arr = center_crop_to(dsm_arr, TARGET_H, TARGET_W)  # (1892, 1903)

        # to tensors with channel dim
        dsm = torch.from_numpy(dsm_arr).unsqueeze(0)  # (1, 1892, 1903)
        shade = torch.from_numpy(shade_arr).unsqueeze(0)  # (1, 1892, 1903)

        # Normalize dsm to range 0-1
        dsmin = dsm.min()
        dsm = dsm - dsmin

        dsmax = dsm.max()
        dsm = dsm / dsmax

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

        # Todo: do we want transforms?
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
