import math

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from torchgeo.datasets import RasterDataset
import re

# Todo these are torchgeo datasets, probably will not support what I am trying to do, replace with custom solution using rasterio?
# class DSMDataset(RasterDataset):
#     filename_glob = "*.tif"
#     filename_regex = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_(?P<date>\d{4}_\d{2}_\d{2})_dsm.tif$"
#     date_format = "%Y_%m_%d"
#     is_image = True          # model input
#
# class ShadowDataset(RasterDataset):
#     filename_glob = "*.tif"
#     filename_regex = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_Shadow_(?P<date>\d{8}_\d{4})_LST.tif$"
#     date_format = "%Y%m%d_%H%M"
#     is_image = False         # ground truth
#
#     # Override as we do need float32 for this task (todo: right?)
#     def dtype(self) -> torch.dtype:
#         return torch.float32
#
# class ShadePredictionDataset(Dataset):
#     def __init__(self, base_ds, sun_table, tile_size=512):
#         self.base_ds = base_ds        # IntersectionDataset(DSM, Shadow)
#         # self.sun_table = sun_table    # e.g. dict[(scene_id, dt)] -> tensor(4,)
#         self.tile_size = tile_size
#
#     def __len__(self):
#         return len(self.base_ds)
#
#     def __getitem__(self, idx):
#         sample = self.base_ds[idx]
#
#         # TorchGeo returns dicts with 'image' and 'mask' keys
#         dsm = sample["image"]          # (C_dsm, H, W), likely (1, H, W)
#         shadow = sample["mask"]        # (C_shadow, H, W), likely (1, H, W)
#
#         # You need a way to identify the scene & datetime:
#         scene_id = sample["image"]["id"] if "id" in sample["image"] else None
#         dt_key = sample["image"]["date"] if "date" in sample["image"] else None
#
#         # sun_feat = self.sun_table[(scene_id, dt_key)]   # shape (4,)
#         H, W = dsm.shape[-2:]
#         # sun_map = sun_feat.view(-1, 1, 1).expand(-1, H, W)  # (4, H, W)
#
#         # x = torch.cat([dsm, sun_map], dim=0)  # (1+4, H, W)
#
#         return {
#             "input": dsm,              # to generator & discriminator
#             "target": shadow,        # shadow map
#             # "sun_feat": sun_feat,    # optional for logging / alt conditioning
#         }


TARGET_W, TARGET_H = 1892, 1903


# Center crop a 2D numpy array to (out_h, out_w).
def center_crop_to(arr, out_h, out_w):
    h, w = arr.shape
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return arr[y0:y0 + out_h, x0:x0 + out_w]


# Return 4D tensor [sin_az, cos_az, sin_el, cos_el]. (removing circular encoding of angles)
def compute_sun_features(zenith, azimuth):
    return torch.tensor([
        math.sin(azimuth), math.cos(azimuth),
        math.sin(zenith), math.cos(zenith)
    ], dtype=torch.float32)  # (4,)


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

    def __len__(self):
        return len(self.df)

    # Todo: testing
    def _read_band(self, path):
        # Todo: path isn't absolute, just filenames (pass root to dataset class?)
        with rasterio.open(path) as src:
            arr = src.read(1)  # (H, W)
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dsm_path =  f"{self.dsm_path}{row['dsm']}"
        shade_path = f"{self.shade_path}{row['tile']}/{row['shade_map']}"

        dsm_arr = self._read_band(dsm_path)  # (1992, 2003)
        shade_arr = self._read_band(shade_path)  # (1892, 1903)

        # center-crop DSM to shade size
        dsm_arr = center_crop_to(dsm_arr, TARGET_H, TARGET_W)

        # to tensors with channel dim
        dsm = torch.from_numpy(dsm_arr).unsqueeze(0)  # (1, H, W)
        shade = torch.from_numpy(shade_arr).unsqueeze(0)  # (1, H, W)

        # optional further random cropping to smaller tiles
        # Todo: deterministic cropping with overlaps?
        # Todo: other transforms? Rotations, scaling, flipping?
        # if self.tile_size is not None:
        #     th = tw = self.tile_size
        #     _, H, W = dsm.shape
        #     y0 = torch.randint(0, H - th + 1, ()).item()
        #     x0 = torch.randint(0, W - tw + 1, ()).item()
        #     dsm = dsm[:, y0:y0 + th, x0:x0 + tw]
        #     shade = shade[:, y0:y0 + th, x0:x0 + tw]

        # sun features from datetime
        sun_feat = compute_sun_features(row["zenith"], row["azimuth"])  # (4,)

        # broadcast to constant maps and concatenate with DSM
        C_sun = sun_feat.shape[0]
        _, H, W = dsm.shape
        sun_maps = sun_feat.view(C_sun, 1, 1).expand(C_sun, H, W)  # (4, H, W)
        x = torch.cat([dsm, sun_maps], dim=0)  # (1 + 4, H, W)

        if self.transforms is not None:
            x, shade = self.transforms(x, shade)

        return {
            "input": x,  # DSM + sun channels
            "target": shade,  # shade map
            # Todo: testing metadata, remove if not used
            "sun_feat": sun_feat,
            "dsm": row['dsm'],
            "shade": row['shade_map']
        }
