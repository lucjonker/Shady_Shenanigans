import rasterio
import torch
from torch.utils.data import Dataset
from torchgeo.datasets import RasterDataset
import re


# Todo figure it out
class DSMDataset(RasterDataset):
    filename_glob = "*.tif"
    filename_regex = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_(?P<date>\d{4}_\d{2}_\d{2})_dsm.tif$"
    date_format = "%Y_%m_%d"
    is_image = True          # model input

class ShadowDataset(RasterDataset):
    filename_glob = "*.tif"
    filename_regex = r"^(?P<osmid>\d+)_p_(?P<tile>\d+)_Shadow_(?P<date>\d{8}_\d{4})_LST.tif$"
    date_format = "%Y%m%d_%H%M"
    is_image = False         # ground truth

    # Override as we do need float32 for this task (todo: right?)
    def dtype(self) -> torch.dtype:
        return torch.float32

class ShadePredictionDataset(Dataset):
    def __init__(self, base_ds, sun_table, tile_size=512):
        self.base_ds = base_ds        # IntersectionDataset(DSM, Shadow)
        # self.sun_table = sun_table    # e.g. dict[(scene_id, dt)] -> tensor(4,)
        self.tile_size = tile_size

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]

        # TorchGeo returns dicts with 'image' and 'mask' keys
        dsm = sample["image"]          # (C_dsm, H, W), likely (1, H, W)
        shadow = sample["mask"]        # (C_shadow, H, W), likely (1, H, W)

        # You need a way to identify the scene & datetime:
        scene_id = sample["image"]["id"] if "id" in sample["image"] else None
        dt_key = sample["image"]["date"] if "date" in sample["image"] else None

        # sun_feat = self.sun_table[(scene_id, dt_key)]   # shape (4,)
        H, W = dsm.shape[-2:]
        # sun_map = sun_feat.view(-1, 1, 1).expand(-1, H, W)  # (4, H, W)

        # x = torch.cat([dsm, sun_map], dim=0)  # (1+4, H, W)

        return {
            "input": dsm,              # to generator & discriminator
            "target": shadow,        # shadow map
            # "sun_feat": sun_feat,    # optional for logging / alt conditioning
        }

#Todo: Is this the way to go?
class CustomGeoDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths  # List of file paths for geospatial data
        self.transform = transform  # Data augmentation/transformations

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Open the geospatial file using Rasterio
        with rasterio.open(self.file_paths[idx], 'r') as src:
            data = src.read()  # Read the data (e.g., satellite imagery)
            # Apply any preprocessing or transformations here
            if self.transform:
                data = self.transform(data)
        return data