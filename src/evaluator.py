import numpy as np
import rasterio
import torch
import lightning as L
from pysolar.solar import get_altitude, get_azimuth
from rasterio.windows import Window

from src.model import ShadyModel
from lightning.fabric.utilities import AttributeDict
from src.utils import get_location, compute_sun_features, get_tile_coordinates


class ShadeEvaluator:
    def __init__(self, device, checkpoint_path, results_dir):
        self.fabric = L.Fabric(accelerator=str(device))
        self.device = device
        self.generator = None
        self.init_model(checkpoint_path)
        self.results_dir = results_dir

    def init_model(self, checkpoint_path):
        self.fabric.launch()
        self.fabric.seed_everything(42)
        with self.fabric.init_module():
            model = ShadyModel()
            model.eval()
        self.generator = self.fabric.setup(model.generator)
        state = AttributeDict(generator=self.generator)
        self.fabric.load(checkpoint_path, state)
        print("Model initialized...")

    # Todo: save multiple outputs
    # Todo: visualize outputs?
    # Todo: optional clipping of the output
    def evaluate(self, dsm_path, date_time, overlap, strategy="max"):
        # Get latitude and longitude
        lat, lon = get_location(dsm_path)

        # Calculate solar angles
        zenith = get_altitude(lat, lon, date_time)
        azimuth = get_azimuth(lat, lon, date_time)

        tile_size = 512
        stride = tile_size - overlap

        with rasterio.open(dsm_path) as src:
            H, W = src.height, src.width
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, count=1)

            out_max = None
            out_acc = None
            out_hits = None
            if strategy == "max":
                out_max = np.full((H, W), -np.inf, dtype=np.float32)
            if strategy == "mean":
                out_acc = np.zeros((H, W), dtype=np.float32)
                out_hits = np.zeros((H, W), dtype=np.float32)

            # Iterating over input image
            for row in range(0, H, stride):
                for col in range(0, W, stride):
                    # Prep tile
                    x, x0, y0 = self.prepare_tile(H, W, azimuth, col, row, src, tile_size, zenith)

                    # Perform prediction
                    with torch.no_grad():
                        x = x.to(self.device)
                        pred = self.generator(x)

                    # Removing batch dimension
                    pred = pred.squeeze(0)

                    pred_np = pred.detach().cpu().numpy().astype(np.float32).squeeze(0)

                    # Take max of predicted values
                    if strategy == "max":
                        patch = out_max[y0:y0 + tile_size, x0:x0 + tile_size]
                        out_max[y0:y0 + tile_size, x0:x0 + tile_size] = np.maximum(patch, pred_np)
                    if strategy == "mean":
                        out_acc[y0:y0 + tile_size, x0:x0 + tile_size] += pred_np
                        out_hits[y0:y0 + tile_size, x0:x0 + tile_size] += float(1.0)

                res = None
                # Average results
                if strategy == "mean":
                    # out_hits[out_hits == 0] = 1
                    res = np.divide(out_acc, out_hits, dtype=np.float32)
                if strategy == "max":
                    res = out_max

                # Write out
                with rasterio.open(self.results_dir + "/result.tif", "w", **profile) as dst:
                    dst.write(res, 1)

    def prepare_tile(self, H, W, azimuth: float, col: int, row: int, src, tile_size: int, zenith: float):
        x0, y0 = get_tile_coordinates(H, W, col, row, tile_size)

        # Read window
        window = Window(col_off=x0, row_off=y0, width=tile_size, height=tile_size)
        tile = src.read(window=window)

        dsm = torch.from_numpy(tile.astype(np.float32))
        # Normalize dsm to range 0-1
        dsmin = dsm.min()
        dsm = dsm - dsmin

        dsmax = dsm.max()
        dsm = dsm / dsmax

        sun_feat = compute_sun_features(zenith, azimuth)  # (4,)

        # broadcast to constant maps and concatenate with DSM
        sun_shape = sun_feat.shape[0]
        sun_maps = sun_feat.view(sun_shape, 1, 1).expand(sun_shape, tile_size, tile_size)  # (4, H, W)
        x = torch.cat([dsm, sun_maps], dim=0)  # (1 + 4, H, W)
        x = x.unsqueeze(0)  # (1, 1 + 4, H, W)
        return x, x0, y0
