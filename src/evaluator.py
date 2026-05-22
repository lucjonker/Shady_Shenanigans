import time

import lightning as L
import numpy as np
from lightning.fabric.utilities import AttributeDict
from rasterio.windows import Window
from tqdm import tqdm

from src.loss_functions import ssim, l1_loss, rmse
from src.model import ShadyModel
from src.utils import *


class ShadeEvaluator:
    def __init__(self, device, checkpoint_path, results_dir):
        self.fabric = L.Fabric(accelerator=str(device))
        self.device = device
        self.generator = None
        self.training_max = None
        self.init_model(checkpoint_path)
        self.results_dir = results_dir

    def init_model(self, checkpoint_path):
        self.fabric.launch()
        self.fabric.seed_everything(42)
        with self.fabric.init_module():
            model = ShadyModel()
            model.eval()
        self.generator = self.fabric.setup(model.generator)
        state = AttributeDict(generator=self.generator, train_height_max=self.training_max)
        self.fabric.load(checkpoint_path, state, weights_only=False)
        # Force state loading
        self.training_max = state.train_height_max
        print("Model initialized...")

    # Given a dsm, tree mask, and desired date + time, generates model output and combines with a given overlap and strategy
    # Note: strategies max and mean are not recommended for use
    def generate_output(self, dsm_path, tree_mask_path, unaware_date_time, overlap, strategy, crop, tf,
                        measure_runtime=False, building_only=False):
        tile_size = 512
        stride = tile_size - (overlap * 2)

        with rasterio.open(dsm_path) as dsm_src:
            with rasterio.open(tree_mask_path) as tree_mask:
                lat, lon = get_lat_lon(dsm_src)
                date_time = get_tz_aware_dt(lat, lon, unaware_date_time, tf)

                # Calculate solar angles
                zenith = get_altitude(lat, lon, date_time)
                azimuth = get_azimuth(lat, lon, date_time)

                if zenith <= 15:
                    return None, None, None, None

                H, W = dsm_src.height, dsm_src.width
                profile = dsm_src.profile.copy()
                profile.update(dtype=rasterio.float32, count=1)

                out_max = None
                out_acc = None
                out_hits = None
                if strategy == "max":
                    out_max = np.full((H, W), -np.inf, dtype=np.float32)
                if strategy == "mean":
                    out_acc = np.zeros((H, W), dtype=np.float32)
                    out_hits = np.zeros((H, W), dtype=np.float32)
                if strategy == "stitch":
                    out_acc = np.zeros((H, W), dtype=np.float32)

                # Iterating over input image
                if measure_runtime:
                    start_time = time.time()

                for row in range(0, H, stride):
                    for col in range(0, W, stride):
                        # Prep tile
                        x, x0, y0 = self.prepare_tile(H, W, azimuth, col, row, dsm_src, tree_mask, tile_size, zenith,
                                                      building_only)

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
                        if strategy == "stitch":
                            row_start = (y0 + overlap if y0 > 0 else y0)
                            row_end = (y0 + tile_size - overlap if y0 + tile_size < H else H)
                            col_start = (x0 + overlap if x0 > 0 else x0)
                            col_end = (x0 + tile_size - overlap if x0 + tile_size < W else W)
                            out_acc[row_start:row_end, col_start:col_end] = pred_np[
                                (overlap if y0 > 0 else 0):(tile_size - overlap if y0 + tile_size < H else tile_size),
                                (overlap if x0 > 0 else 0):(tile_size - overlap if x0 + tile_size < W else tile_size)]

            runtime = None
            if measure_runtime:
                runtime = time.time() - start_time

            res = None
            # Average results
            if strategy == "mean":
                res = np.divide(out_acc, out_hits, dtype=np.float32)
            if strategy == "max":
                res = out_max
            if strategy == "stitch":
                res = out_acc
            if crop and not building_only:
                # All rasters generated by lbeuster approach have a 50 pixel margin
                res = res[50:H - 50, 50:W - 50]
                xsize, ysize = W - 100, H - 100
                xoff, yoff = 50, 50
                window = Window(xoff, yoff, xsize, ysize)
                window_transform = dsm_src.window_transform(window)
                profile.update({
                    'height': ysize,
                    'width': xsize,
                    'transform': window_transform})

            return profile, res, zenith, runtime

    def generate_and_write_output(self, dsm_path, tree_mask_path, unaware_date_time, overlap, strategy="stitch",
                                  filename="result", crop=False, building_only=False):
        tf = TimezoneFinder(in_memory=True)
        profile, res, _, _ = self.generate_output(dsm_path, tree_mask_path, unaware_date_time, overlap, strategy, crop,
                                                  tf, building_only)
        # Write out
        result_path = self.results_dir + f"/{filename}.tif"
        with rasterio.open(result_path, "w", **profile) as dst:
            dst.write(res, 1)

    def write_metrics_csv(self, data_path, csv_path, filename="metrics", dsm_regex=DSM_REGEX, shade_regex=SHADE_REGEX,
                          overlap=0, strategy="stitch", building_only=False):
        d = {'osmid': [], 'tile': [], 'date_time': [], 'RMSE': [], 'SSIM': [], 'MAE': [], 'runtime': []}
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
                dsm_path = f"{data_path}{city_filename}/input/{dsm_filename}"
                dsm_date = get_regex_group(match, 'date')

                mask_path = f"{data_path}{city_filename}/masks/{dsm_osmid}_p_{dsm_tile_num}_{dsm_date}_rgb_segmented.tif"

                print(f"Writing for tile: {dsm_tile_num}...")
                num_skipped = 0
                # For each shade map corresponding to the same tile
                for shade_filename in tqdm(os.listdir(f"{data_path}{city_filename}/targets/{dsm_tile_num}")):
                    match = re.search(shade_regex, shade_filename)
                    if not match:
                        continue

                    shade_path = f"{data_path}{city_filename}/targets/{dsm_tile_num}/{shade_filename}"
                    # Get timezone
                    tile_date = get_regex_group(match, 'date')
                    unaware_date_time = datetime.strptime(tile_date, '%Y%m%d_%H%M')
                    profile, res, zenith, runtime = self.generate_output(dsm_path, mask_path, unaware_date_time,
                                                                         overlap, strategy, not building_only, tf, True)

                    # Skipped if zenith is < 15 degrees
                    if zenith is not None:
                        with rasterio.open(shade_path) as src:
                            arr = src.read(1)  # (H, W)
                        fl32arr = arr.astype(np.float32)

                        # Generate evaluation metrics
                        target = torch.from_numpy(fl32arr).unsqueeze(0).unsqueeze(0).contiguous()
                        generated = torch.from_numpy(res).unsqueeze(0).unsqueeze(0).contiguous()
                        rmse_value = rmse(target, generated).item()
                        ssim_value = ssim(target, generated).item()
                        mae_value = l1_loss(target, generated).item()

                        # Append a row for each sample
                        d["osmid"].append(dsm_osmid)
                        d['tile'].append(dsm_tile_num)
                        d['date_time'].append(unaware_date_time)
                        d['RMSE'].append(rmse_value)
                        d['SSIM'].append(ssim_value)
                        d['MAE'].append(mae_value)
                        d['runtime'].append(runtime)
                    else:
                        num_skipped += 1

                print(f"Skipped {num_skipped} due to zenith < 15 degrees")

        df_to_csv(csv_path, f"{filename}.csv", d)

    def generate_outputs(self, data_path, dsm_regex=DSM_REGEX, shade_regex=SHADE_REGEX, overlap=64, strategy="stitch",
                         building_only=False):
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
                dsm_path = f"{data_path}{city_filename}/input/{dsm_filename}"
                dsm_date = get_regex_group(match, 'date')

                mask_path = f"{data_path}{city_filename}/masks/{dsm_osmid}_p_{dsm_tile_num}_{dsm_date}_rgb_segmented.tif"

                print(f"Writing for tile: {dsm_tile_num}...")
                num_skipped = 0
                # For each shade map corresponding to the same tile
                for shade_filename in os.listdir(f"{data_path}{city_filename}/targets/{dsm_tile_num}"):
                    match = re.search(shade_regex, shade_filename)
                    if not match:
                        continue

                    # Get timezone
                    tile_date = get_regex_group(match, 'date')
                    unaware_date_time = datetime.strptime(tile_date, '%Y%m%d_%H%M')
                    profile, res, zenith, runtime = self.generate_output(dsm_path, mask_path, unaware_date_time,
                                                                         overlap, strategy, not building_only, tf,
                                                                         building_only)

                    # Skipped if zenith is < 15 degrees
                    if zenith is not None:
                        result_dir = self.results_dir + f"/{dsm_tile_num}"
                        if not os.path.exists(result_dir):
                            # Create the directory
                            os.makedirs(result_dir)
                        result_path = result_dir + f"/{dsm_osmid}_p_{dsm_tile_num}_{tile_date}_model.tif"
                        with rasterio.open(result_path, "w", **profile) as dst:
                            dst.write(res, 1)
                    else:
                        num_skipped += 1

                print(f"Skipped {num_skipped} due to zenith < 15 degrees")

    def prepare_tile(self, H, W, azimuth: float, col: int, row: int, dsm_src, tree_mask, tile_size: int, zenith: float,
                     building_only: bool):
        x0, y0 = get_tile_coordinates(H, W, col, row, tile_size)

        # Read window
        window = Window(col_off=x0, row_off=y0, width=tile_size, height=tile_size)
        d_tile = dsm_src.read(window=window)
        t_tile = tree_mask.read(window=window)

        dsm = torch.from_numpy(d_tile.astype(np.float32))
        mask = torch.from_numpy(t_tile.astype(np.float32))
        if building_only:
            mask = np.zeros_like(mask, dtype=np.float32)

        # Normalize dsm to the highest point in the training dataset (so that input is still correct relative to what model understands)
        # Todo: does this cause problems if the dsm contains a higher max value than the model has seen?
        # Todo: May need to have some kind of check if the max of this dsm exceeds the training max
        dsmin = dsm.min()
        dsm = dsm - dsmin
        dsm = dsm / self.training_max

        if not building_only:
            mask = mask / mask.max()  # Normalize to [0,1] instead of [0,255]

        sun_feat = compute_sun_features(zenith, azimuth)  # (4,)

        # broadcast to constant maps and concatenate with DSM
        sun_shape = sun_feat.shape[0]
        sun_maps = sun_feat.view(sun_shape, 1, 1).expand(sun_shape, tile_size, tile_size)  # (4, H, W)
        x = torch.cat([dsm, mask, sun_maps], dim=0)  # (1 + 4, H, W)
        x = x.unsqueeze(0)  # (1, 1 + 4, H, W)
        return x, x0, y0
