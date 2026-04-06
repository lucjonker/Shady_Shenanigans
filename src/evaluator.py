import lightning as L
from lightning.fabric.utilities import AttributeDict
from rasterio.windows import Window

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

    def generate_output(self, dsm_path, unaware_date_time, overlap, strategy, crop, tf):
        tile_size = 512
        stride = tile_size - overlap

        with rasterio.open(dsm_path) as src:
            lat, lon = get_lat_lon(src)
            date_time = get_tz_aware_dt(lat, lon, unaware_date_time, tf)

            # Calculate solar angles
            zenith = get_altitude(lat, lon, date_time)
            azimuth = get_azimuth(lat, lon, date_time)

            if zenith < 15:
                return None, None, None

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
                res = np.divide(out_acc, out_hits, dtype=np.float32)
            if strategy == "max":
                res = out_max
            if crop:
                res = res[50:H - 50, 50:W - 50]
                xsize, ysize = W - 100, H - 100
                xoff, yoff = 50, 50
                window = Window(xoff, yoff, xsize, ysize)
                window_transform = src.window_transform(window)
                profile.update({
                    'height': ysize,
                    'width': xsize,
                    'transform': window_transform})

            return profile, res, zenith

    def generate_and_write_output(self, dsm_path, unaware_date_time, overlap, strategy="max", filename="result",
                                  crop=False):
        tf = TimezoneFinder(in_memory=True)
        profile, res, _ = self.generate_output(dsm_path, unaware_date_time, overlap, strategy, crop, tf)
        self.write_output_to_geotiff(filename, profile, res)

    def write_output_to_geotiff(self, filename: str, profile, res):
        # Write out
        result_path = self.results_dir + f"/{filename}.tif"
        with rasterio.open(result_path, "w", **profile) as dst:
            dst.write(res, 1)

    def write_metrics_csv(self, data_path, csv_path, dsm_regex=DSM_REGEX, shade_regex=SHADE_REGEX):
        d = {'osmid': [], 'tile': [], 'date_time': [], 'RMSE': [], 'SSIM': [], 'MAE': []}
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

                print(f"Writing for tile: {dsm_tile_num}...")
                num_skipped = 0
                # For each shade map corresponding to the same tile
                for shade_filename in os.listdir(f"{data_path}{city_filename}/targets/{dsm_tile_num}"):
                    match = re.search(shade_regex, shade_filename)
                    if not match:
                        continue

                    shade_path = f"{data_path}{city_filename}/targets/{dsm_tile_num}/{shade_filename}"
                    # Get timezone
                    tile_date = get_regex_group(match, 'date')
                    unaware_date_time = datetime.strptime(tile_date, '%Y%m%d_%H%M')
                    # Todo, test different overlaps and strategies?
                    _, res, zenith = self.generate_output(dsm_path, unaware_date_time, 0, "max", True, tf)

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
                    else:
                        num_skipped += 1

                print(f"Skipped {num_skipped} due to zenith < 15 degrees")

        df_to_csv(csv_path, "metrics.csv", d)

    def prepare_tile(self, H, W, azimuth: float, col: int, row: int, src, tile_size: int, zenith: float):
        x0, y0 = get_tile_coordinates(H, W, col, row, tile_size)

        # Read window
        window = Window(col_off=x0, row_off=y0, width=tile_size, height=tile_size)
        tile = src.read(window=window)

        dsm = torch.from_numpy(tile.astype(np.float32))

        # Normalize dsm to the highest point in the training dataset (so that input is still correct relatively)
        dsmin = dsm.min()
        dsm = dsm - dsmin
        dsm = dsm / self.training_max

        sun_feat = compute_sun_features(zenith, azimuth)  # (4,)

        # broadcast to constant maps and concatenate with DSM
        sun_shape = sun_feat.shape[0]
        sun_maps = sun_feat.view(sun_shape, 1, 1).expand(sun_shape, tile_size, tile_size)  # (4, H, W)
        x = torch.cat([dsm, sun_maps], dim=0)  # (1 + 4, H, W)
        x = x.unsqueeze(0)  # (1, 1 + 4, H, W)
        return x, x0, y0
