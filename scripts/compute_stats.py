import datetime as dt
import json

import numpy as np
from tqdm import tqdm

from cams.sample import Sample
from cams.settings import CAMS_DATASET_DIR, STATS_PATH

# Get list of samples
list_runs = sorted(list(CAMS_DATASET_DIR.glob("input/*.netcdf")))
list_dates = [dt.datetime.strptime(path.stem, "%Y_%m_%d") for path in list_runs]
list_samples = [Sample(date_run, 15) for date_run in list_dates]
list_samples = [sample for sample in list_samples if sample.is_valid]

vmin, vmax = np.inf, -np.inf  # Init min and max

for sample in tqdm(list_samples, desc="Computing statistics"):
    data = sample.target_data.tensor[0].numpy()
    new_min, new_max = np.min(data), np.max(data)
    vmin = min(vmin, new_min)
    vmax = max(vmax, new_max)

stats = {"O3": {"min": float(vmin), "max": float(vmax)}}
print("stats : ", stats)

# Save stats as json
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=4)
