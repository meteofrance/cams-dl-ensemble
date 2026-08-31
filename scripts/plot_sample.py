"""Plots a CAMS sample.

usage: plot_sample.py [-h] [--save_dir SAVE_DIR] date

positional arguments:
  date                 Date of the sample. Format: YYYY-MM-DD

options:
  --save_dir SAVE_DIR  Directory where the plot will be saved
"""

import argparse
import datetime as dt
from pathlib import Path

from cams.plots import plot_sample
from cams.sample import Sample

parser = argparse.ArgumentParser(description="Plots a CAMS sample.")
parser.add_argument(
    "date",
    type=str,
    help="Date of the sample. Format: YYYY-MM-DD",
)
parser.add_argument(
    "--save_dir",
    type=Path,
    default=Path("."),
    help="Directory where the plot will be saved",
    dest="save_dir",
)

args = parser.parse_args()

date = dt.datetime.strptime(args.date, "%Y-%m-%d")
sample = Sample(
    date,
    models=[
        "chimere",
        "mocage",
        "match",
        "minni",
        "monarch",
        "euradim",
        "gemaq",
        "silam",
        "dehm",
        "lotos",
    ],
    lead_times=[15],
    species=["O3"],
    levels=[0],
)
if not sample.is_valid:
    raise ValueError(f"Sample not valid: {sample}")

print(f"Plotting sample for {date}...")
save_path = args.save_dir / f"{date.strftime('%Y-%m-%d_O3')}.png"
plot_sample(sample, args.save_dir / f"{date.strftime('%Y-%m-%d_O3')}.png")
print(f"Plot saved at {save_path}")
