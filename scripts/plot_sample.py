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

from tqdm import tqdm

from cams.plots import plot_sample
from cams.sample import Sample
from cams.types import SpeciesNames

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
species: list[SpeciesNames] = ["CO", "NO2", "PM10", "PM2P5", "SO2", "O3"]
sample = Sample(
    date,
    models=[
        "CHIMERE",
        "MOCAGE",
        "MATCH",
        "MINNI",
        "MONARCH",
        "EURADIM",
        "GEMAQ",
        "SILAM",
        "DEHM",
        "LOTOS",
        "EMEP",
    ],
    lead_times=[15],
    species=species,
    levels=[0],
)
if not sample.is_valid:
    raise ValueError(f"Sample not valid: {sample}")

pbar = tqdm(species, desc="Plotting...")
for species_name in pbar:
    save_path = args.save_dir / f"{date.strftime('%Y-%m-%d_O3')}_{species_name}.png"
    plot_sample(
        sample=sample,
        save_path=save_path,
        species=species_name,
        lead_time=15,
        level=0,
    )
    pbar.write(f"{species_name} saved at {save_path}")
