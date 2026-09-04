import datetime as dt
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import Hashable, Literal, TypedDict

import xarray as xr

from cams.settings import RAW_DATA_DIR


# Path to cached overall species means (JSON)
OVERALL_MEANS_PATH = Path(__file__).with_name("stats.json")


class Stats(TypedDict):
    mean: float

class ModelStats(TypedDict):
    co_conc: Stats
    no2_conc: Stats
    o3_conc: Stats
    pm10_conc: Stats
    pm2p5_conc: Stats
    so2_conc: Stats

class DateStats(TypedDict):
    chimere: ModelStats
    dehm: ModelStats
    emep: ModelStats
    euradim: ModelStats
    gemaq: ModelStats
    lotos: ModelStats
    match: ModelStats
    minni: ModelStats
    mocage: ModelStats
    monarch: ModelStats
    reanalysis: ModelStats
    silam: ModelStats
    weighted_ensemble: ModelStats


def compute_stats_for_date(date: dt.date) -> DateStats:
    """Iterate over all NetCDF files for the given date and compute stats
     value per model and per species.

    Args:
        date: Date for wich to compute stats.
    
    Returs:
        DateStats: Stats for the given date.
    """
