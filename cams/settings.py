import os
from pathlib import Path

# Define path to the cams dataset dir to the CAMS_DATASET_DIR
# environment variable if it exists, otherwise takes a default value.
CAMS_DATASET_DIR: Path = (
    Path("/scratch/shared/cams-dl-ensemble/o3_15h_SOL/")
    if (dataset_dir_from_env := os.getenv("CAMS_DATASET_DIR")) is None
    else Path(dataset_dir_from_env)
)

# Define paths to the CAMS dataset directory content
RAW_DATA_DIR: Path = CAMS_DATASET_DIR / "raw"
PROCESSED_DATA_DIR: Path = CAMS_DATASET_DIR / "processed"
STATS_PATH = CAMS_DATASET_DIR / "species_stats.json"

# Model names
MODEL_NAMES = [
    "MATCH",
    "MINNI",
    "CHIMERE",
    "MOCAGE",
    "MONARCH",
    "EURADIM",
    "EMEP",
    "GEMAQ",
    "SILAM",
    "DEHM",
    "LOTOS",
]


# Parameter name mapping
ECMWF_MF_PARAMETER_NAME_MAPPING: dict[str, str] = {
    "ozone": "O3",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2",
    "nitrogen_monoxide": "NO",
    "particulate_matter_2.5um": "PM25",
    "particulate_matter_10um": "PM10",
    "ammonia": "NH3",
    "non_methane_vocs": "NMVOC",
    "peroxyacyl_nitrates": "PANS",
    "secondary_inorganic_aerosol": "SIA",
    "dust": "DUST",
    "pm10_wildfires": "PM_WF",
    "total_elementary_carbon": "EC_TOT",
    "residential_elementary_carbon": "EC_RES",
    "formaldehyde": "HCHO",
    "glyoxal": "CHOCHO",
    "pm10_sea_salt_dry": "DYNSAL",
    "pm2.5_total_organic_matter": "PM25_OM",
    # "": "NO3_DRY",
    # "": "NH4_DRY",
    # "": "SO4_DRY",
}

# Physics constants
KILOGRAM_TO_MICROGRAM = 10**9

# Size of the working grid
SIZE_LAT = 420
SIZE_LON = 700

HAUTEUR_LEVELS = (50, 100, 250, 500, 750, 1000, 2000, 3000, 5000)
SOL_LEVELS = (0,)
