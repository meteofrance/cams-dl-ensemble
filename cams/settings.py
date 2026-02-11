from pathlib import Path

CAMS_DATA_DIR = Path("/scratch/shared/cams/")
CAMS_DATASET_DIR = CAMS_DATA_DIR / "o3_15h_SOL/"
STATS_PATH = Path("cams/species_stats.json")

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
