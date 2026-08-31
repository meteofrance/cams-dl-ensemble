from typing import Literal

STATISTICS_NAMES = (
    "mean",
    "amin",
    "argmin",
    "amax",
    "argmax",
    "median",
    "skew",
    "kurtosis",
    "std",
)

StatisticsNames = Literal[
    "mean",
    "amin",
    "argmin",
    "amax",
    "argmax",
    "median",
    "skew",
    "kurtosis",
    "std",
]

MODELS_NAMES = (
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
)

ModelsNames = Literal[
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

AVAILABLE_SPECIES = ("CO", "NO2", "O3", "PM10", "PM2P5", "SO2")

AvailableSpecies = Literal["CO", "NO2", "O3", "PM10", "PM2P5", "SO2"]
