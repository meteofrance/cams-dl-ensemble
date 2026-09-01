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

MODELS_NAMES: list[ModelsNames] = [
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

AvailableSpecies = Literal["CO", "NO2", "O3", "PM10", "PM2P5", "SO2"]

AVAILABLE_SPECIES: list[AvailableSpecies] = ["CO", "NO2", "O3", "PM10", "PM2P5", "SO2"]
