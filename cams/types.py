from typing import Literal

StatisticsNames = Literal[
    "mean",
    "amin",
    "argmin",
    "amax",
    "argmax",
    "median",
    "skew",
    "kurtosis",
]

STATISTICS_NAMES: set[StatisticsNames] = (
    "mean",
    "amin",
    "argmin",
    "amax",
    "argmax",
    "median",
    "skew",
    "kurtosis",
)
