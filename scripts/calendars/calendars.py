import datetime as dt
from pathlib import Path

from calendardataviz import InspectorABC, RichString, start_app
from calendardataviz.file_count_inspector import FileCountInspector
from calendardataviz.colors import JET, RDYLGN, color_from_pct, COOLWARM
from typing_extensions import override

from cams.settings import CAMS_DATASET_DIR


if __name__ == "__main__":
    start_app(
        inspector=[
            FileCountInspector(
                name="Input file count",
                root_dir=CAMS_DATASET_DIR / "median_dataset" / "ensemble",
                glob_pattern=r"%Y_%m_%d*.netcdf",
                nb_expected_files_per_day=10,
                colors=RDYLGN,
            ),
            FileCountInspector(
                name="Input failed file count",
                root_dir=CAMS_DATASET_DIR / "median_dataset" / "ensemble",
                glob_pattern=r"%Y_%m_%d*.fail",
                nb_expected_files_per_day=10,
                colors=COOLWARM,
            ),
            FileCountInspector(
                name="Input zip file count",
                root_dir=CAMS_DATASET_DIR / "median_dataset" / "ensemble",
                glob_pattern=r"%Y_%m_%d*.zip",
                nb_expected_files_per_day=10,
                colors=COOLWARM,
            ),
            FileCountInspector(
                name="Target file count",
                root_dir=CAMS_DATASET_DIR / "median_dataset" / "reanalysis",
                glob_pattern=r"%Y-%m-01*.netcdf",
                nb_expected_files_per_day=10 * 19,
                colors=RDYLGN,
            ),
        ],
        years=[2023, 2024, 2025, 2026],
        nb_processes=8,
    )
