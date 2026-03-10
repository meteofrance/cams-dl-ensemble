"""Scripts that validates a CAMS dataset. Checks:
- The dataset directory hierarchy.
- Coordinates.
- Units.
- Every input has a target.

Usage:
```bash
python scripts/data/1_validation.py \
    -d --dataset_dir  # Path to the dataset dir. Default value in settings.py
    -o --output       # Path where the validation plot is saved.
```

The script `scripts/data/2_compute_stats.py` should be executed after this one.
"""

import datetime as dt
import math
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

from cams.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_NAMES


def validate(raw_dir: Path, processed_dir: Path, plot_save_path: Path) -> None:
    """Validates a CAMS dataset.

    Args:
        dataset_dir: Path to the dir containing the unprocessed downloaded files.
        plot_save_path: Path where the dataset calendar plot will be saved.
    """

    # ---------------------------------------------------------
    # Check dataset directory hierarchy
    # ---------------------------------------------------------

    # Check directory hierarchy
    if not all(
        path.exists() and len(list(path.iterdir())) > 0
        for path in [
            raw_dir,
            raw_dir / "ensemble",
            processed_dir,
            processed_dir / "input",
            processed_dir / "target",
        ]
    ):
        raise FileNotFoundError(
            f"Dataset directory hierarchy not respected at dataset path {dataset_dir}."
        )

    input_dir: Path = processed_dir / "input"
    target_dir: Path = processed_dir / "target"
    input_file_paths: list[Path] = list(input_dir.glob("*.netcdf"))
    target_file_paths: list[Path] = list(target_dir.glob("*.netcdf"))

    # ---------------------------------------------------------
    # Load a sample input file ant check its validity
    # ---------------------------------------------------------

    # Load sample input dataarray to compare to the others
    input_sample = xr.open_dataarray(input_file_paths[0])
    input_sample_order_of_magnitude = math.floor(math.log10(abs(input_sample.mean())))

    # Check that input sample has the right coordinates
    if not set(input_sample.coords.keys()) == set(
        ["model", "species", "level", "leadtime", "latitude", "longitude", "run_date"]
    ):
        raise ValueError(
            f"Input has {list(input_sample.coords.keys())} "
            "coordinates when [model, species, level, leadtime, latitude, "
            "longitude, run_date]"
        )

    # Check that input sample has all the 11 CAMS models
    if not len(input_sample.model) == 11:
        raise ValueError("asdfasdfasdf")
    if not set(str(model_name) for model_name in input_sample.model.values) == set(
        MODEL_NAMES
    ):
        coords_model_names = set(
            str(model_name) for model_name in input_sample.model.values
        )
        missing_model_names = set(
            model_name
            for model_name in MODEL_NAMES
            if model_name not in coords_model_names
        )
        raise ValueError(
            "Not all models present in processed input model coordinates\n"
            f"Present: {coords_model_names}\n"
            f"Missing: {missing_model_names}"
        )

    # ---------------------------------------------------------
    # Check input files are similar to input sample
    # ---------------------------------------------------------

    for input_path in tqdm(input_file_paths, desc="Coordinates and units"):
        input_dataarray = xr.open_dataarray(input_path)

        # Check that the input has the right coordinates
        if not set(input_sample.coords.keys()) == set(
            [
                "model",
                "species",
                "level",
                "leadtime",
                "latitude",
                "longitude",
                "run_date",
            ]
        ):
            raise ValueError(
                f"Input has {list(input_sample.coords.keys())} "
                "coordinates when [model, species, level, leadtime, "
                "latitude, longitude, valid_date] is expected."
            )

        # Check that the input and output sample coordinates match
        if not all(
            (
                len(input_dataarray.coords[coord]) == len(input_sample.coords[coord])
                and (input_dataarray.coords[coord].to_index() == input_sample.coords[coord].to_index()).all().item()
                for coord in (
                    "model",
                    "species",
                    "level",
                    "leadtime",
                    "latitude",
                    "longitude",
                )
            )
        ):
            raise ValueError(
                "Target file coordinates does not match input coordinates."
                f"sample input:\n{input_sample}\n\nother input: {input_dataarray}"
            )
        
        if not len(input_dataarray.model) == 11:
            breakpoint()
            raise ValueError("22222 asdfasdfasdf")

        # Check run_date coordinate
        file_name_date = np.datetime64(input_path.stem.replace("_", "-"))
        if file_name_date != input_dataarray.run_date:
            raise ValueError(
                f"Input file name date {file_name_date} and "
                f"run time coordinate {input_dataarray.run_date} do not match."
            )

        # Compute order of magnitude
        input_mean = input_dataarray.mean()
        input_order_of_magnitude = math.floor(math.log10(abs(input_mean)))

        # Compare that they are to the same power of 10
        if input_order_of_magnitude != input_sample_order_of_magnitude:
            raise ValueError(
                "Not all input dataarray have the same unit, their order "
                f"of magnitude differ: {input_order_of_magnitude} "
                f"{input_sample_order_of_magnitude}."
            )

    print(f"\nAll input files are similar to this one:\n{input_sample}\n")

    # ----------------------------------------------------------
    # Check all the necessary target files exist
    # ----------------------------------------------------------

    # Extract leadtimes from the input file just processed
    leadtime_coords = [
        int(leadtime) for leadtime in input_sample.coords["leadtime"].values
    ]

    # Check that a target file exists for every lead time of every input file
    for input_path in tqdm(input_file_paths, desc="File existence check"):
        date = dt.datetime.strptime(input_path.stem, r"%Y_%m_%d")
        if not all(
            (
                target_dir
                / (date + dt.timedelta(hours=leadtime)).strftime(r"%Y_%m_%d_%H.netcdf")
            ).exists()
            for leadtime in leadtime_coords
        ):
            raise FileNotFoundError(
                f"Missing target file for input file {input_path.name}."
            )

    # ----------------------------------------------------------
    # Check all target files have the same coordinates and units
    # ----------------------------------------------------------

    for target_path in tqdm(target_file_paths, desc="Target files check"):
        target_dataarray = xr.load_dataarray(target_path)

        # Check that target has the right coordinates
        if not set(target_dataarray.coords.keys()) == set(
            ["species", "level", "latitude", "longitude", "valid_date"]
        ):
            raise ValueError(
                f"Target has {list(target_dataarray.coords.keys())} "
                "coordinates when [species, level, latitude, longitude, "
                "valid_date] is expected."
            )

        # Check target and input coordinates match
        if not all(
            (
                target_dataarray.coords[coord].values
                == input_sample.coords[coord].values
            ).all()
            for coord in ("species", "level", "latitude", "longitude")
        ):
            raise ValueError(
                "Target file coordinates does not match input coordinates."
            )

        # Check valid_date coordinate
        file_name_date = dt.datetime.strptime(target_path.stem, r"%Y_%m_%d_%H")
        valid_date = dt.datetime.fromtimestamp(
            cast(int, target_dataarray.coords["valid_date"].item()) / 10**9
        )
        if file_name_date != valid_date:
            raise ValueError(
                f"Target file name date {file_name_date} and "
                f"valid time coordinate {valid_date} do not match"
            )

        # Compute order of magnitude
        target_mean = target_dataarray.mean()
        target_order_of_magnitude = math.floor(math.log10(abs(target_mean)))

        # Compare that they are to the same power of 10
        if target_order_of_magnitude != input_sample_order_of_magnitude:
            raise ValueError(
                "Target does not have the same unit, as input sample "
                f"their order of magnitude differ: {target_order_of_magnitude} "
                f"{input_sample_order_of_magnitude}."
            )

    # ----------------------------------------------------------
    # Display report
    # ----------------------------------------------------------

    months = sorted(
        list(set(target_path.stem[:7] for target_path in target_file_paths))
    )

    _, ax = plt.subplots()
    ax.bar(
        x=months,
        height=[len(list(target_dir.glob(f"{month}_*"))) for month in months],
    )
    ax.set_title(f"Validation calendar for dataset {processed_dir.parents[-2].stem}")
    ax.set_ylabel("Number of leadtime available")
    ax.tick_params("x", rotation=30)
    plt.xticks(
        [month for i, month in enumerate(months) if i % 4 == 0],
        [month for i, month in enumerate(months) if i % 4 == 0],
    )
    plt.savefig(plot_save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./data/validation_calendar.png"),
        help="Path where the validation plot is saved.",
    )
    args = parser.parse_args()
    plot_save_path: Path = args.output

    # Validation
    validate(RAW_DATA_DIR, PROCESSED_DATA_DIR, plot_save_path)
