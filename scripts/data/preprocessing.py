"""Preprocessing script for a CAMS dataset.

Usage:
```bash
python scripts/data/0_preprocessing.py \
    -n --nb-jobs   # Number of parallel processes used. Defaults to 15.
                     Choose one to be able to catch error effectively.
    --overwrite    # If given, will delete the processed data folder before
                     writing to it.
    --plot_output  # Where the date availability plot will be saved.
                     Defaults to `./availability_calendar.png`
```

The script `scripts/data/validation.py` should be executed after this one.
"""

import datetime as dt
import os
import pickle as pkl
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from warnings import warn

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

from cams.settings import (
    ECMWF_MF_PARAMETER_NAME_MAPPING,
    KILOGRAM_TO_MICROGRAM,
    MODEL_NAMES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

PMACC_MODEL_NAMES = ["PMACC" + model_name for model_name in MODEL_NAMES]

HAUTEUR_LEVELS = (50, 100, 250, 500, 750, 1000, 2000, 3000, 5000)
SOL_LEVELS = (0,)

INPUT_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
    r"_(?P<species>[A-Z0-9_]+?)_(?P<leadtime>\d{2})h_(?P<level>[A-Z0-9_m]+)(\.grib)?$"
)

TARGET_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<month>\d{2})_(?P<species>[a-z0-9_.]+?)"
    r"_(?P<level>\d+)m_(?P<reanalysis_type>ira|vra)(\.netcdf)?$"
)


class CAMSCoordinateError(Exception):
    """Exception raised when encountering a data point with missing coordinates."""


class CAMSOrphanFileError(Exception):
    """Exception raised when cleaning orphan files"""


@dataclass
class ProcessingError:
    """Structured record of a single processing failure.

    Attributes :
        date: Identifier of the failing item.
        stage: Pipeline stage: "input" or "target".
        error_type: Short class name of the exception.
        message: Human-readable description of the failure.
    """

    date: str
    stage: str
    error_type: str
    message: str

    @property
    def month(self) -> str:
        """Returns the month from the date"""

        return self.date.replace("_", "-")[:7]


# --------------------------- #
#   Availability reporting    #
# --------------------------- #


def _gather_availability_info() -> dict:
    """
    Collect raw availability metadata from the dataset directory.

    Returns:
        A dict with keys: input_file_stems, target_file_stems,
        available_dates, available_species, available_levels, available_models,
    """

    # Gather file names
    input_file_stems: set[str] = set(
        path.stem for path in RAW_DATA_DIR.glob("PMACC*/*.grib")
    )
    target_file_stems: set[str] = set(
        path.stem for path in RAW_DATA_DIR.glob("ensemble/**/*.netcdf")
    )

    if len(input_file_stems) == 0:
        raise FileNotFoundError(f"No file like PMACC*/*.grib found in {RAW_DATA_DIR}")

    if len(target_file_stems) == 0:
        raise FileNotFoundError(
            f"No file like ensemble/**/*.netcdf found in {RAW_DATA_DIR}"
        )

    required_input_leadtimes: set[int] = set()
    required_input_species: set[str] = set()
    required_input_levels: set[int] = set()
    required_input_dates: set[dt.datetime] = set()
    required_input_months: set[dt.datetime] = set()

    for input_file in input_file_stems:
        match = INPUT_RE.match(input_file)
        if not match:
            raise ValueError(f"Inconsistant file name : {input_file}")

        required_input_species.add(match.group("species"))
        required_input_leadtimes.add(int(match.group("leadtime")))

        if "SOL" == match.group("level"):
            required_input_levels.add(0)
        elif "HAUTEUR" == match.group("level"):
            for level in HAUTEUR_LEVELS:
                required_input_levels.add(level)
        date_str = (
            match.group("year") + "_" + match.group("month") + "_" + match.group("day")
        )
        required_input_dates.add(
            dt.datetime.strptime(date_str, r"%Y_%m_%d")
            + dt.timedelta(hours=int(match.group("leadtime")))
        )
        month_str = match.group("year") + "_" + match.group("month")
        required_input_months.add(dt.datetime.strptime(month_str, r"%Y_%m"))

    available_target_species: set[str] = set()
    available_target_type_reanalysis: set[str] = set()

    for target_file in target_file_stems:
        match = TARGET_RE.match(target_file)
        if not match:
            raise ValueError(f"Inconsistant file name : {target_file}")

        date_str = match.group("year") + "_" + match.group("month")
        available_target_species.add(
            ECMWF_MF_PARAMETER_NAME_MAPPING[match.group("species")]
        )
        available_target_type_reanalysis.add(match.group("reanalysis_type"))

    # Gather available species
    available_species: set[str] = required_input_species | available_target_species

    # Gather available models
    available_models: set[str] = set(
        path.stem for path in RAW_DATA_DIR.iterdir() if path.stem in PMACC_MODEL_NAMES
    )

    return {
        "input_file_stems": input_file_stems,
        "target_file_stems": target_file_stems,
        "available_models": available_models,
        "available_species": available_species,
        "required_leadtimes": required_input_leadtimes,
        "required_dates": required_input_dates,
        "required_months": required_input_months,
        "required_levels": required_input_levels,
        "available_target_type_reanalysis": available_target_type_reanalysis,
    }


def report_available_data() -> None:
    """Print a summary of available raw data and save and availability plot.

    Args:
        plot_save_path: Path where the availability calendar plot will be saved.
    """
    print("\n Gathering data availability...")
    info = _gather_availability_info()

    print(f"  Models : {info['available_models']}")
    print(f"  Species : {info['available_species']}")
    print(f"  Required Leadtimes : {info['required_leadtimes']}")
    print(f"  Required Levels : {info['required_levels']}")
    print(f"  Required Dates : {len(info['required_dates'])} date(s) found")
    print(f"  Required Months : {len(info['required_months'])} month(s) found")
    print(f"  Target Reanalysis Available : {info['available_target_type_reanalysis']}")


# ----------------------------- #
#   Input processing helpers    #
# ----------------------------- #


def _assign_levels_dimension(dataset: xr.Dataset) -> xr.Dataset:
    """Rename or create levels dimension
    Args:
        dataset: dataset to process

    Returns:
        dataset with levels dimension
    """
    # Dataset containing SOL values have surface coordinates
    if "surface" in list(dataset.coords.keys()):
        dataset = dataset.drop_vars("surface")
        dataset = dataset.expand_dims({"levels": [0]})
    # Dataset containing HAUTEUR values have heightAboveGround dimension
    elif "heightAboveGround" in list(dataset.coords.keys()):
        dataset = dataset.rename({"heightAboveGround": "levels"})

    return dataset


def _drop_unused_coords(dataset: xr.Dataset) -> xr.Dataset:
    """Drop coordinated that are not needed after merging.

    Args:
        dataset: Input xarray Dataset

    Returns:
        Dataset with unused coordinates removed.
    """
    unused = [
        "valid_time",
        "step",
        "valid_time",
        "time",
        "surface",
    ]
    for var in unused:
        if var in list(dataset.coords.keys()):
            dataset = dataset.drop_vars(var)
    return dataset


def _add_merge_dimensions(
    dataset: xr.Dataset,
    model_name: str,
    species_name: str,
    leadtime: str,
) -> xr.Dataset:
    """Expand and assign merge dimensions to an input dataset.

    Args:
        dataset: Input xarray Dataset.
        model_name: Name of the CTM model.
        species_name: Atmospheric species name.
        levels: List of vertical levels identifier.
        leadtime: Forecast leadtime string.

    Returns:
        Dataset with expanded dimensions and assigned coordinates.
    """
    dataset = dataset.expand_dims(dim=["model", "species", "leadtime"], axis=[0, 1, 2])
    return dataset.assign_coords(
        {
            "model": [model_name],
            "species": [species_name],
            "leadtime": [leadtime],
        }
    )


def _normalize_grid(
    dataset: xr.Dataset,
    model_name: str,
    lat_coordinates: xr.DataArray,
    lon_coordinates: xr.DataArray,
) -> xr.Dataset:
    """Redefine lat/lon for models known to be slightly different grid.


    LOTOS and SILAM are on slightly different grids.
    We simply redefine their longitude and latitude to be the
    same as the other models, creating an accepted imprecision.

    Args:
        dataset: Input xarray Dataset.
        model_name: Name of the CTM model.
        lat_coordinates: Latitude xarray.
        lon_coordinates: Longitude xarray.

    Returns:
        Dataset with normalized coordinates name if needed
    """
    if model_name in ("LOTOS", "SILAM"):
        dataset.coords["latitude"] = lat_coordinates.latitude.values
        dataset.coords["longitude"] = lon_coordinates.longitude.values

    return dataset


def _round_coordinates(
    dataset: xr.Dataset,
    source_path: Path,
) -> xr.Dataset:
    """Round latitude and longitude coordinates to 2 decimal places.

    Emits a warning if rounding introduces a significant deviation.

    Args:
        dataset: Input xarray Dataset.
        source_path: Path of the source file (used in warning message).

    Returns:
        Dataset with rounded coordinates

    """

    rounded_lat = np.round(dataset.latitude.values, decimals=2)
    rounded_lon = np.round(dataset.longitude.values, decimals=2)
    if not np.allclose(dataset.latitude, rounded_lat) or not np.allclose(
        dataset.longitude, rounded_lon
    ):
        warn(
            "Rounded longitude or latitude is not close to the "
            f"original coordinate for {source_path}."
        )
    return dataset.assign_coords(
        latitude=np.round(dataset.coords["latitude"].values, decimals=2),
        longitude=np.round(dataset.coords["longitude"].values, decimals=2),
    )


def _validate_model_coords(dataset: xr.Dataset) -> None:
    """Ensure all expected CTM models are present in the merged dataset.

    Args:
        dataset: Merged xarray Dataset.

    Raises:
        CAMSCoordinateError: If one or more expected models are missing

    """
    present = set(str(name) for name in dataset.model.values)
    expected = set(MODEL_NAMES)
    missing = (present | expected) - (present & expected)
    if missing:
        raise CAMSCoordinateError(f"Missing model(s): {missing}")


def _process_input_date(
    run_date_string: str,
    lat_coordinates: xr.DataArray,
    lon_coordinates: xr.DataArray,
) -> ProcessingError | None:
    """Process input data for a run date.

    Some of the input data are not on the same grid as the others.
    The difference is by a very small distance, so we normalize them
    all on the same latitude and longitude.

    Args:
        run_date_string: The run date string, written as YYYY_MM_DD.
        lat_coordinates: Latitude coordinates to normalize data to.
        lon_coordinates: Longitude coordinates to normalize data to.
    """

    # Check if the processed file exists
    save_path = PROCESSED_DATA_DIR / "input" / f"{run_date_string}.netcdf"
    if save_path.exists():
        return

    def preprocess_input(dataset: xr.Dataset) -> xr.Dataset:
        """Function called by xr.open_mfdataset to preprocess the
        `.netcdf` files before merging them.

        Args:
            dataset: The dataset object passed by xr.open_mfdataset.
        """
        # Gather informations about which file is being processed
        path = Path(dataset.encoding["source"])
        model_name: str = path.parent.stem[5:]  # Remove PMACC from model name
        match = INPUT_RE.match(path.name)
        if not match:
            raise ValueError(f"Inconsistant file name : {path.name}")
        species_name = match.group("species")
        leadtime = match.group("leadtime")
        dataset = _assign_levels_dimension(dataset)
        dataset = _drop_unused_coords(dataset)
        dataset = _add_merge_dimensions(dataset, model_name, species_name, leadtime)
        dataset = _normalize_grid(dataset, model_name, lat_coordinates, lon_coordinates)
        dataset = _round_coordinates(dataset, source_path=path)

        # Convert from kg/m3 to micro gram per cubic meter
        dataset = dataset.assign_attrs(units="µg/m3")
        dataset *= KILOGRAM_TO_MICROGRAM

        return dataset

    try:
        grib_paths = list(RAW_DATA_DIR.glob(f"**/{run_date_string}*.grib"))
        if not grib_paths:
            raise FileNotFoundError(
                f"No .grib files found for run date {run_date_string}."
            )

        # Open grib files as xr.Dataset and classify them based on the weather
        # parameter they represent.
        output_dataset = xr.open_mfdataset(
            paths=grib_paths,
            preprocess=preprocess_input,
            coords="minimal",
            compat="equals",
            join="outer",
            errors="warn",
        )
        # Add run_date coordinate
        output_dataset = output_dataset.assign_coords(
            run_date=np.datetime64(run_date_string.replace("_", "-"))
        )

        _validate_model_coords(output_dataset)
        # Save
        output_dataset.to_netcdf(save_path)

    except Exception as exc:
        return ProcessingError(
            date=run_date_string,
            stage="input",
            error_type=type(exc).__name__,
            message=str(exc),
        )

    return None


# ------------------------------ #
#   Target processing helpers    #
# ------------------------------ #


def _load_target_dataarray(file_path: Path) -> xr.DataArray:
    """Load and prepare a single target netCDF file as an xr.DataArray.

    Adds species/level dimensions, renames axes, flips latitudes and
    round coordinates.

    Args:
        file_path: Path to the monthly reanalysis netCDF file.

    Returns:
        Prepared DataArray ready for concatenation.
    """
    data_array: xr.DataArray = xr.open_dataarray(file_path)
    filename = file_path.name
    match = TARGET_RE.match(filename)
    if not match:
        raise ValueError(f"Inconsistant file name : {filename}")

    # Add species dimension and coordinates
    species_name: str = ECMWF_MF_PARAMETER_NAME_MAPPING[match.group("species")]
    level: int = int(match.group("level"))
    data_array = data_array.expand_dims(dim=["species", "level"], axis=[0, 1])
    data_array = data_array.assign_coords(
        {"species": [species_name], "level": [str(level)]}
    )

    # Rename variables
    data_array = data_array.rename(
        {
            "time": "valid_date",
            "lat": "latitude",
            "lon": "longitude",
        }
    )

    # Remove variable coordinate (replaced by species)
    if "variable" in data_array.attrs:
        data_array = data_array.drop_vars("variable")

    # Add units attributes
    data_array = data_array.assign_attrs(units="µg/m3")

    # Vertical flip, reindex the latitudes in reverse order
    data_array = data_array.reindex(latitude=list(reversed(data_array.latitude)))

    # Round latitude coordinates
    data_array = data_array.assign_coords(
        {
            "latitude": np.round(data_array.coords["latitude"].values, decimals=2),
            "longitude": np.round(data_array.coords["longitude"].values, decimals=2),
        }
    )
    return data_array


def _extract_hour_dataarray(
    month_dataarrays: list[xr.DataArray],
    date: dt.date,
    levels: list[int],
) -> xr.DataArray:
    """Slice and concatenate monthly dataarrays to a single-hour DataArray.

    Args:
        month_dataarrays: List of DataArrays coverring the full month.
        date: The specific date to extract.
        levels: List of vertical levels to include

    Returns:
        DataArray for the requested hour, concatenated over species and levels.
    """
    return xr.concat(
        objs=(
            xr.concat(
                objs=(
                    dataarray.sel(
                        valid_date=date,
                        level=str(level),
                    )
                    .expand_dims("level", 1)
                    .assign_coords(level=[level])
                    for dataarray in month_dataarrays
                    if str(level) in dataarray.level.values
                ),
                dim="species",
                join="outer",
            )
            for level in levels
        ),
        dim="level",
        join="outer",
    )


def _process_target_month(
    required_dates: list[dt.datetime],
    levels: list[int],
) -> ProcessingError | None:
    """Processes some raw netcdf files of monthly target (reanalysis) data.
    Split the reanalysis monthly files into one file for each hour of the month
    they contain.

    Args:
        required_dates: List of dates expected to be extracted
            from one month of reanalysis.
            will be written.
        levels: List of levels to extract from raw data.
    """
    # Define the date month
    target_date = dt.date(required_dates[0].year, required_dates[0].month, 1)
    month_str = target_date.strftime("%Y-%m")

    try:
        # Find the netcdf files for the given month
        all_files = RAW_DATA_DIR.glob("ensemble/*.netcdf")
        file_paths: list[Path] = []
        for target_file in all_files:
            match = TARGET_RE.match(target_file.name)
            if not match:
                raise ValueError(f"Inconsistant file name : {target_file.name}")
            if (
                match.group("year") == month_str.split("-")[0]
                and match.group("month") == month_str.split("-")[1]
                and match.group("level") in list(map(str, levels))
            ):
                file_paths.append(target_file)

        if not file_paths:
            return None
        # Open all the month's netcdf files, one for each weather parameter
        month_dataarrays: list[xr.DataArray] = [
            _load_target_dataarray(file_path) for file_path in file_paths
        ]

        # Validate
        if not month_dataarrays:
            raise FileNotFoundError(f"No data arrays loaded for month {target_date}.")
        # Save a target file for each dates required
        for date in required_dates:
            # Check if output path exists
            save_path = (
                PROCESSED_DATA_DIR
                / "target"
                / f"{date.strftime(r'%Y_%m_%d_%H')}.netcdf"
            )
            if save_path.exists():
                continue

            # Select the right date for each dataaray
            hour_dataarray = _extract_hour_dataarray(month_dataarrays, date, levels)

            # Save
            hour_dataarray.name = date.strftime(r"%Y_%m_%d_%H reanalisis")
            hour_dataarray.to_netcdf(save_path)

    except Exception as exc:
        return ProcessingError(
            date=month_str,
            stage="target",
            error_type=type(exc).__name__,
            message=str(exc),
        )


# ------------ #
#   Cleanup    #
# ------------ #


def _cleanup_orphan_inputs(leadtimes: list[int]) -> list[ProcessingError]:
    """Remove input files that have no matching target files.

    Args:
        leadtimes: List of integer forecast leadtimes to check.

    Raises:
        CAMSOrphanFileError when a file is removed.
    """
    errors: list[ProcessingError] = []
    for input_path in tqdm(
        list((PROCESSED_DATA_DIR / "input").glob("*.netcdf")), desc="Cleanup"
    ):
        date = dt.datetime.strptime(input_path.stem, r"%Y_%m_%d")
        month_str = date.strftime("%Y-%m")
        all_target_exist = all(
            (
                PROCESSED_DATA_DIR
                / "target"
                / (date + dt.timedelta(hours=leadtime)).strftime(r"%Y_%m_%d_%H.netcdf")
            ).exists()
            for leadtime in leadtimes
        )
        if not all_target_exist:
            input_path.unlink()
            errors.append(
                ProcessingError(
                    date=month_str,
                    stage="input",
                    error_type=CAMSOrphanFileError.__name__,
                    message="Orphan file cleaned at: {input_path}",
                )
            )

    return errors


def _print_error_summary(errors: list[ProcessingError]) -> None:
    """Print a structured terminal summary of all processing errors.

    Args:
        errors: List of ProcessingError records collected during the run.
    """
    if not errors:
        print(" NO ERRORS")
        return

    by_type: dict[str, int] = defaultdict(int)
    for e in errors:
        by_type[e.error_type] += 1

    bar = "-" * 60
    print(f"\nWARNING: {len(errors)} error(s) across {len(by_type)} type(s)")
    for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {etype:<28} {count:>4} occurence(s)")
    print(bar)
    for e in sorted(errors, key=lambda x: (x.stage, x.date)):
        print(f"  [{e.stage:>6}] {e.date} -> {e.error_type}")
        print(f"    {e.message}")
    print(bar)
    for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {etype:<28} {count:>4} occurence(s)")


def _plot_error_report(
    errors: list[ProcessingError],
    plot_save_path: Path,
) -> None:
    """Save a grouped bar chart of errors per month and per type.

    Args:
        errors: List of ProcessingError records.
        plot_save_path: Destination path for the PNG.
    """
    if not errors:
        return

    data: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in errors:
        data[e.month][e.error_type] += 1

    months = sorted(data.keys())
    error_types = sorted({e.error_type for e in errors})
    x = np.arange(len(months))
    bar_width = 0.8 / max(len(error_types), 1)

    _, ax = plt.subplots()
    for i, etype in enumerate(error_types):
        counts = [data[m].get(etype, 0) for m in months]
        offset = (i - len(error_types) / 2 + 0.5) * bar_width
        ax.bar(x + offset, counts, width=bar_width, label=etype)

    ax.set_title("Processing errors per month")
    ax.set_ylabel("Number of errors")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m if i % 2 == 0 else "" for i, m in enumerate(months)], rotation=30, ha="right"
    )
    ax.legend(title="Error type")
    plt.savefig(plot_save_path)


def process(
    nb_jobs: int = 15,
    overwrite: bool = False,
    plot_error_path: Path = Path("./data/error_report.png"),
) -> None:
    """Prepares a CAMS dataset for use in training.

    Args:
        nb_jobs: Number of parallel jobs to use for preprocessing.
            Defaults to 15.
        overwrite: If True, will remove existing files in the output dir.
        plot_error_path: Path to the error report folder.
        plot_error_path: Path to the error report folder.
    """
    errors: list[ProcessingError] = []

    # Overwrite
    if overwrite:
        files = list(PROCESSED_DATA_DIR.glob("**/*.netcdf"))
        print(f"\nINFO: Overwrite requested - deleting {len(files)} file(s)...")
        for file_path in files:
            file_path.unlink()

    # Validate directory structure
    print("\nINFO: Validating raw directory structure...")
    expected_dirs = set(PMACC_MODEL_NAMES) | {"ensemble"}
    actual_dirs = set(os.listdir(RAW_DATA_DIR))
    unknown = actual_dirs - expected_dirs
    if unknown:
        raise ValueError(
            f"Unknown directirues ub raw datra folder: {unknown}. "
            "Expected only model dirs and 'ensemble'."
        )

    # Create output dirs
    (PROCESSED_DATA_DIR / "input").mkdir(exist_ok=True, parents=True)
    (PROCESSED_DATA_DIR / "target").mkdir(exist_ok=True, parents=True)

    # Gather dates
    run_date_strings: set[str] = set(
        file_path.stem[:10] for file_path in RAW_DATA_DIR.glob(r"**/*.grib")
    )
    print(f"\nINFO: Found {len(run_date_strings)} run date(s) to process.")

    # ---------------------------------------------------------------------
    # -------                      input                           --------
    # ---------------------------------------------------------------------

    # Open reference MACCGE01 grid.
    print("INFO: Loading reference MACCGE01 grid...")
    with open("data/MACCGE01.pkl", "br") as file:
        lat, lon = pkl.load(file)

    # Process the input with parallel jobs.
    print("\nINFO: Processing input files...")
    input_results = joblib.Parallel(n_jobs=nb_jobs)(
        joblib.delayed(_process_input_date)(
            run_date_string,
            lat,
            lon,
        )
        for run_date_string in tqdm(run_date_strings, desc="Input processing")
    )
    errors.extend(r for r in input_results if r is not None)

    # ---------------------------------------------------------------------
    # -------                    target                            --------
    # ---------------------------------------------------------------------

    info = _gather_availability_info()

    print(f"\nINFO: Processing {len(info['required_months'])} target month(s)...")
    # Process the target with parallel jobs.
    target_results = joblib.Parallel(n_jobs=nb_jobs)(
        joblib.delayed(_process_target_month)(
            required_dates=[
                date
                for date in info["required_dates"]
                if (date.year == date_month.year and date.month == date_month.month)
            ],
            levels=info["required_levels"],
        )
        for date_month in tqdm(info["required_months"], desc="Target processing")
    )

    errors.extend(r for r in target_results if r is not None)

    # ---------------------------------------------------------------------
    # -------                   cleanup                            --------
    # ---------------------------------------------------------------------

    # Delete processed input files that do not have an associated target file
    print("\nINFO Cleaning orphan input files...")
    cleanup_errors = _cleanup_orphan_inputs(info["required_leadtimes"])
    errors.extend(cleanup_errors)
    print(f" Deleted {len(cleanup_errors)} ophan input file(s).")

    # Error summary
    _print_error_summary(errors)
    _plot_error_report(errors, plot_error_path)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Processes a CAMS dataset raw folder into processed folder.",
    )
    parser.add_argument(
        "--nb-jobs",
        "-j",
        type=int,
        default=15,
        help=(
            "Number of parallel processes used. Defaults to 15. "
            "Choose one to be able to catch error effectively."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("If given, will delete the processed data folder before writing to it."),
    )
    parser.add_argument(
        "--plot_error",
        type=Path,
        default=Path("./data/error_report.png"),
        help=(
            "Where the error plot will be saved. Defaults to `./data/error_report.png`"
        ),
    )

    args = parser.parse_args()

    # Validate command line arguments
    nb_jobs: int = args.nb_jobs
    overwrite: bool = args.overwrite
    plot_error_path: Path = args.plot_error

    # Report data availability
    report_available_data()

    # Process raw dataset
    process(nb_jobs=nb_jobs, overwrite=overwrite, plot_error_path=plot_error_path)
