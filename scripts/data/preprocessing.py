"""Preprocessing script for a CAMS dataset.

Usage:
```bash
python scripts/data/0_preprocessing.py \
    --dataset-dir  # Path to the dataset dir. Defaults to value in settings.py
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
from pathlib import Path
from warnings import warn, filterwarnings
import atexit
import shutil

SCRATCH_TMP = '/scratch/shared/cams/tmp_lespilettec'
os.makedirs(SCRATCH_TMP, exist_ok=True)

atexit.register(shutil.rmtree, SCRATCH_TMP, ignore_errors=True)


import earthkit.data as ekd
import gribapi
import joblib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

from cams.settings import (
    CAMS_DATASET_DIR,
    ECMWF_MF_PARAMETER_NAME_MAPPING,
    KILOGRAM_TO_MICROGRAM,
    MODEL_NAMES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

os.environ["EARTHKIT_DATA_HOME"] = "/tmp/earthkit_cache"

filterwarnings("ignore", message="ecCodes.*recommended")

PMACC_MODEL_NAMES = ["PMACC" + model_name for model_name in MODEL_NAMES]


class CAMSCoordinateError(Exception):
    """Exception raised when encountering a data point with missing coordinates."""

    pass


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

    # Gather available dates
    available_input_leadtimes: set[str] = set(
        stem.split("_")[-3] for stem in input_file_stems
    )

    available_target_months: set[str] = set(stem[:7] for stem in target_file_stems)

    available_dates: set[dt.datetime] = set(
        dt.datetime.strptime(stem[:10], r"%Y_%m_%d") + dt.timedelta(hours=int(leadtime))
        for stem in input_file_stems
        for leadtime in available_input_leadtimes
        if stem[:7] in available_target_months
    )

    # Gather available species
    available_species: set[str] = set(
        stem.split("_")[-1] for stem in input_file_stems
    ) | set(
        ECMWF_MF_PARAMETER_NAME_MAPPING[path.stem]
        for path in RAW_DATA_DIR.glob("ensemble/*")
    )

    # Gather available levels
    available_levels: set[str] = set(
        stem.split("_")[-2] for stem in input_file_stems
    ) | set(stem.split("_")[-1] for stem in target_file_stems)

    # Gather available models
    available_models: set[str] = set(
        path.stem for path in RAW_DATA_DIR.iterdir() if path.stem in PMACC_MODEL_NAMES
    )

    return {
        "input_file_stems": input_file_stems,
        "target_file_stems": target_file_stems,
        "available_dates" : available_dates,
        "available_species" : available_species,
        "available_levels" : available_levels,
        "available_models" : available_models,
    }

def _plot_availability(
        target_file_stems: set[str],
        available_levels: set[str],
        plot_save_path: Path,
        dataset_dir: Path,
) -> None:
    """
    """
    months = sorted(list(set(target_stem[:7] for target_stem in target_file_stems)))

    _, ax = plt.subplots()
    ax.bar(
        x=months,
        height=[len(available_levels)] * len(months),
    )
    ax.set_title(f"What raw data seems available for dataset {dataset_dir.stem}")
    ax.set_ylabel("Number of leadtime available")
    ax.tick_params("x", rotation=30)
    plt.xticks(
        [month for i, month in enumerate(months) if i % 4 == 0],
        [month for i, month in enumerate(months) if i % 4 == 0],
    )
    plt.savefig(plot_save_path)
    print(f"Availability plot saved in {plot_save_path}")

def report_available_data(
    plot_save_path: Path,
) -> None:
    """Print a summary of available raw data and save and availability plot.

    Args:
        plot_save_path: Path where the availability calendar plot will be saved.
        dataset_dir: Path to the dataset directory(for plot title).
    """
    print("\n Gathering data availability...")
    info = _gather_availability_info()

    print(f"  Models  : {info['available_models']}")
    print(f"  Species  : {info['available_species']}")
    print(f"  Levels  : {info['available_levels']}")
    print(f"  Dates  : {len(info['available_dates'])} dates found")

    _plot_availability(
        target_file_stems=info["available_models"],
        available_levels=info["available_levels"],
        plot_save_path=plot_save_path,
        dataset_dir=dataset_dir,
    )


# ----------------------------- #
#   Input processing helpers    #
# ----------------------------- #


def _drop_unused_coords(dataset: xr.Dataset) -> xr.Dataset:
    """ Drop coordinated that are not needed after merging.

    Args:
        dataset: Input xarray Dataset

    Returns:
        Dataset with unused coordinates removed.
    """
    unused = ["valid_time", "step", "valid_time", "heightAboveGround", "time", "surface",]
    for var in unused:
        if var in list(dataset.coords.keys()):
            dataset.drop_vars(var)
    return dataset

def _add_merge_dimensions(
        dataset: xr.Dataset,
        model_name: str,
        species_name: str,
        level: str,
        leadtime: str,
    ) -> xr.Dataset:
    """Expand and assign merge dimensions to an input dataset.
    
    Args:
        dataset: Input xarray Dataset.
        model_name: Name of the CTM model.
        species_name: Atmospheric species name.
        level: Vertical level identifier.
        leadtime: Forecast leadtime string.

    Returns:
        Dataset with expanded dimensions and assigned coordinates.
    """
    dataset = dataset.expand_dims(
            dim=["model", "species", "level", "leadtime"], axis=[0, 1, 2, 3]
        )
    return dataset.assign_coords(
        {
            "model": [model_name],
            "species": [species_name],
            "level": [level],
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
) -> None:
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
        _, _, _, leadtime, level, species_name = path.stem.split("_")

        dataset = _drop_unused_coords(dataset=dataset)
        dataset = _add_merge_dimensions(dataset, model_name, species_name, level, leadtime)
        dataset = _normalize_grid(dataset, model_name, lat_coordinates, lon_coordinates)
        dataset = _round_coordinates(dataset, source_path=path)

        # Convert from kg/m3 to micro gram per cubic meter
        dataset = dataset.assign_attrs(units="µg/m3")
        dataset *= KILOGRAM_TO_MICROGRAM

        return dataset

    try:
        # Open grib files as xr.Dataset and classify them based on the weather
        # parameter they represent.
        output_dataset = xr.open_mfdataset(
            paths=list(RAW_DATA_DIR.glob(f"**/{run_date_string}*.grib")),
            preprocess=preprocess_input,
            coords="minimal",  # type: ignore[reportArgumentType]
            compat="equals",
            join="outer",
        )

        # Add run_date coordinate
        output_dataset = output_dataset.assign_coords(
            run_date=np.datetime64(run_date_string.replace("_", "-"))
        )

        _validate_model_coords(output_dataset)
        # Save
        output_dataset.to_netcdf(save_path)

    except (gribapi.errors.WrongGridError, CAMSCoordinateError) as error:
        # Catch in consistent input data, and log it
        print(f"Error on {run_date_string}: {error}")

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
    data_array: xr.DataArray = (
        ekd.from_source(
            "file",
            file_path,
        )
        .to_xarray()
        .to_dataarray()[0]
    )

    # Add species dimension and coordinates
    species_name: str = ECMWF_MF_PARAMETER_NAME_MAPPING[file_path.parent.stem]
    level: int = int(file_path.stem.split("_")[-1].rstrip("m"))
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
    data_array = data_array.drop_vars("variable")

    # Add units attributes
    data_array = data_array.assign_attrs(units="µg/m3")

    # Vertical flip, reindex the latitudes in reverse order
    data_array = data_array.reindex(latitude=list(reversed(data_array.latitude)))

    # Round latitude coordinates
    data_array = data_array.assign_coords(
        {
            "latitude": np.round(data_array.coords["latitude"].values, decimals=2),
            "longitude": np.round(
                data_array.coords["longitude"].values, decimals=2
            ),
        }
    )
    return data_array

def _extract_hour_dataarray(
        month_dataarrays: list[xr.DataArray],
        date: dt.date,
        levels: list[str],
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
                            level=level,
                        )
                        .expand_dims("level", 1)
                        .assign_coords(level=[level])
                        for dataarray in month_dataarrays
                        if level in dataarray.coords["level"].values
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
    levels: list[str],
) -> None:
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
    date_month = dt.date(required_dates[0].year, required_dates[0].month, 1)

    # Find the netcdf files for the given month
    file_paths: list[Path] = list(
        RAW_DATA_DIR.glob(
            f"ensemble/**/{date_month.year}_{date_month.month:02}_*m.netcdf"
        )
    )
    if file_paths == []:
        return

    # Open all the month's netcdf files, one for each weather parameter
    month_dataarrays: list[xr.DataArray] = []
    for file_path in file_paths:
        # Load data array
        data_array = _load_target_dataarray(file_path)

        # Concat to month dataaray
        month_dataarrays.append(data_array)

    # Validate
    if len(month_dataarrays) == 0:
        raise FileNotFoundError(f"File not founds for input {date_month}.")

    # Save a target file for each dates required
    for date in required_dates:
        # Check if output path exists
        save_path = (
            PROCESSED_DATA_DIR / "target" / f"{date.strftime(r'%Y_%m_%d_%H')}.netcdf"
        )
        if save_path.exists():
            continue

        # Select the right date for each dataaray
        hour_dataarray = _extract_hour_dataarray(month_dataarrays, date, levels)

        # Save
        hour_dataarray.name = date.strftime(r"%Y_%m_%d_%H reanalisis")
        hour_dataarray.to_netcdf(save_path)


# ------------ #
#   Cleanup    #
# ------------ #

def _cleanup_orphan_inputs(leadtimes: list[int]) -> int:
    """Remove input files that have no matching target files.

    Args:
        leadtimes: List of integer forecast leadtimes to check.

    Returns:
        Number of file deleted.
    """
    deleted = 0
    for input_path in tqdm(
    list((PROCESSED_DATA_DIR / "input").glob("*.netcdf")), desc="Cleanup"
    ):
        date = dt.datetime.strptime(input_path.stem, r"%Y_%m_%d")
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
            deleted += 1
    return deleted

def _print_error_summary(errors: dict[str, str]) -> None:
    """
    """
    if not errors:
        print(" NO ERRORS")
        return
    
    bar = '-' * 60
    print(f"\nWARNING: {len(errors)} error(s) occurred during processing:")
    print(bar)
    for key, message in errors:
        print(f"  {key}")
        print(f"   {message}")
    print(bar)

def process(nb_jobs: int = 15, overwrite: bool = False) -> None:
    """Prepares a CAMS dataset for use in training.

    Args:
        dataset_dir: Path to the dataset dir.
        nb_jobs: Number of parallel jobs to use for preprocessing.
            Defaults to 15.
        overwrite: If True, will remove existing files in the output dir.
    """
    errors: dict[str, str] = {}

    # Overwrite
    if overwrite:
        files = list(PROCESSED_DATA_DIR.glob("**/*.netcdf"))
        print(f"\nINFO: Overwrite requested - deleting {len(files)} file(s)...")
        for file_path in files:
            file_path.unlink()

    # Validate directory structure
    print("\nINFO: Validating raw directory structure...")
    expected_dirs = set(PMACC_MODEL_NAMES) | {'ensemble'}
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
    print('INFO: Loading reference MACCGE01 grid...')
    with open("data/MACCGE01.pkl", "br") as file:
        lat, lon = pkl.load(file)

    # Process the input with parallel jobs.
    print('\nINFO: Processing input files...')
    input_results = joblib.Parallel(n_jobs=nb_jobs)(
        joblib.delayed(_process_input_date)(
            run_date_string,
            lat,
            lon,
        )
        for run_date_string in tqdm(run_date_strings, desc="Input processing")
    )
    for result in input_results:
        if result is not None:
            date_str, err_msg = result
            errors[f"input/{date_str}"] = err_msg

    # Extract leadtimes from the input file just processed
    input_sample_path = list((PROCESSED_DATA_DIR / "input").glob("*.netcdf"))[0]
    input_sample = xr.load_dataarray(input_sample_path)
    leadtimes = [int(leadtime) for leadtime in input_sample.coords["leadtime"].values]
    print(f"INFO: Detected {len(leadtimes)} leadtime(s): {leadtimes}")
    
    # ---------------------------------------------------------------------
    # -------                    target                            --------
    # ---------------------------------------------------------------------

    # Gather months existing in the input
    required_dates: set[dt.datetime] = set(
        dt.datetime.strptime(path.stem[:10], r"%Y_%m_%d") + dt.timedelta(hours=leadtime)
        for leadtime in leadtimes
        for path in PROCESSED_DATA_DIR.glob(r"input/*.netcdf")
    )
    required_months: set[dt.datetime] = set(
        dt.datetime.strptime(path.stem[:7], r"%Y_%m")
        for path in PROCESSED_DATA_DIR.glob(r"input/*.netcdf")
    )
    print(f'\nINFO: Processing {len(required_months)} target month(s)...')
    
    # Process the target with parallel jobs.
    target_results = joblib.Parallel(n_jobs=nb_jobs)(
        joblib.delayed(_process_target_month)(
            required_dates=[
                date
                for date in required_dates
                if (date.year == date_month.year and date.month == date_month.month)
            ],
            levels=input_sample.coords["level"].values,
        )
        for date_month in tqdm(required_months, desc="Target processing")
    )

    for result in target_results:
        if result is not None:
            date_str, err_msg = result
            errors[f"target/{date_str}"] = err_msg

    # ---------------------------------------------------------------------
    # -------                   cleanup                            --------
    # ---------------------------------------------------------------------

    # Delete processed input files that do not have an associated target file
    print("\nINFO Cleaning orphan input files...")
    deleted = _cleanup_orphan_inputs(leadtimes)
    print(f" Deleted {deleted} ophan input file(s).")

    # Error summary
    _print_error_summary(errors)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Processes a CAMS dataset raw folder into processed folder.",
    )
    parser.add_argument(
        "--dataset-dir",
        "-i",
        type=Path,
        default=CAMS_DATASET_DIR,
        help="Path to the dataset dir. Defaults to value in settings.py",
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
        "--plot_output",
        type=Path,
        default=Path("./data/availability_calendar.png"),
        help=(
            "Where the date availability plot will be saved. "
            "Defaults to `./data/availability_calendar.png`"
        ),
    )
    args = parser.parse_args()

    # Validate command line arguments
    dataset_dir: Path = args.dataset_dir
    nb_jobs: int = args.nb_jobs
    overwrite: bool = args.overwrite
    plot_save_path: Path = args.plot_output

    # Report data availability
    report_available_data(
        plot_save_path=plot_save_path,
    )

    # Process raw dataset
    process(
        nb_jobs=nb_jobs,
        overwrite=overwrite,
    )
