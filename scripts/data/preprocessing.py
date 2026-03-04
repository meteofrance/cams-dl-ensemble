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
from warnings import warn

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
)

PMACC_MODEL_NAMES = ["PMACC" + model_name for model_name in MODEL_NAMES]


def report_available_data(
    raw_dir: Path,
    plot_save_path: Path,
) -> None:
    """Prints what data is available in the raw dir.
    Informs on what will be processed

    Args:
        raw_dir: Path to the raw data dir.
        plot_save_path: Path where the availability calendar
            plot will be saved.
    """

    # Gather file names
    input_file_stems: set[str] = set(
        path.stem for path in raw_dir.glob("PMACC*/*.grib")
    )
    target_file_stems: set[str] = set(
        path.stem for path in raw_dir.glob("ensemble/**/*.netcdf")
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
        for path in raw_dir.glob("ensemble/*")
    )

    # Gather available levels
    available_levels: set[str] = set(
        stem.split("_")[-2] for stem in input_file_stems
    ) | set(stem.split("_")[-1] for stem in target_file_stems)

    # Gather available models
    available_models: set[str] = set(
        path.stem for path in raw_dir.iterdir() if path.stem in PMACC_MODEL_NAMES
    )

    # Print report
    print(f"models: {available_models}")
    print(f"species: {available_species}")
    print(f"levels: {available_levels}")
    print(f"dates: {available_dates}")

    # Plot dates
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


def _process_input_date(
    run_date_string: str,
    raw_dir: Path,
    processed_dir: Path,
    lat_coordinates: xr.DataArray,
    lon_coordinates: xr.DataArray,
) -> None:
    """Process input data for a run date.
    Some of the input data are not on the same grid as the others.
    The difference is by a very small distance, so we normalize them
    all on the same latitude and longitude.

    Args:
        run_date_string: The run date string, written as YYYY_MM_DD.
        raw_dir: Path to the raw dir.
        processed_dir: Path to the processed dir.
        lat_coordinates: Latitude coordinates to normalize data to.
        lon_coordinates: Longitude coordinates to normalize data to.
    """

    # Check if the processed file exists
    save_path = processed_dir / "input" / f"{run_date_string}.netcdf"
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

        # Drop unused coordinates
        for variable_name in [
            "valid_time",
            "step",
            "valid_time",
            "heightAboveGround",
            "time",
            "surface",
        ]:
            if variable_name in list(dataset.coords.keys()):
                dataset = dataset.drop_vars(variable_name)

        # Add dimentions and coordinates needed to be merged
        dataset = dataset.expand_dims(
            dim=["model", "species", "level", "leadtime"], axis=[0, 1, 2, 3]
        )
        dataset = dataset.assign_coords(
            {
                "model": [model_name],
                "species": [species_name],
                "level": [level],
                "leadtime": [leadtime],
            }
        )

        # 2 of the CTM models are on slightly different grids.
        # We simply redefine their longitude and latitude to be the
        # same as the other models, creating an accepted imprecision.
        if model_name in ("LOTOS", "SILAM"):
            dataset.coords["latitude"] = lat_coordinates.latitude.values
            dataset.coords["longitude"] = lon_coordinates.longitude.values

        # Round latitude coordinates
        rounded_lat = np.round(dataset.latitude.values, decimals=2)
        rounded_lon = np.round(dataset.longitude.values, decimals=2)
        if not np.allclose(dataset.latitude, rounded_lat) or not np.allclose(
            dataset.longitude, rounded_lon
        ):
            warn(
                "Rounded longitude or latitude is not close to the "
                f"original coordinate for {path}."
            )
        dataset = dataset.assign_coords(
            latitude=np.round(dataset.coords["latitude"].values, decimals=2),
            longitude=np.round(dataset.coords["longitude"].values, decimals=2),
        )

        # Convert from kg/m3 to micro gram per cubic meter
        dataset = dataset.assign_attrs(units="µg/m3")
        dataset *= KILOGRAM_TO_MICROGRAM

        return dataset

    try:
        # Open grib files as xr.Dataset and classify them based on the weather
        # parameter they represent.
        output_dataset = xr.open_mfdataset(
            paths=list(raw_dir.glob(f"**/{run_date_string}*.grib")),
            preprocess=preprocess_input,
            coords="minimal",  # type: ignore[reportArgumentType]
            compat="equals",
            join="outer",
        )

        # Add run_date coordinate
        output_dataset = output_dataset.assign_coords(
            run_date=np.datetime64(run_date_string.replace("_", "-"))
        )

        # Save
        output_dataset.to_netcdf(save_path)

    except gribapi.errors.WrongGridError as error:
        # Catch in consistent input data, and log it
        print(f"Error on {run_date_string}")
        print(error)


def _process_target_month(
    required_dates: list[dt.datetime],
    raw_dir: Path,
    processed_dir: Path,
    levels: list[str],
) -> None:
    """Processes some raw netcdf files of monthly target (reanalysis) data.
    Split the reanalysis monthly files into one file for each hour of the month
    they contain.

    Args:
        required_dates: List of dates expected to be extracted
            from one month of reanalysis.
        raw_dir: Path to the dir containing the downloaded dataset.
        processed_dir: Path to the dir where the processed dataset
            will be written.
        levels: List of levels to extract from raw data.
    """

    # Define the date month
    date_month = dt.date(required_dates[0].year, required_dates[0].month, 1)

    # Find the netcdf files for the given month
    file_paths: list[Path] = list(
        raw_dir.glob(f"ensemble/**/{date_month.year}_{date_month.month:02}_*m.netcdf")
    )
    if file_paths == []:
        return

    # Open all the month's netcdf files, one for each weather parameter
    month_dataarrays: list[xr.DataArray] = []
    for file_path in file_paths:
        # Load data array
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

        # Concat to month dataaray
        month_dataarrays.append(data_array)

    # Validate
    if len(month_dataarrays) == 0:
        raise FileNotFoundError(f"File not founds for input {date_month}.")

    # Save a target file for each dates required
    for date in required_dates:
        # Check if output path exists
        save_path = processed_dir / "target" / f"{date.strftime(r'%Y_%m_%d_%H')}.netcdf"
        if save_path.exists():
            continue

        # Select the right date for each dataaray
        hour_dataarray = xr.concat(
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

        # Save
        hour_dataarray.name = date.strftime(r"%Y_%m_%d_%H reanalisis")
        hour_dataarray.to_netcdf(save_path)


def process(dataset_dir: Path, nb_jobs: int = 15, overwrite: bool = False) -> None:
    """Prepares a CAMS dataset for use in training.

    Args:
        dataset_dir: Path to the dataset dir.
        nb_jobs: Number of parallel jobs to use for preprocessing.
            Defaults to 15.
        overwrite: If True, will remove existing files in the output dir.
    """

    # Create paths
    raw_dir: Path = dataset_dir / "raw"
    processed_dir: Path = dataset_dir / "processed"

    # Overwrite
    if overwrite:
        for file_path in processed_dir.glob("**/*.netcdf"):
            file_path.unlink()

    # Verify the input dir hierarchy
    if not all(
        dir in list(PMACC_MODEL_NAMES) + ["ensemble"] for dir in os.listdir(raw_dir)
    ):
        raise ValueError("The dir given to process has an unknown file structure.")

    # Create dirs
    (processed_dir / "input").mkdir(exist_ok=True, parents=True)
    (processed_dir / "target").mkdir(exist_ok=True, parents=True)

    # Gather dates
    run_date_strings: set[str] = set(
        file_path.stem[:10] for file_path in raw_dir.glob(r"**/*.grib")
    )

    # ---------------------------------------------------------------------
    # -------                      input                           --------
    # ---------------------------------------------------------------------

    # Open reference MACCGE01 grid.
    with open("data/MACCGE01.pkl", "br") as file:
        lat, lon = pkl.load(file)

    # Process the input with parallel jobs.
    joblib.Parallel(n_jobs=nb_jobs)(
        joblib.delayed(_process_input_date)(
            run_date_string,
            raw_dir,
            processed_dir,
            lat,
            lon,
        )
        for run_date_string in tqdm(run_date_strings, desc="Input processing")
    )

    # Extract leadtimes from the input file just processed
    input_sample_path = list((processed_dir / "input").glob("*.netcdf"))[0]
    input_sample = xr.load_dataarray(input_sample_path)
    leadtimes = [int(leadtime) for leadtime in input_sample.coords["leadtime"].values]

    # ---------------------------------------------------------------------
    # -------                    target                            --------
    # ---------------------------------------------------------------------

    # Gather months existing in the input
    required_dates: set[dt.datetime] = set(
        dt.datetime.strptime(path.stem[:10], r"%Y_%m_%d") + dt.timedelta(hours=leadtime)
        for leadtime in leadtimes
        for path in processed_dir.glob(r"input/*.netcdf")
    )
    required_months: set[dt.datetime] = set(
        dt.datetime.strptime(path.stem[:7], r"%Y_%m")
        for path in processed_dir.glob(r"input/*.netcdf")
    )

    # Process the target with parallel jobs.
    joblib.Parallel(n_jobs=nb_jobs)(
        joblib.delayed(_process_target_month)(
            required_dates=[
                date
                for date in required_dates
                if (date.year == date_month.year and date.month == date_month.month)
            ],
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            levels=input_sample.coords["level"].values,
        )
        for date_month in tqdm(required_months, desc="Target processing")
    )

    # ---------------------------------------------------------------------
    # -------                   cleanup                            --------
    # ---------------------------------------------------------------------

    # Delete processed input files that do not have an associated target file
    for input_path in tqdm(
        list((processed_dir / "input").glob("*.netcdf")), desc="Cleanup"
    ):
        date = dt.datetime.strptime(input_path.stem, r"%Y_%m_%d")
        if not all(
            (
                processed_dir
                / "target"
                / (date + dt.timedelta(hours=leadtime)).strftime(r"%Y_%m_%d_%H.netcdf")
            ).exists()
            for leadtime in leadtimes
        ):
            input_path.unlink()


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
        raw_dir=dataset_dir / "raw",
        plot_save_path=plot_save_path,
    )

    # Process raw dataset
    process(
        dataset_dir=dataset_dir,
        nb_jobs=nb_jobs,
        overwrite=overwrite,
    )
