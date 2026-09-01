import re
from collections import defaultdict
from pathlib import Path

import dayplot as dp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from cams.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from cams.types import MODELS_NAMES

NETCDF_INPUT_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<month>\d{2})"
    r"_(?P<day>\d{2})\.netcdf?$"
)
NETCDF_TARGET_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<month>\d{2})"
    r"_(?P<day>\d{2})_(?P<hour>\d{2})\.netcdf?$"
)

FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
    r"_(?P<species>[A-Z0-9_]+?)_(?P<lt>\d{2})h_(?P<level>[A-Z0-9_m]+)(\.grib)?$"
)


def make_status_cmap(vmax: int):
    """Discrete colormap with cyan for vmax.

    Args:
        vmax: The maximum value

    Returns (cmap, norm) ready to pass to dayplot / matplotlib.
    """
    base_colors = [
        (0.86, 0.16, 0.16),
        (0.90, 0.35, 0.10),
        (0.92, 0.55, 0.08),
        (0.93, 0.72, 0.06),
        (0.87, 0.87, 0.07),
        (0.65, 0.85, 0.08),
        (0.45, 0.80, 0.08),
        (0.28, 0.75, 0.08),
        (0.12, 0.68, 0.12),
        (0.06, 0.58, 0.16),
        (0, 1, 1),
    ]
    n = len(base_colors)

    boundaries = [round(i / (n - 2) * (vmax)) for i in range(n - 1)] + [vmax]
    boundaries = sorted(set(boundaries))
    cmap = LinearSegmentedColormap.from_list(
        "status", base_colors, N=len(boundaries) - 1
    )
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    return cmap, norm


def iter_grib_files(model_dir: Path):
    """Yield (date_key, filename) for each valid .grib file in model_dir.

    Args:
        model_dir: Path to the model directory.

    Yields:
        Tuple of (date_key, filename) where date_key is YYYY-MM-DD.
    """
    for path in model_dir.glob("*.grib"):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        date_key = f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
        yield date_key, path.name


def scan_model_presence(model_dir: Path) -> set[str]:
    """Return a set of dates that contains at least one .grib file.

    Args:
        model_dir: Path to the model directory.

    Returns:
        set[str]: Set of every date that contains at least 1 .grib file.
    """
    dates_seen: set[str] = set()
    for date_key, _ in iter_grib_files(model_dir):
        dates_seen.add(date_key)
    return dates_seen


def scan_model_count_per_day(dataset_dir: Path) -> dict[str, int]:
    """Return {date: number_of_models_present} for all input models.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        dict[str, int]: Dictionary with the count of models
        that have at least 1 file per date.
    """
    model_counts: dict[str, int] = defaultdict(int)
    for model in MODELS_NAMES:
        model_dir = dataset_dir / ("PMACC" + model)
        presence = scan_model_presence(model_dir)
        print(f"  {model:<12} -> {len(presence)} days with files")
        for date in presence:
            model_counts[date] += 1
    return dict(model_counts)


def scan_total_counts(dataset_dir: Path) -> dict[str, int]:
    """Return {date: total_file_count} for all input models.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        dict[str, int]: Dictionary with the count of .grib files
        from every model for a date.
    """
    total: dict[str, int] = defaultdict(int)
    for model in MODELS_NAMES:
        model_dir = dataset_dir / ("PMACC" + model)
        file_count = 0
        for date_key, _ in iter_grib_files(model_dir):
            total[date_key] += 1
            file_count += 1
        print(f"  {model:<12} -> {file_count} files")
    return dict(total)


def scan_species_per_day(dataset_dir: Path) -> dict[str, int]:
    """Return {date: number_of_distinct_species} across all input models.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        dict[str, int]: Dictionary with the count of distinct species per date.
    """
    species_per_day: dict[str, set[str]] = defaultdict(set)
    for model in MODELS_NAMES:
        model_dir = dataset_dir / ("PMACC" + model)
        for date_key, filename in iter_grib_files(model_dir):
            m = FILENAME_RE.match(filename)
            if m:
                species_per_day[date_key].add(m.group("species"))
    return {date: len(species) for date, species in species_per_day.items()}


def infer_date_range(counts: dict[str, int]) -> tuple[str, str]:
    """Return (min_date, max_date) from a {date: value} dict.

    Args:
        counts: A dictionary with dates as keys.

    Returns:
        A tuple (start_date, end_date) as YYYY-MM-DD strings.
    """
    dates = sorted(counts.keys())
    return dates[0], dates[-1]


def split_by_year(counts: dict[str, int]) -> dict[str, dict[str, int]]:
    """Split a {date: value} dict into {year: {date: value}}.

    Args:
        counts: A dictionary with dates as YYYY-MM-DD keys.

    Returns:
        dict[str, dict[str, int]]: Dictionary with years as keys
        and {date: value} as values.
    """
    by_year: dict[str, dict[str, int]] = defaultdict(dict)
    for date, value in counts.items():
        by_year[date[:4]][date] = value
    return dict(by_year)


def _plot_calendar(
    counts: dict[str, int],
    output_path: Path,
    title: str,
    vmax: int,
    colorbar_label: str,
) -> None:
    """Calendar heatmap with one row per year — save to output_path.

    Args:
        counts: {date: value} dict to plot.
        output_path: Path where the PNG will be saved.
        title: Figure suptitle.
        cmap: Matplotlib colormap name.
        vmax: Maximum value for the colorbar scale (shared across all years).
        colorbar_label: Label for the horizontal colorbar.
        color_for_none: Color for None values
    """
    cmap, norm = make_status_cmap(vmax)
    by_year = split_by_year(counts)
    years = sorted(by_year.keys())
    n_years = len(years)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, axes = plt.subplots(nrows=n_years, ncols=1, figsize=(18, n_years * 3), dpi=150)
    if n_years == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, year in zip(axes, years):
        patches = dp.calendar(
            dates=list(by_year[year].keys()),
            values=list(by_year[year].values()),
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            edgewidth=0.1,
            color_for_none="#ffcccc",
            ax=ax,
        )

        date_to_value = by_year[year]
        # Colors are overwrited due to calendar not handling normalization
        dates_sorted = sorted(date_to_value.keys())
        for patch, date in zip(patches, dates_sorted):
            rgba = scalar_mappable.to_rgba(np.ndarray([date_to_value[date]]))
            patch.set_facecolor(rgba)

        ax.text(s=year, x=-4, y=3.5, size=30, rotation=90, color="#aaa", va="center")

    # Single shared colorbar at the bottom
    scalar_mappable.set_array([])
    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes[-1],
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
        aspect=30,
        anchor=(1.0, 1.0),
    )
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved -> {output_path}")


def plot_presence(model_counts: dict[str, int], output_path: Path) -> None:
    """Save a calendar heatmap showing how many models are present per day (0-11).

    Args:
        model_counts: {date: model_count} as returned by scan_model_count_per_day.
        output_path: Path where the PNG will be saved.
    """
    _plot_calendar(
        counts=model_counts,
        output_path=output_path,
        title="Number of models present per day\n"
        "(at least 1 file — all species / levels / leadtimes combined)",
        vmax=len(MODELS_NAMES),
        colorbar_label="Number of models",
    )


def plot_counts(counts: dict[str, int], output_path: Path) -> None:
    """Save a calendar heatmap showing total file count per day across all models.

    Args:
        counts: {date: total_file_count} as returned by scan_total_counts.
        output_path: Path where the PNG will be saved.
    """
    _plot_calendar(
        counts=counts,
        output_path=output_path,
        title="Total file volume per day\n(all models x lt x levels x species "
        f"- max: {max(counts.values()):,} files/day)",
        vmax=38016,
        colorbar_label="File count",
    )


def plot_species(species_counts: dict[str, int], output_path: Path) -> None:
    """Save a calendar heatmap showing distinct species count per day (0-18).

    Args:
        species_counts: {date: species_count} as returned by scan_species_per_day.
        output_path: Path where the PNG will be saved.
    """
    _plot_calendar(
        counts=species_counts,
        output_path=output_path,
        title="Number of distinct species per day\n"
        "(across all models / levels / leadtimes)",
        vmax=18,
        colorbar_label="Distinct species count",
    )


def report_incomplete_days(
    species_counts: dict[str, int], max_species: int = 18
) -> None:
    """Print a summary of days where the maximum species count is not reached.

    Args:
        species_counts: {date: species_count} as returned by scan_species_per_day.
        max_species: Expected maximum number of species.
    """
    incomplete = {
        date: count for date, count in species_counts.items() if count < max_species
    }
    total = len(species_counts)
    n_incomplete = len(incomplete)

    print(f"\n{'=' * 40}")
    print("Species coverage report")
    print(f"{'=' * 40}")
    print(f"  Total days scanned : {total}")
    print(f"  Complete days (={max_species} species) : {total - n_incomplete}")
    print(f"  Incomplete days    : {n_incomplete}")

    if incomplete:
        print("\n  Detail (sorted by date):")
        for date in sorted(incomplete):
            print(f"    {date} -> {incomplete[date]:>2}/{max_species} species")


def scan_netcdf_input(processed_dir: Path) -> dict[str, int]:
    """Return {date: 1} for every day that has exactly 1 input NetCDF file.

    Files are expected to match YYYY_MM_DD.netcdf in processed_dir/input.

    Args:
        processed_dir: Path to the processed data directory.

    Returns:
        dict[str, int]: Dictionary mapping each date found to
        its file count (should be 1).
    """
    input_dir = processed_dir / "input"
    counts: dict[str, int] = defaultdict(int)
    for f in input_dir.glob("*.netcdf"):
        m = NETCDF_INPUT_RE.match(f.name)
        if not m:
            continue
        date_key = f"{m.group('year')}-{m.group('month')}-{m.group('day')}"
        counts[date_key] += 1
    return dict(counts)


def scan_netcdf_target(processed_dir: Path) -> dict[str, int]:
    """Return {date: file_count} for target NetCDF files (expected 24/day).

    Files are expected to match YYYY_MM_DD_HH.netcdf in processed_dir/target.

    Args:
        processed_dir: Path to the processed data directory.

    Returns:
        dict[str, int]: Dictionary mapping each date to
        its hourly file count (expected 24).
    """
    target_dir = processed_dir / "target"
    counts: dict[str, int] = defaultdict(int)
    for f in target_dir.glob("*.netcdf"):
        m = NETCDF_TARGET_RE.match(f.name)
        if not m:
            continue
        date_key = f"{m.group('year')}-{m.group('month')}-{m.group('day')}"
        counts[date_key] += 1
    return dict(counts)


def plot_netcdf_input(counts: dict[str, int], output_path: Path) -> None:
    """Save a calendar heatmap showing input NetCDF file count per day (expected: 1).

    Args:
        counts: {date: file_count} as returned by scan_netcdf_input.
        output_path: Path where the PNG will be saved.
    """
    _plot_calendar(
        counts=counts,
        output_path=output_path,
        title="Processed input — NetCDF file count per day\n"
        "(expected: 1 file per day — YYYY_MM_DD.netcdf)",
        vmax=1,
        colorbar_label="File count (expected: 1)",
    )


def plot_netcdf_target(counts: dict[str, int], output_path: Path) -> None:
    """Save a calendar heatmap showing target NetCDF file count per day (expected: 24).

    Args:
        counts: {date: file_count} as returned by scan_netcdf_target.
        output_path: Path where the PNG will be saved.
    """
    _plot_calendar(
        counts=counts,
        output_path=output_path,
        title="Processed target — NetCDF file count per day\n"
        "(expected: 24 files per day — YYYY_MM_DD_HH.netcdf)",
        vmax=24,
        colorbar_label="File count (expected: 24)",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize CAMS raw and processed dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./data"),
        help="Where all the plots will be saved. Defaults to ./data",
    )

    args = parser.parse_args()
    output_dir: Path = args.output_dir

    # --- Raw GRIB scans (RAW_DATA_DIR) ---
    print(f"Scanning raw data in {RAW_DATA_DIR} ...\n")

    all_presence = scan_model_count_per_day(RAW_DATA_DIR)
    total_counts = scan_total_counts(RAW_DATA_DIR)
    species_count = scan_species_per_day(RAW_DATA_DIR)

    # --- Processed NetCDF scans (PROCESSED_DATA_DIR) ---
    print(f"\nScanning processed data in {PROCESSED_DATA_DIR} ...\n")

    netcdf_input_counts = scan_netcdf_input(PROCESSED_DATA_DIR)
    netcdf_target_counts = scan_netcdf_target(PROCESSED_DATA_DIR)

    print(f"  input  -> {len(netcdf_input_counts)} days with files")
    print(f"  target -> {len(netcdf_target_counts)} days with files")

    # --- Plots ---
    print(f"\nPlotting in {output_dir} ...\n")

    plot_presence(all_presence, output_path=output_dir / "plot1_presence.png")
    plot_counts(total_counts, output_path=output_dir / "plot2_counts.png")
    plot_species(species_count, output_path=output_dir / "plot3_species.png")
    plot_netcdf_input(
        netcdf_input_counts, output_path=output_dir / "plot4_netcdf_input.png"
    )
    plot_netcdf_target(
        netcdf_target_counts, output_path=output_dir / "plot5_netcdf_target.png"
    )

    # --- Reports ---
    print("\nReports\n")
    report_incomplete_days(species_count, max_species=18)
