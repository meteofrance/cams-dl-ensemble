import re
from collections import defaultdict
from pathlib import Path

import dayplot as dp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize

from cams.settings import CAMS_DATASET_DIR, MODEL_NAMES

INPUT_MODELS = MODEL_NAMES

FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
    r"_(?P<species>[A-Z0-9_]+?)_(?P<lt>\d{2})h_(?P<level>[A-Z0-9_m]+)(\.grib)?$"
)


def iter_grib_files(model_dir: Path):
    """Yield (date_key, filename) for each valid .grib file in model_dir.

    Args:
        model_dir: Path to the model directory.

    Yields:
        Tuple of (date_key, filename) where date_key is YYYY-MM-DD.
    """
    for f in model_dir.glob("*.grib"):
        m = FILENAME_RE.match(f.name)
        if not m:
            continue
        date_key = f"{m.group('year')}-{m.group('month')}-{m.group('day')}"
        yield date_key, f.name


def scan_model_presence(model_dir: Path) -> dict[str, int]:
    """Return {date: 1} for every day that has at least one .grib file.

    Args:
        model_dir: Path to the model directory.

    Returns:
        A dictionary with every date that contains at least 1 .grib file.
    """
    dates_seen: set[str] = set()
    for date_key, _ in iter_grib_files(model_dir):
        dates_seen.add(date_key)
    return {d: 1 for d in dates_seen}


def scan_model_count_per_day(dataset_dir: Path) -> dict[str, int]:
    """Return {date: number_of_models_present} for all input models.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        A dictionary with the count of models that have at least 1 file per date.
    """
    model_counts: dict[str, int] = defaultdict(int)
    for model in INPUT_MODELS:
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
        A dictionary with the count of .grib files from every model for a date.
    """
    total: dict[str, int] = defaultdict(int)
    for model in INPUT_MODELS:
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
        A dictionary with the count of distinct species per date.
    """
    species_per_day: dict[str, set[str]] = defaultdict(set)
    for model in INPUT_MODELS:
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
        A dictionary with years as keys and {date: value} as values.
    """
    by_year: dict[str, dict[str, int]] = defaultdict(dict)
    for date, value in counts.items():
        by_year[date[:4]][date] = value
    return dict(by_year)


def _plot_calendar(
    counts: dict[str, int],
    output_path: Path,
    title: str,
    cmap: str,
    vmax: int,
    colorbar_label: str,
    color_for_none: str = "white",
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
    by_year = split_by_year(counts)
    years = sorted(by_year.keys())
    n_years = len(years)

    fig, axes = plt.subplots(nrows=n_years, ncols=1, figsize=(18, n_years * 3), dpi=150)
    if n_years == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, year in zip(axes, years):
        dp.calendar(
            dates=list(by_year[year].keys()),
            values=list(by_year[year].values()),
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            edgewidth=0.1,
            color_for_none=color_for_none,
            ax=ax,
        )
        ax.text(s=year, x=-4, y=3.5, size=30, rotation=90, color="#aaa", va="center")

    # Single shared colorbar at the bottom
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
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
        cmap="turbo",
        vmax=len(INPUT_MODELS),
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
        cmap="turbo",
        vmax=max(counts.values()),
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
        cmap="turbo",
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize a CAMS dataset raw folder.",
    )
    parser.add_argument(
        "--dataset-dir",
        "-i",
        type=Path,
        default=CAMS_DATASET_DIR,
        help="Path to the dataset dir. Defaults to value in settings.py",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./data"),
        help="Where all the plots will be saved. Defaults to ./data",
    )

    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir
    output_dir: Path = args.output_dir

    print(f"Scanning {dataset_dir} ...\n")

    all_presence = scan_model_count_per_day(dataset_dir)
    total_counts = scan_total_counts(dataset_dir)
    species_count = scan_species_per_day(dataset_dir)

    print(f"\nPlotting in {output_dir} ...\n")

    plot_presence(all_presence, output_path=output_dir / "plot1_presence.png")
    plot_counts(total_counts, output_path=output_dir / "plot2_counts.png")
    plot_species(species_count, output_path=output_dir / "plot3_species.png")

    print("\nReports\n")
    report_incomplete_days(species_count, max_species=18)
