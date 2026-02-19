from pathlib import Path

from scripts.data.compute_stats import get_list_samples, compute_stats

PROCESSED_DIR = Path("tests/data/")

def test_compute_stats():

    samples = get_list_samples(PROCESSED_DIR)

    assert len(samples) == 2

    stats = compute_stats(samples)

    assert stats == {"O3": {"min": 0.07472377270460129, "max": 99.25}}
