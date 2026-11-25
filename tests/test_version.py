"""Tests that the package version comes from the VERSION file."""

from pathlib import Path

import mloom


def test_version_matches_version_file() -> None:
    """``mloom.__version__`` is the content of VERSION."""
    expected = (Path(__file__).parents[1] / "VERSION").read_text().strip()
    assert mloom.__version__ == expected
