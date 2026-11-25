"""Write a synthetic process-sensor Parquet file for trying mloom.

The data has hourly rows over 30 lots, 30 sensors (``TI-101`` to
``TI-130``), a constant sensor, a mostly missing sensor, a text column, a
numeric target ``quality``, and a class target ``grade``. ``TI-101`` drives
the targets most and is listed in ``examples/unavailable-sensors.json``, so
its exclusion is visible in ``run.json``.

Usage: ``uv run examples/make_sample_data.py [output]`` (default
``data/sample.parquet``).
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

N_ROWS = 3000
N_SENSORS = 30
ROWS_PER_LOT = 100


def make_sample(seed: int = 0) -> pl.DataFrame:
    """Make the synthetic data.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        The sample data.
    """
    rng = np.random.default_rng(seed)
    lot_effect = np.repeat(rng.normal(scale=0.5, size=N_ROWS // ROWS_PER_LOT), ROWS_PER_LOT)
    # A slow drift shared by the sensors, so that neighboring rows are alike.
    drift = np.cumsum(rng.normal(scale=0.05, size=N_ROWS))
    sensors = rng.normal(size=(N_ROWS, N_SENSORS)) + drift[:, None] * rng.uniform(0, 1, N_SENSORS)
    quality = (
        3.0 * sensors[:, 0] + 1.5 * sensors[:, 1] - 1.0 * sensors[:, 2] + 0.5 * sensors[:, 3]
        + lot_effect + rng.normal(scale=0.5, size=N_ROWS)
    )
    mostly_missing = np.where(rng.uniform(size=N_ROWS) < 0.8, np.nan, rng.normal(size=N_ROWS))
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2026, 1, 1), pl.datetime(2026, 1, 1) + pl.duration(hours=N_ROWS - 1), "1h", eager=True
            ),
            "lot": [f"LOT{i // ROWS_PER_LOT:03d}" for i in range(N_ROWS)],
            **{f"TI-{101 + i}": sensors[:, i] for i in range(N_SENSORS)},
            "PI-900": np.full(N_ROWS, 1.0),
            "FI-950": mostly_missing,
            "operator": rng.choice(["A", "B", "C"], size=N_ROWS),
            "quality": quality,
            "grade": np.where(quality > np.median(quality), "good", "bad"),
        }
    )


def main() -> None:
    """Write the sample data to the path given on the command line."""
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    make_sample().write_parquet(output)
    print(output)


if __name__ == "__main__":
    main()
