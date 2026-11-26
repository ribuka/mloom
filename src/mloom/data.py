"""Reading the input data into a polars DataFrame."""

from pathlib import Path

import polars as pl

from mloom.errors import InputError

Data = str | Path | pl.DataFrame


def load_data(data: Data) -> tuple[pl.DataFrame, str | None]:
    """Read the input data.

    Parameters
    ----------
    data : Data
        Path of a Parquet file, or a polars DataFrame.

    Returns
    -------
    tuple[pl.DataFrame, str | None]
        The data, and the path it was read from (``None`` for a DataFrame).

    Raises
    ------
    InputError
        When the file does not exist or the data has no rows.
    """
    if isinstance(data, pl.DataFrame):
        frame, path = data, None
    else:
        source = Path(data)
        if not source.is_file():
            raise InputError(f"data file not found: {source}")
        frame, path = pl.read_parquet(source), str(source)
    if frame.height == 0:
        raise InputError("data has no rows")
    return frame, path
