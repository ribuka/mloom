"""The end of the training period (``training_rows.last_time``) and the rows after it.

See "スコア" in ``docs/spec.md``. The values of ``time_column`` are compared
in the order the ``timeseries`` CV sorts them: numbers and booleans by value,
strings (``Categorical`` and ``Enum`` as strings) by their text, dates by
day, and datetimes by microsecond, as the row hashes read them
(``mloom.row_hash``). A datetime of a column with a time zone is kept in UTC.
"""

from datetime import UTC, date, datetime
from decimal import Decimal

import numpy as np
import polars as pl

from mloom.dtypes import BOOLEAN, DATE, DATETIME, NUMBER, STRING, value_kind
from mloom.errors import InputError

TimeValue = datetime | date | int | float | Decimal | bool | str


def last_time(rows: pl.DataFrame, time_column: str | None) -> TimeValue | None:
    """Find the latest time of the rows the final model was fitted on.

    Parameters
    ----------
    rows : pl.DataFrame
        The rows the final model was fitted on; ``time_column`` has no
        missing value (``mloom.columns.check_role_values``).
    time_column : str | None
        ``time_column`` of the config.

    Returns
    -------
    TimeValue | None
        The largest value as a plain Python value, or ``None`` without
        ``time_column``. A datetime is cast to microseconds as the rows of
        the data are when they are compared (``after_training``), so that a
        row trained on with a finer time (``ns``) is not later than it.
    """
    if time_column is None:
        return None
    value = _comparable(rows.schema[time_column], time_column).max()
    latest = rows.select(value).item()
    if isinstance(latest, datetime) and latest.tzinfo is not None:
        return latest.astimezone(UTC)
    return latest


def after_training(frame: pl.DataFrame, time_column: str, latest: TimeValue) -> np.ndarray:
    """Tell the rows whose time is later than the end of the training period.

    Parameters
    ----------
    frame : pl.DataFrame
        The data to predict.
    time_column : str
        ``training_rows.time_column`` of the model.
    latest : TimeValue
        ``training_rows.last_time`` of the model.

    Returns
    -------
    np.ndarray
        Boolean mask; a row whose time is missing (or NaN) is not later.

    Raises
    ------
    InputError
        When ``time_column`` is not in the data, or its values are not of
        the kind of ``latest`` (a datetime with and without a time zone are
        of different kinds).
    """
    if time_column not in frame.columns:
        raise InputError(f"time_column {time_column!r} is not in the data; the scores need it")
    dtype = frame.schema[time_column]
    if not _same_kind(dtype, latest):
        raise InputError(
            f"time_column {time_column!r} is of type {dtype}, which cannot be compared with the end of "
            f"the training period {latest!r}"
        )
    later = _comparable(dtype, time_column).gt(pl.lit(latest)).fill_null(value=False)
    return frame.select(later).to_series().to_numpy()


def _comparable(dtype: pl.DataType, name: str) -> pl.Expr:
    column = pl.col(name)
    kind = value_kind(dtype)
    if kind == STRING:
        return column.cast(pl.String)
    if kind == DATETIME:
        column = column.dt.cast_time_unit("us")
        return column.dt.convert_time_zone("UTC") if isinstance(dtype, pl.Datetime) and dtype.time_zone else column
    if dtype.is_float():
        return column.fill_nan(None)
    return column


def _same_kind(dtype: pl.DataType, latest: TimeValue) -> bool:
    kind = value_kind(dtype)
    if isinstance(latest, datetime):
        has_zone = isinstance(dtype, pl.Datetime) and dtype.time_zone is not None
        return kind == DATETIME and has_zone == (latest.tzinfo is not None)
    if isinstance(latest, date):
        return kind == DATE
    if isinstance(latest, bool):
        return kind == BOOLEAN
    if isinstance(latest, int | float | Decimal):
        return kind == NUMBER
    return kind == STRING
