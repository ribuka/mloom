"""Tests of choosing the training rows."""

from collections.abc import Callable

import polars as pl
import pytest

from mloom.config import Config
from mloom.errors import InputError
from mloom.rows import select_rows
from mloom.sensors import UnavailableSensors

NO_SENSORS = UnavailableSensors(path=None, found=False, reasons={})
INF = float("inf")


def _set(column: str, rows: list[int], value: float | None) -> pl.Expr:
    return pl.when(pl.col("t").is_in(rows)).then(value).otherwise(pl.col(column)).alias(column)


def test_drops_rows_without_target_then_rows_with_infinite(
    frame: pl.DataFrame, make_config: Callable[..., Config]
) -> None:
    """Rows without a target are dropped first; rows with +-inf in the target or a feature are dropped next."""
    data = frame.with_columns(
        _set("y", [0, 1], float("nan")),
        _set("s0", [1, 2, 3], INF),
        _set("s1", [3, 4], -INF),
    ).with_columns(_set("y", [5], INF))
    rows = select_rows(data, make_config(), NO_SENSORS)
    assert rows.n_without_target == 2
    # Row 1 has no target; rows 2, 3, 4 and 5 have an infinite value.
    assert rows.n_with_infinite == 4
    assert rows.infinite_by_column == {"s0": 2, "s1": 2, "y": 1}
    assert rows.frame.height == frame.height - 6


def test_columns_excluded_by_name_are_not_checked(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Infinite values in a user-excluded column or an unavailable sensor keep the row."""
    data = frame.with_columns(_set("s0", [0], INF), _set("s1", [1], INF))
    sensors = UnavailableSensors(path="x.json", found=True, reasons={"s1": None})
    rows = select_rows(data, make_config(exclude_columns=("s0", "c")), sensors)
    assert rows.n_with_infinite == 0
    assert rows.frame.height == frame.height


def test_no_row_left_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """When every row holds an infinite value, ``InputError`` is raised."""
    with pytest.raises(InputError, match="no row is left"):
        select_rows(frame.with_columns(s0=pl.lit(INF)), make_config(), NO_SENSORS)


def test_target_without_values_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """A target with no values raises ``InputError``."""
    with pytest.raises(InputError, match="has no values"):
        select_rows(frame.with_columns(y=pl.lit(None, dtype=pl.Float64)), make_config(), NO_SENSORS)
