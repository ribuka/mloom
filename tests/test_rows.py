"""Tests of choosing the labeled and unlabeled rows."""

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


@pytest.fixture
def data(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows 0-2 have no target; rows 1, 3, 4, 5 and 6 hold +-inf."""
    return frame.with_columns(
        _set("y", [0, 1, 2], float("nan")),
        _set("s0", [1, 3, 4], INF),
        _set("s1", [4, 5], -INF),
    ).with_columns(_set("y", [6], INF))


def test_unlabeled_rows_are_kept_without_infinite(data: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """With unlabeled rows, rows with +-inf are dropped from both sets and counted together."""
    rows = select_rows(data, make_config(), NO_SENSORS, use_unlabeled=True)
    assert rows.n_without_target == 3
    assert sorted(rows.unlabeled["t"].to_list()) == [0, 2]
    assert rows.n_with_infinite == 5
    assert rows.infinite_by_column == {"s0": 3, "s1": 2, "y": 1}
    assert rows.labeled.height == data.height - 7


def test_unlabeled_rows_can_be_left_out(data: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Without unlabeled rows, only the labeled rows are checked for +-inf."""
    rows = select_rows(data, make_config(), NO_SENSORS, use_unlabeled=False)
    assert rows.unlabeled.height == 0
    assert rows.n_without_target == 3
    assert rows.n_with_infinite == 4
    assert rows.infinite_by_column == {"s0": 2, "s1": 2, "y": 1}
    assert rows.labeled.height == data.height - 7


def test_columns_excluded_by_name_are_not_checked(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Infinite values in a user-excluded column or an unavailable sensor keep the row."""
    data = frame.with_columns(_set("s0", [0], INF), _set("s1", [1], INF))
    sensors = UnavailableSensors(path="x.json", found=True, reasons={"s1": None})
    rows = select_rows(data, make_config(exclude_columns=("s0", "c")), sensors, use_unlabeled=True)
    assert rows.n_with_infinite == 0
    assert rows.labeled.height == frame.height


def test_no_labeled_row_left_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """When every labeled row holds an infinite value, ``InputError`` is raised."""
    with pytest.raises(InputError, match="no row with a target is left"):
        select_rows(frame.with_columns(s0=pl.lit(INF)), make_config(), NO_SENSORS, use_unlabeled=True)


def test_target_without_values_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """A target with no values raises ``InputError``."""
    with pytest.raises(InputError, match="has no values"):
        select_rows(frame.with_columns(y=pl.lit(None, dtype=pl.Float64)), make_config(), NO_SENSORS, use_unlabeled=True)
