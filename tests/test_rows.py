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
    """Rows 0-2 have no target; rows 1, 3, 4 and 5 hold +-inf in a feature, and row 6 in the target."""
    return frame.with_columns(
        _set("y", [0, 1, 2], float("nan")),
        _set("s0", [1, 3, 4], INF),
        _set("s1", [4, 5], -INF),
    ).with_columns(_set("y", [6], INF))


def test_only_rows_with_an_infinite_target_are_dropped(data: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Rows with +-inf in a feature are kept, labeled or not; a row whose target is +-inf is dropped."""
    rows = select_rows(data, make_config(), NO_SENSORS, use_unlabeled=True)
    assert rows.n_without_target == 3
    assert sorted(rows.unlabeled["t"].to_list()) == [0, 1, 2]
    assert rows.n_with_infinite_target == 1
    assert rows.infinite_by_column == {"s0": 3, "s1": 2, "y": 1}
    assert rows.labeled.height == data.height - 4
    assert 6 not in rows.labeled["t"].to_list()
    assert {3, 4, 5} <= set(rows.labeled["t"].to_list())


def test_unlabeled_rows_can_be_left_out(data: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Without unlabeled rows, only the labeled rows are counted for +-inf."""
    rows = select_rows(data, make_config(), NO_SENSORS, use_unlabeled=False)
    assert rows.unlabeled.height == 0
    assert rows.n_without_target == 3
    assert rows.n_with_infinite_target == 1
    assert rows.infinite_by_column == {"s0": 2, "s1": 2, "y": 1}
    assert rows.labeled.height == data.height - 4


def test_columns_excluded_by_name_are_not_counted(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Infinite values in a user-excluded column or an unavailable sensor are not counted."""
    data = frame.with_columns(_set("s0", [0], INF), _set("s1", [1], INF), _set("s2", [2], INF))
    sensors = UnavailableSensors(path="x.json", found=True, reasons={"s1": None})
    rows = select_rows(data, make_config(exclude_columns=("s0", "c")), sensors, use_unlabeled=True)
    assert rows.infinite_by_column == {"s2": 1}
    assert rows.n_with_infinite_target == 0
    assert rows.labeled.height == frame.height


def test_no_labeled_row_left_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """When every target is +-inf, ``InputError`` is raised; infinite features alone keep the rows."""
    assert select_rows(frame.with_columns(s0=pl.lit(INF)), make_config(), NO_SENSORS, use_unlabeled=True).labeled.height
    with pytest.raises(InputError, match="no row with a target is left"):
        select_rows(frame.with_columns(y=pl.lit(-INF)), make_config(), NO_SENSORS, use_unlabeled=True)


def test_target_without_values_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """A target with no values raises ``InputError``."""
    with pytest.raises(InputError, match="has no values"):
        select_rows(frame.with_columns(y=pl.lit(None, dtype=pl.Float64)), make_config(), NO_SENSORS, use_unlabeled=True)
