"""Tests of excluding columns before feature selection."""

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from mloom.columns import check_columns, plan_columns
from mloom.config import Config
from mloom.errors import InputError
from mloom.sensors import UnavailableSensors

NO_SENSORS = UnavailableSensors(path=None, found=False, reasons={})


def _sensors(**reasons: str | None) -> UnavailableSensors:
    return UnavailableSensors(path="x.json", found=True, reasons=reasons)


def test_each_column_gets_its_reason(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Every reason of "列の除外" is recorded for the column it applies to."""
    data = frame.with_columns(
        empty=pl.lit(None, dtype=pl.Float64),
        sparse=pl.Series(np.where(np.arange(frame.height) % 4 == 0, np.arange(frame.height), np.nan)),
    )
    config = make_config(cv="group", group_column="lot", exclude_columns=("c", "s5"))
    plan = plan_columns(data, config, _sensors(s4="故障", gone=None))
    reasons = {column.name: column.reason for column in plan.excluded}
    assert reasons == {
        "y": "target",
        "lot": "role",
        "c": "user_excluded",
        "s5": "user_excluded",
        "s4": "unavailable_sensor",
        "text": "non_numeric",
        "empty": "all_null",
        "const": "constant",
        "sparse": "high_missing",
    }
    assert plan.candidates == ["t", "s0", "s1", "s2", "s3"]
    assert plan.sensors_not_in_data == ["gone"]


def test_unavailable_sensor_keeps_its_reason(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """The reason in the JSON file is kept as the detail."""
    plan = plan_columns(frame, make_config(), _sensors(s0="故障"))
    [excluded] = [column for column in plan.excluded if column.name == "s0"]
    assert excluded.detail == "故障"


def test_no_candidate_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """When every column is excluded, ``InputError`` is raised."""
    config = make_config(exclude_columns=("t", "s0", "s1", "s2", "s3", "s4", "s5", "c"))
    with pytest.raises(InputError, match="no column is left"):
        plan_columns(frame, config, NO_SENSORS)


@pytest.mark.parametrize(
    ("overrides", "sensors", "message"),
    [
        ({"target": "missing"}, NO_SENSORS, "'missing' is not in the data"),
        ({"cv": "timeseries", "time_column": "time"}, NO_SENSORS, "'time' is not in the data"),
        ({}, _sensors(y="故障"), "listed as an unavailable sensor"),
    ],
)
def test_check_columns_raises(
    frame: pl.DataFrame,
    make_config: Callable[..., Config],
    overrides: dict,
    sensors: UnavailableSensors,
    message: str,
) -> None:
    """Named columns must exist, and the target must not be an unavailable sensor."""
    with pytest.raises(InputError, match=message):
        check_columns(frame, make_config(**overrides), sensors)


def test_role_column_with_missing_values_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """A time or group column with missing values cannot split the rows."""
    data = frame.with_columns(lot=pl.when(pl.col("t") == 0).then(None).otherwise(pl.col("lot")))
    with pytest.raises(InputError, match="has missing values"):
        check_columns(data, make_config(cv="group", group_column="lot"), NO_SENSORS)
