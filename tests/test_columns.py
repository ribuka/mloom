"""Tests of excluding columns before feature selection."""

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from mloom.columns import check_columns, check_role_values, plan_columns
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
    plan = plan_columns(data, data.clear(), config, _sensors(s4="故障", gone=None), 0.5)
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
    plan = plan_columns(frame, frame.clear(), make_config(), _sensors(s0="故障"), 0.5)
    [excluded] = [column for column in plan.excluded if column.name == "s0"]
    assert excluded.detail == "故障"


@pytest.mark.parametrize(
    ("labeled_missing", "unlabeled_missing", "detail"),
    [
        (0.6, 0.0, "labeled 0.600, all 0.300"),
        (0.2, 1.0, "labeled 0.200, all 0.600"),
        (0.4, 0.4, None),
    ],
)
def test_high_missing_uses_both_rates(
    frame: pl.DataFrame,
    make_config: Callable[..., Config],
    labeled_missing: float,
    unlabeled_missing: float,
    detail: str | None,
) -> None:
    """A column is excluded when its missing rate in the labeled rows or in all rows is above the threshold."""

    def with_missing(rows: pl.DataFrame, rate: float) -> pl.DataFrame:
        n_missing = round(rows.height * rate)
        return rows.with_columns(s0=pl.when(pl.int_range(pl.len()) < n_missing).then(None).otherwise(pl.col("s0")))

    labeled, unlabeled = with_missing(frame.head(60), labeled_missing), with_missing(frame.tail(60), unlabeled_missing)
    plan = plan_columns(labeled, unlabeled, make_config(), NO_SENSORS, 0.5)
    details = {column.name: column.detail for column in plan.excluded if column.reason == "high_missing"}
    assert details.get("s0") == detail


def test_infinite_values_count_as_missing(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """+-inf counts as missing for ``all_null``, ``constant`` and ``high_missing``, in labeled and unlabeled rows."""
    inf = float("inf")
    labeled = frame.head(60).with_columns(
        all_infinite=pl.lit(inf),
        # 1.0 and +-inf only: constant once +-inf is missing.
        one_value=pl.when(pl.int_range(pl.len()) % 2 == 0).then(1.0).otherwise(-inf),
        # +inf in 20% of the labeled rows and 100% of the unlabeled ones: all 0.6.
        mostly_infinite=pl.when(pl.int_range(pl.len()) < 12).then(inf).otherwise(pl.col("s0")),
    )
    unlabeled = frame.tail(60).with_columns(all_infinite=pl.lit(1.0), one_value=pl.lit(1.0), mostly_infinite=pl.lit(inf))
    plan = plan_columns(labeled, unlabeled, make_config(), NO_SENSORS, 0.5)
    excluded = {column.name: (column.reason, column.detail) for column in plan.excluded}
    assert excluded["all_infinite"] == ("all_null", None)
    assert excluded["one_value"] == ("constant", None)
    assert excluded["mostly_infinite"] == ("high_missing", "labeled 0.200, all 0.600")


def test_no_candidate_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """When every column is excluded, ``InputError`` is raised."""
    config = make_config(exclude_columns=("t", "s0", "s1", "s2", "s3", "s4", "s5", "c"))
    with pytest.raises(InputError, match="no column is left"):
        plan_columns(frame, frame.clear(), config, NO_SENSORS, 0.5)


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
        check_role_values(data, make_config(cv="group", group_column="lot"))
