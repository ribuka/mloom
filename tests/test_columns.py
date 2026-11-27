"""Tests of excluding columns before feature selection."""

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from mloom.columns import (
    check_columns,
    check_role_values,
    plan_columns,
    require_candidates,
)
from mloom.config import Config
from mloom.errors import InputError
from mloom.sensors import EXACT, SensorEntry, UnavailableSensors

NO_SENSORS = UnavailableSensors(path=None, found=False, entries=[])


def _sensors(**reasons: str | None) -> UnavailableSensors:
    # Entries matched exactly, named by the keywords.
    return _entries(*(SensorEntry(name, EXACT, reason) for name, reason in reasons.items()))


def _entries(*entries: SensorEntry) -> UnavailableSensors:
    return UnavailableSensors(path="x.json", found=True, entries=list(entries))


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
    assert plan.sensor_columns == [["s4"], []]


@pytest.mark.parametrize(("model", "categorical"), [("lightgbm", True), ("ridge", False), ("pls", False), ("random_forest", False)])
def test_string_columns_are_categorical_candidates_of_lightgbm_only(
    make_config: Callable[..., Config], model: str, categorical: bool
) -> None:
    """String, Categorical and Enum columns follow the numeric candidates for ``lightgbm``, and are ``non_numeric`` otherwise."""
    values = ["a", "b", None, "c"] * 5
    frame = pl.DataFrame(
        {
            "enum": pl.Series(values, dtype=pl.Enum(["a", "b", "c"])),
            "x": np.arange(20.0),
            "string": values,
            "flag": [True, False] * 10,
            "category": pl.Series(values, dtype=pl.Categorical),
            "y": np.arange(20.0),
        }
    )
    plan = plan_columns(frame, frame.clear(), make_config(model=model), NO_SENSORS, 0.5)
    strings = ["enum", "string", "category"]
    assert plan.numeric == ["x", "flag"]
    assert plan.categorical == (strings if categorical else [])
    assert plan.candidates == ["x", "flag", *plan.categorical]
    reasons = {column.name: column.reason for column in plan.excluded}
    non_numeric = {} if categorical else dict.fromkeys(strings, "non_numeric")
    assert reasons == {"y": "target", **non_numeric}


def test_categorical_candidates_get_the_reasons_of_their_values(make_config: Callable[..., Config]) -> None:
    """A string column that is all null, has one value apart from null, or misses too many values is excluded."""
    labeled = pl.DataFrame(
        {
            "empty": pl.Series([None] * 10, dtype=pl.String),
            "one": pl.Series(["a", None] * 5, dtype=pl.Categorical),
            "sparse": ["a", "b", None, None, None, None, None, None, None, None],
            # Not too many missing in the labeled rows (0.4), but too many with the unlabeled ones (0.7).
            "sparse_unlabeled": ["a", "b", "c", "d", "e", "f", None, None, None, None],
            "kept": ["a", "b"] * 5,
            "y": np.arange(10.0),
        }
    )
    unlabeled = labeled.with_columns(sparse_unlabeled=pl.lit(None, dtype=pl.String), y=pl.lit(None, dtype=pl.Float64))
    plan = plan_columns(labeled, unlabeled, make_config(model="lightgbm"), NO_SENSORS, 0.5)
    reasons = {column.name: column.reason for column in plan.excluded}
    assert reasons == {
        "empty": "all_null",
        "one": "constant",
        "sparse": "high_missing",
        "sparse_unlabeled": "high_missing",
        "y": "target",
    }
    assert plan.categorical == ["kept"]


def test_unavailable_sensor_keeps_its_reason(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """The reason in the JSON file is kept as the detail."""
    plan = plan_columns(frame, frame.clear(), make_config(), _sensors(s0="故障"), 0.5)
    [excluded] = [column for column in plan.excluded if column.name == "s0"]
    assert excluded.detail == "故障"


def test_every_column_an_entry_matches_is_excluded(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Each way of matching excludes all the columns it matches, with the reason of the first entry that matches."""
    data = frame.rename({"s0": "TI-101_mean", "s1": "TI-101_max", "s2": "TI-1010", "s3": "FI-2_raw", "s4": "xTESTx"})
    sensors = _entries(
        SensorEntry("TI-101_", "starts_with", "故障"),
        SensorEntry("_max", "ends_with", "後に書いた行"),
        SensorEntry("_raw", "ends_with", None),
        SensorEntry("TEST", "contains", "試験"),
        SensorEntry("t", EXACT, None),
        SensorEntry("TI-101", EXACT, None),
    )
    plan = plan_columns(data, data.clear(), make_config(cv="timeseries", time_column="t"), sensors, 0.5)
    details = {column.name: column.detail for column in plan.excluded if column.reason == "unavailable_sensor"}
    assert details == {"TI-101_mean": "故障", "TI-101_max": "故障", "FI-2_raw": None, "xTESTx": "試験"}
    assert plan.candidates == ["TI-1010", "s5"]
    # The role column t is matched too, but excluded as a role.
    assert [column.reason for column in plan.excluded if column.name == "t"] == ["role"]
    assert plan.sensor_columns == [["TI-101_mean", "TI-101_max"], ["TI-101_max"], ["FI-2_raw"], ["xTESTx"], ["t"], []]


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
    plan = plan_columns(frame, frame.clear(), config, NO_SENSORS, 0.5)
    assert plan.candidates == []
    with pytest.raises(InputError, match="no column is left"):
        require_candidates(plan)


@pytest.mark.parametrize(
    ("overrides", "sensors", "message"),
    [
        ({"target": "missing"}, NO_SENSORS, "'missing' is not in the data"),
        ({"cv": "timeseries", "time_column": "time"}, NO_SENSORS, "'time' is not in the data"),
        ({}, _sensors(y="故障"), "'y' is an unavailable sensor: it matches 'y' \\(match 'exact'\\)"),
        ({}, _entries(SensorEntry("y", "starts_with", None)), "matches 'y' \\(match 'starts_with'\\)"),
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
