"""Excluding columns before feature selection, recording why each was excluded.

The reasons and their order are those of "列の除外" in ``docs/spec.md``;
a column gets the first reason that applies.
"""

from dataclasses import dataclass

import polars as pl

from mloom.config import Config
from mloom.errors import InputError
from mloom.pipeline import as_float
from mloom.sensors import UnavailableSensors


@dataclass(frozen=True)
class ExcludedColumn:
    """A column that is not a feature candidate.

    Parameters
    ----------
    name : str
        Column name.
    reason : str
        One of the reasons of "列の除外" in ``docs/spec.md``.
    detail : str | None
        Extra information, such as the reason given in
        ``unavailable-sensors.json``.
    """

    name: str
    reason: str
    detail: str | None = None


@dataclass(frozen=True)
class ColumnPlan:
    """Which columns are feature candidates.

    Parameters
    ----------
    candidates : list[str]
        Feature candidates, in the order of the data.
    excluded : list[ExcludedColumn]
        Excluded columns, in the order of the data.
    sensors_not_in_data : list[str]
        Unavailable sensors that are not columns of the data.
    """

    candidates: list[str]
    excluded: list[ExcludedColumn]
    sensors_not_in_data: list[str]


def check_columns(frame: pl.DataFrame, config: Config, sensors: UnavailableSensors) -> None:
    """Check that the columns the config names exist and are usable.

    Parameters
    ----------
    frame : pl.DataFrame
        The input data.
    config : Config
        The config of the run.
    sensors : UnavailableSensors
        The unavailable sensors.

    Raises
    ------
    InputError
        When a named column is missing, or the target is an unavailable sensor.
    """
    for name in (config.target, *config.role_columns):
        if name not in frame.columns:
            raise InputError(f"column {name!r} is not in the data")
    if config.target in sensors.reasons:
        raise InputError(f"target {config.target!r} is listed as an unavailable sensor")
    for name in config.role_columns:
        if frame[name].null_count() > 0:
            raise InputError(f"column {name!r} has missing values")


def plan_columns(
    frame: pl.DataFrame, config: Config, sensors: UnavailableSensors, max_missing_rate: float
) -> ColumnPlan:
    """Decide the feature candidates.

    Parameters
    ----------
    frame : pl.DataFrame
        The training rows (``mloom.rows.select_rows``).
    config : Config
        The config of the run.
    sensors : UnavailableSensors
        The unavailable sensors.
    max_missing_rate : float
        A column whose missing rate is above this is excluded as
        ``high_missing``.

    Returns
    -------
    ColumnPlan
        The candidates and the excluded columns with their reasons.

    Raises
    ------
    InputError
        When no column is left as a candidate.
    """
    candidates: list[str] = []
    excluded: list[ExcludedColumn] = []
    for name in frame.columns:
        reason = name_reason(name, config, sensors) or _value_reason(frame, name, max_missing_rate)
        if reason is None:
            candidates.append(name)
        else:
            excluded.append(reason)
    if not candidates:
        raise InputError("no column is left as a feature candidate")
    not_in_data = [name for name in sensors.reasons if name not in frame.columns]
    return ColumnPlan(candidates=candidates, excluded=excluded, sensors_not_in_data=not_in_data)


def name_reason(name: str, config: Config, sensors: UnavailableSensors) -> ExcludedColumn | None:
    """Exclude a column for its name: target, role, user excluded, or unavailable sensor.

    Parameters
    ----------
    name : str
        Column name.
    config : Config
        The config of the run.
    sensors : UnavailableSensors
        The unavailable sensors.

    Returns
    -------
    ExcludedColumn | None
        The exclusion, or ``None`` when the name does not exclude the column.
    """
    if name == config.target:
        return ExcludedColumn(name, "target")
    if name in config.role_columns:
        return ExcludedColumn(name, "role")
    if name in config.exclude_columns:
        return ExcludedColumn(name, "user_excluded")
    if name in sensors.reasons:
        return ExcludedColumn(name, "unavailable_sensor", sensors.reasons[name])
    return None


def _value_reason(frame: pl.DataFrame, name: str, max_missing_rate: float) -> ExcludedColumn | None:
    dtype = frame.schema[name]
    if not (dtype.is_numeric() or dtype == pl.Boolean):
        return ExcludedColumn(name, "non_numeric", str(dtype))
    values = frame.select(as_float(name)).to_series()
    n_missing = values.null_count()
    if n_missing == frame.height:
        return ExcludedColumn(name, "all_null")
    if values.drop_nulls().n_unique() == 1:
        return ExcludedColumn(name, "constant")
    rate = n_missing / frame.height
    if rate > max_missing_rate:
        return ExcludedColumn(name, "high_missing", f"{rate:.3f}")
    return None

