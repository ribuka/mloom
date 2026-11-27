"""Excluding columns before feature selection, recording why each was excluded.

The columns left are the feature candidates, numeric or categorical.

The reasons and their order are those of "列の除外" in ``docs/spec.md``;
a column gets the first reason that applies.
"""

from dataclasses import dataclass

import polars as pl

from mloom.config import Config
from mloom.dtypes import BOOLEAN, NUMBER, STRING, value_kind
from mloom.errors import InputError
from mloom.pipeline import as_float
from mloom.sensors import UnavailableSensors

# The kinds of feature candidates (``ColumnPlan.numeric`` and ``ColumnPlan.categorical``).
NUMERIC = "numeric"
CATEGORICAL = "categorical"


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
    numeric : list[str]
        Numeric and boolean feature candidates, in the order of the data.
    categorical : list[str]
        String, Categorical and Enum feature candidates, in the order of the
        data; empty unless the model ``handles_categorical``.
    excluded : list[ExcludedColumn]
        Excluded columns, in the order of the data.
    sensor_columns : list[list[str]]
        For each entry of ``unavailable-sensors.json`` (in the order of the
        file), the columns of the data it matches, in the order of the data;
        also those excluded for another reason, such as ``role``.
    """

    numeric: list[str]
    categorical: list[str]
    excluded: list[ExcludedColumn]
    sensor_columns: list[list[str]]

    @property
    def candidates(self) -> list[str]:
        """All the feature candidates: the numeric ones, then the categorical ones.

        Returns
        -------
        list[str]
            The candidates, in the order of the columns of the feature
            matrix, of the input of the estimator and of the record.
        """
        return [*self.numeric, *self.categorical]


def check_columns(frame: pl.DataFrame, config: Config, sensors: UnavailableSensors) -> None:
    """Check that the columns the config names exist and the target may be used.

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
    entry = sensors.first_match(config.target)
    if entry is not None:
        raise InputError(
            f"target {config.target!r} is an unavailable sensor: it matches {entry.name!r} (match {entry.match!r})"
        )


def check_role_values(frame: pl.DataFrame, config: Config) -> None:
    """Check that ``time_column`` and ``group_column`` have no missing value.

    Parameters
    ----------
    frame : pl.DataFrame
        The rows left after the row filters; the rows the filters removed
        are not used, so they may have missing values.
    config : Config
        The config of the run.

    Raises
    ------
    InputError
        When a role column has a missing value.
    """
    for name in config.role_columns:
        if frame[name].null_count() > 0:
            raise InputError(f"column {name!r} has missing values")


def plan_columns(
    frame: pl.DataFrame,
    unlabeled: pl.DataFrame,
    config: Config,
    sensors: UnavailableSensors,
    max_missing_rate: float,
) -> ColumnPlan:
    """Decide the feature candidates.

    Parameters
    ----------
    frame : pl.DataFrame
        The labeled rows (``mloom.rows.TrainingRows.labeled``).
    unlabeled : pl.DataFrame
        The unlabeled rows used for statistics (may be empty).
    config : Config
        The config of the run.
    sensors : UnavailableSensors
        The unavailable sensors.
    max_missing_rate : float
        A column whose missing rate is above this, in the labeled rows or in
        the labeled and unlabeled rows together, is excluded as
        ``high_missing``.

    Returns
    -------
    ColumnPlan
        The candidates and the excluded columns with their reasons. The
        candidates may be empty (``require_candidates``). String,
        Categorical and Enum columns are categorical candidates when the
        model ``handles_categorical``, and ``non_numeric`` otherwise.
    """
    candidates: dict[str, list[str]] = {NUMERIC: [], CATEGORICAL: []}
    excluded: list[ExcludedColumn] = []
    for name in frame.columns:
        kind = _feature_kind(frame.schema[name], config.model_spec.handles_categorical)
        reason = name_reason(name, config, sensors) or _value_reason(frame, unlabeled, name, kind, max_missing_rate)
        if reason is not None:
            excluded.append(reason)
        elif kind is not None:  # always: a column without a kind is non_numeric
            candidates[kind].append(name)
    sensor_columns = [[name for name in frame.columns if entry.matches(name)] for entry in sensors.entries]
    return ColumnPlan(
        numeric=candidates[NUMERIC],
        categorical=candidates[CATEGORICAL],
        excluded=excluded,
        sensor_columns=sensor_columns,
    )


def require_candidates(plan: ColumnPlan) -> None:
    """Check that a column is left as a feature candidate.

    Parameters
    ----------
    plan : ColumnPlan
        The plan from ``plan_columns``.

    Raises
    ------
    InputError
        When no column is left as a candidate.
    """
    if not plan.candidates:
        raise InputError("no column is left as a feature candidate")


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
    entry = sensors.first_match(name)
    if entry is not None:
        return ExcludedColumn(name, "unavailable_sensor", entry.reason)
    return None


def _feature_kind(dtype: pl.DataType, handles_categorical: bool) -> str | None:
    # NUMERIC or CATEGORICAL for a column of the type, or None when it cannot be a feature (non_numeric).
    kind = value_kind(dtype)
    if kind in (NUMBER, BOOLEAN):
        return NUMERIC
    if kind == STRING and handles_categorical:
        return CATEGORICAL
    return None


def _value_reason(
    frame: pl.DataFrame, unlabeled: pl.DataFrame, name: str, kind: str | None, max_missing_rate: float
) -> ExcludedColumn | None:
    # all_null and constant look at the labeled rows: the model learns from them only.
    if kind is None:
        return ExcludedColumn(name, "non_numeric", str(frame.schema[name]))
    feature = _feature(name) if kind == NUMERIC else pl.col(name).cast(pl.String)
    values = frame.select(feature).to_series()
    n_missing = values.null_count()
    if n_missing == frame.height:
        return ExcludedColumn(name, "all_null")
    if values.drop_nulls().n_unique() == 1:
        return ExcludedColumn(name, "constant")
    labeled_rate = n_missing / frame.height
    n_missing_all = n_missing + unlabeled.select(feature.null_count()).item()
    all_rate = n_missing_all / (frame.height + unlabeled.height)
    if max(labeled_rate, all_rate) > max_missing_rate:
        return ExcludedColumn(name, "high_missing", f"labeled {labeled_rate:.3f}, all {all_rate:.3f}")
    return None


def _feature(name: str) -> pl.Expr:
    # +inf and -inf count as missing, as the pipeline turns them into NaN.
    value = as_float(name)
    return pl.when(value.is_infinite()).then(None).otherwise(value)
