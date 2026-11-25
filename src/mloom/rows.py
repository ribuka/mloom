"""Choosing the training rows: rows with a target and without infinite values.

See "行の除外" in ``docs/spec.md``. A row is dropped when its target is
missing, or when the target or a column that can become a feature holds
+inf or -inf.
"""

from dataclasses import dataclass

import polars as pl

from mloom.columns import name_reason
from mloom.config import Config
from mloom.errors import InputError
from mloom.sensors import UnavailableSensors


@dataclass(frozen=True)
class TrainingRows:
    """The training rows and what was dropped.

    Parameters
    ----------
    frame : pl.DataFrame
        The training rows, in the order of the input.
    n_without_target : int
        Rows dropped because the target is missing (null or NaN).
    n_with_infinite : int
        Rows dropped, among those with a target, because of +inf or -inf.
    infinite_by_column : dict[str, int]
        For each column holding +inf or -inf, the number of such rows among
        those with a target. A row can count for several columns.
    """

    frame: pl.DataFrame
    n_without_target: int
    n_with_infinite: int
    infinite_by_column: dict[str, int]


def select_rows(frame: pl.DataFrame, config: Config, sensors: UnavailableSensors) -> TrainingRows:
    """Drop the rows that cannot be used for training.

    Parameters
    ----------
    frame : pl.DataFrame
        The input data.
    config : Config
        The config of the run.
    sensors : UnavailableSensors
        The unavailable sensors; their columns are not checked for infinite
        values.

    Returns
    -------
    TrainingRows
        The training rows and the counts of dropped rows.

    Raises
    ------
    InputError
        When no row is left.
    """
    target = pl.col(config.target)
    if frame.schema[config.target].is_float():
        target = target.fill_nan(None)
    with_target = frame.filter(target.is_not_null())
    if with_target.height == 0:
        raise InputError(f"target {config.target!r} has no values")

    checked = [
        name
        for name, dtype in frame.schema.items()
        if dtype.is_float() and (name == config.target or name_reason(name, config, sensors) is None)
    ]
    infinite = {name: pl.col(name).is_infinite().fill_null(value=False) for name in checked}
    counts = with_target.select(expr.sum() for expr in infinite.values()).row(0) if checked else ()
    by_column = {name: int(count) for name, count in zip(checked, counts, strict=True) if count}
    rows = with_target.filter(~pl.any_horizontal(infinite.values())) if checked else with_target
    if rows.height == 0:
        raise InputError("no row is left after dropping the rows with infinite values")
    return TrainingRows(
        frame=rows,
        n_without_target=frame.height - with_target.height,
        n_with_infinite=with_target.height - rows.height,
        infinite_by_column=by_column,
    )
