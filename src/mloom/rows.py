"""Choosing the rows: labeled rows for training, unlabeled rows for statistics.

See "行の除外" in ``docs/spec.md``. Rows without a target are "unlabeled";
they are kept only when ``use_unlabeled`` is set, for the missing rates and
the imputation and scaling statistics. A labeled or kept unlabeled row is
dropped when the target or a column that can become a feature holds +inf
or -inf.
"""

from dataclasses import dataclass

import polars as pl

from mloom.columns import name_reason
from mloom.config import Config
from mloom.errors import InputError
from mloom.sensors import UnavailableSensors


@dataclass(frozen=True)
class TrainingRows:
    """The rows of a run and what was dropped.

    Parameters
    ----------
    labeled : pl.DataFrame
        Rows with a target, used for training and evaluation.
    unlabeled : pl.DataFrame
        Rows without a target, used only for statistics of the features;
        empty when they are not used.
    n_without_target : int
        Rows of the input whose target is missing (null or NaN).
    n_with_infinite : int
        Rows dropped, among the labeled and the kept unlabeled rows, because
        of +inf or -inf.
    infinite_by_column : dict[str, int]
        For each column holding +inf or -inf, the number of such rows among
        the labeled and the kept unlabeled rows. A row can count for several
        columns.
    """

    labeled: pl.DataFrame
    unlabeled: pl.DataFrame
    n_without_target: int
    n_with_infinite: int
    infinite_by_column: dict[str, int]


def select_rows(
    frame: pl.DataFrame, config: Config, sensors: UnavailableSensors, *, use_unlabeled: bool
) -> TrainingRows:
    """Split the rows into labeled and unlabeled ones and drop infinite values.

    Parameters
    ----------
    frame : pl.DataFrame
        The input data.
    config : Config
        The config of the run.
    sensors : UnavailableSensors
        The unavailable sensors; their columns are not checked for infinite
        values.
    use_unlabeled : bool
        Whether to keep the rows without a target.

    Returns
    -------
    TrainingRows
        The rows and the counts of dropped rows.

    Raises
    ------
    InputError
        When no labeled row is left.
    """
    target = pl.col(config.target)
    if frame.schema[config.target].is_float():
        target = target.fill_nan(None)
    has_target = frame.select(target.is_not_null()).to_series()
    if not has_target.any():
        raise InputError(f"target {config.target!r} has no values")
    kept = frame.filter(has_target | use_unlabeled)
    kept_has_target = has_target.filter(has_target | use_unlabeled)

    checked = [
        name
        for name, dtype in frame.schema.items()
        if dtype.is_float() and (name == config.target or name_reason(name, config, sensors) is None)
    ]
    infinite = [pl.col(name).is_infinite().fill_null(value=False) for name in checked]
    counts = kept.select(infinite).sum().row(0) if checked else ()
    finite = kept.select(~pl.any_horizontal(infinite)).to_series() if checked else pl.repeat(True, kept.height, eager=True)
    labeled = kept.filter(finite & kept_has_target)
    if labeled.height == 0:
        raise InputError("no row with a target is left after dropping the rows with infinite values")
    return TrainingRows(
        labeled=labeled,
        unlabeled=kept.filter(finite & ~kept_has_target),
        n_without_target=frame.height - int(has_target.sum()),
        n_with_infinite=kept.height - int(finite.sum()),
        infinite_by_column={name: int(count) for name, count in zip(checked, counts, strict=True) if count},
    )
