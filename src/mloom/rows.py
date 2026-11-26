"""Choosing the rows: labeled rows for training, unlabeled rows for statistics.

See "行の除外" in ``docs/spec.md``. Rows without a target are "unlabeled";
they are kept only when ``use_unlabeled`` is set, for the missing rates and
the statistics of the preprocessing. A labeled row is dropped when its target
is +inf or -inf. The +inf and -inf of the features are counted but kept: the
pipeline treats them as missing.
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
    n_with_infinite_target : int
        Rows dropped because their target is +inf or -inf.
    infinite_by_column : dict[str, int]
        For each column holding +inf or -inf (the target, or a column that
        can become a feature), the number of such rows among the labeled and
        the kept unlabeled rows, before the rows with an infinite target are
        dropped. A row can count for several columns.
    """

    labeled: pl.DataFrame
    unlabeled: pl.DataFrame
    n_without_target: int
    n_with_infinite_target: int
    infinite_by_column: dict[str, int]


def select_rows(
    frame: pl.DataFrame, config: Config, sensors: UnavailableSensors, *, use_unlabeled: bool
) -> TrainingRows:
    """Split the rows into labeled and unlabeled ones and drop the infinite targets.

    Parameters
    ----------
    frame : pl.DataFrame
        The input data.
    config : Config
        The config of the run.
    sensors : UnavailableSensors
        The unavailable sensors; their infinite values are not counted.
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
    counts = kept.select([pl.col(name).is_infinite().sum() for name in checked]).row(0) if checked else ()
    infinite_target = (
        kept.select(pl.col(config.target).is_infinite().fill_null(value=False)).to_series()
        if frame.schema[config.target].is_float()
        else pl.repeat(False, kept.height, eager=True)
    )
    labeled = kept.filter(kept_has_target & ~infinite_target)
    if labeled.height == 0:
        raise InputError("no row with a target is left after dropping the rows whose target is +inf or -inf")
    return TrainingRows(
        labeled=labeled,
        unlabeled=kept.filter(~kept_has_target),
        n_without_target=frame.height - int(has_target.sum()),
        n_with_infinite_target=int(infinite_target.sum()),
        infinite_by_column={name: int(count) for name, count in zip(checked, counts, strict=True) if count},
    )
