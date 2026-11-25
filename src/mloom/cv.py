"""The CV split the user chose (see "CV" in ``docs/spec.md``).

mloom never infers the method from the data; it only builds the splitter
the config names and checks that the data can be split that way.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from mloom.config import Config
from mloom.errors import InputError

Split = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CVPlan:
    """The folds of a run.

    Parameters
    ----------
    splitter : str
        Class name of the scikit-learn splitter.
    folds : list[Split]
        Train and test row indices of each fold.
    """

    splitter: str
    folds: list[Split]

    def describe(self, config: Config) -> dict[str, Any]:
        """Describe the split for ``run.json``.

        Parameters
        ----------
        config : Config
            The config of the run.

        Returns
        -------
        dict[str, Any]
            Method, splitter, columns, and the size of each fold.
        """
        return {
            "method": config.cv,
            "splitter": self.splitter,
            "n_splits": config.n_splits,
            "time_column": config.time_column,
            "group_column": config.group_column,
            "folds": [{"n_train": len(train), "n_test": len(test)} for train, test in self.folds],
        }


def order_rows(frame: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Put the rows in the order the split needs.

    Parameters
    ----------
    frame : pl.DataFrame
        The training rows.
    config : Config
        The config of the run.

    Returns
    -------
    pl.DataFrame
        Rows sorted by ``time_column`` for ``timeseries``; otherwise as given.
    """
    if config.cv == "timeseries" and config.time_column is not None:
        return frame.sort(config.time_column, maintain_order=True)
    return frame


def plan_cv(frame: pl.DataFrame, y: np.ndarray, config: Config) -> CVPlan:
    """Split the rows into folds.

    Parameters
    ----------
    frame : pl.DataFrame
        The training rows, ordered by ``order_rows``.
    y : np.ndarray
        Target of each row.
    config : Config
        The config of the run.

    Returns
    -------
    CVPlan
        The folds.

    Raises
    ------
    InputError
        When the data cannot be split into ``n_splits`` folds.
    """
    groups = None
    if config.cv == "group" and config.group_column is not None:
        groups = frame[config.group_column].to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < config.n_splits:
            raise InputError(f"group_column has {n_groups} groups, fewer than n_splits={config.n_splits}")
    if frame.height <= config.n_splits:
        raise InputError(f"data has {frame.height} rows, too few for n_splits={config.n_splits}")
    splitter = _splitter(config)
    folds = [(train, test) for train, test in splitter.split(np.zeros(frame.height), y, groups)]
    return CVPlan(splitter=type(splitter).__name__, folds=folds)


def _splitter(config: Config) -> Any:  # noqa: ANN401 - any scikit-learn splitter
    classification = config.task == "classification"
    if config.cv == "timeseries":
        return TimeSeriesSplit(n_splits=config.n_splits)
    if config.cv == "group":
        if classification:
            return StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        return GroupKFold(n_splits=config.n_splits)
    if classification:
        return StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    return KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
