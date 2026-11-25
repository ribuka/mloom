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
        Train and test indices of the labeled rows of each fold.
    unlabeled_folds : list[np.ndarray]
        For each fold, indices of the unlabeled rows that are on the training
        side and may join the statistics of the features.
    """

    splitter: str
    folds: list[Split]
    unlabeled_folds: list[np.ndarray]

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
            "folds": [
                {"n_train": len(train), "n_test": len(test), "n_unlabeled": len(unlabeled)}
                for (train, test), unlabeled in zip(self.folds, self.unlabeled_folds, strict=True)
            ],
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


def plan_cv(frame: pl.DataFrame, y: np.ndarray, unlabeled: pl.DataFrame, config: Config) -> CVPlan:
    """Split the rows into folds.

    Parameters
    ----------
    frame : pl.DataFrame
        The labeled rows, ordered by ``order_rows``.
    y : np.ndarray
        Target of each row.
    unlabeled : pl.DataFrame
        The unlabeled rows used for statistics (may be empty).
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
    return CVPlan(
        splitter=type(splitter).__name__,
        folds=folds,
        unlabeled_folds=[_training_side(frame, unlabeled, train, test, config) for train, test in folds],
    )


def _training_side(
    frame: pl.DataFrame, unlabeled: pl.DataFrame, train: np.ndarray, test: np.ndarray, config: Config
) -> np.ndarray:
    # Unlabeled rows that would leak the test side: later times, or the groups under test.
    indices = np.arange(unlabeled.height)
    if unlabeled.height == 0:
        return indices
    if config.cv == "timeseries" and config.time_column is not None:
        last_train_time = frame[config.time_column].to_numpy()[train].max()
        return indices[unlabeled[config.time_column].to_numpy() <= last_train_time]
    if config.cv == "group" and config.group_column is not None:
        test_groups = frame[config.group_column].to_numpy()[test]
        return indices[~np.isin(unlabeled[config.group_column].to_numpy(), test_groups)]
    return indices


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
