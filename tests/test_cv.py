"""Tests of the CV split the config names."""

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from mloom.config import Config
from mloom.cv import order_rows, plan_cv
from mloom.errors import InputError


@pytest.mark.parametrize(
    ("overrides", "splitter"),
    [
        ({}, "KFold"),
        ({"task": "classification", "target": "c"}, "StratifiedKFold"),
        ({"cv": "timeseries", "time_column": "t"}, "TimeSeriesSplit"),
        ({"cv": "group", "group_column": "lot"}, "GroupKFold"),
        ({"cv": "group", "group_column": "lot", "task": "classification", "target": "c"}, "StratifiedGroupKFold"),
    ],
)
def test_splitter_follows_config(
    frame: pl.DataFrame, make_config: Callable[..., Config], overrides: dict, splitter: str
) -> None:
    """The splitter is the one the config names, with stratification for classification."""
    config = make_config(**overrides)
    plan = plan_cv(frame, frame[config.target].to_numpy(), frame.clear(), config)
    assert plan.splitter == splitter
    assert len(plan.folds) == config.n_splits


def _split_labels(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Treat every third row as unlabeled."""
    unlabeled = pl.col("t") % 3 == 0
    return frame.filter(~unlabeled), frame.filter(unlabeled)


def test_timeseries_keeps_later_rows_out(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """With ``timeseries``, each test fold follows its training rows, and later unlabeled rows are left out."""
    config = make_config(cv="timeseries", time_column="t")
    labeled, unlabeled = _split_labels(frame)
    rows = order_rows(labeled, config)
    assert rows["t"].is_sorted()
    t, t_unlabeled = rows["t"].to_numpy(), unlabeled["t"].to_numpy()
    plan = plan_cv(rows, rows["y"].to_numpy(), unlabeled, config)
    for (train, test), kept in zip(plan.folds, plan.unlabeled_folds, strict=True):
        assert t[train].max() < t[test].min()
        expected = np.flatnonzero(t_unlabeled <= t[train].max())
        np.testing.assert_array_equal(kept, expected)


def test_group_keeps_groups_apart(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """With ``group``, no group is on both sides, and unlabeled rows of the test groups are left out."""
    config = make_config(cv="group", group_column="lot")
    labeled, unlabeled = _split_labels(frame)
    lots, lots_unlabeled = labeled["lot"].to_numpy(), unlabeled["lot"].to_numpy()
    plan = plan_cv(labeled, labeled["y"].to_numpy(), unlabeled, config)
    for (train, test), kept in zip(plan.folds, plan.unlabeled_folds, strict=True):
        assert not set(lots[train]) & set(lots[test])
        np.testing.assert_array_equal(kept, np.flatnonzero(~np.isin(lots_unlabeled, lots[test])))


def test_kfold_uses_every_unlabeled_row(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """With ``kfold``, no unlabeled row is on a test side, so every fold uses all of them."""
    labeled, unlabeled = _split_labels(frame)
    plan = plan_cv(labeled, labeled["y"].to_numpy(), unlabeled, make_config())
    for kept in plan.unlabeled_folds:
        np.testing.assert_array_equal(kept, np.arange(unlabeled.height))


def test_too_few_groups_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Fewer groups than folds raise ``InputError``."""
    data = frame.with_columns(lot=pl.Series(np.where(np.arange(frame.height) % 2 == 0, "A", "B")))
    with pytest.raises(InputError, match="2 groups, fewer than n_splits=3"):
        plan_cv(data, data["y"].to_numpy(), data.clear(), make_config(cv="group", group_column="lot"))
