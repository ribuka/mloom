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
    plan = plan_cv(frame, frame[config.target].to_numpy(), config)
    assert plan.splitter == splitter
    assert len(plan.folds) == config.n_splits


def test_timeseries_tests_on_later_rows(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """With ``timeseries``, rows are sorted by time and each test fold follows its training rows."""
    config = make_config(cv="timeseries", time_column="t")
    rows = order_rows(frame, config)
    assert rows["t"].is_sorted()
    t = rows["t"].to_numpy()
    for train, test in plan_cv(rows, rows["y"].to_numpy(), config).folds:
        assert t[train].max() < t[test].min()


def test_group_keeps_groups_apart(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """With ``group``, no group is in both the training and the test rows of a fold."""
    config = make_config(cv="group", group_column="lot")
    lots = frame["lot"].to_numpy()
    for train, test in plan_cv(frame, frame["y"].to_numpy(), config).folds:
        assert not set(lots[train]) & set(lots[test])


def test_too_few_groups_raises(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Fewer groups than folds raise ``InputError``."""
    data = frame.with_columns(lot=pl.Series(np.where(np.arange(frame.height) % 2 == 0, "A", "B")))
    with pytest.raises(InputError, match="2 groups, fewer than n_splits=3"):
        plan_cv(data, data["y"].to_numpy(), make_config(cv="group", group_column="lot"))
