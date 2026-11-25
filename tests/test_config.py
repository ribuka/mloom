"""Tests of the checks ``Config`` makes without the data."""

from collections.abc import Callable

import pytest

from mloom.config import Config
from mloom.errors import InputError


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task": "ranking"}, "task must be one of"),
        ({"cv": "random"}, "cv must be one of"),
        ({"model": "svm"}, "model must be one of"),
        ({"model": "pls", "task": "classification"}, "does not support task"),
        ({"metric": "log_loss"}, "metric must be one of"),
        ({"cv": "timeseries"}, "needs time_column"),
        ({"cv": "group"}, "needs group_column"),
        ({"n_splits": 1}, "n_splits must be 2 or more"),
        ({"n_trials": 0}, "n_trials must be 1 or more"),
    ],
)
def test_invalid_config_raises(make_config: Callable[..., Config], overrides: dict, message: str) -> None:
    """Invalid values raise ``InputError`` before any data is read."""
    with pytest.raises(InputError, match=message):
        make_config(**overrides)


@pytest.mark.parametrize(("task", "metric"), [("regression", "rmse"), ("classification", "log_loss")])
def test_default_metric_follows_task(make_config: Callable[..., Config], task: str, metric: str) -> None:
    """Without ``metric``, each task gets its default metric."""
    assert make_config(task=task).metric_name == metric
