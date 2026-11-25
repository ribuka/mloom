"""Shared fixtures: a small synthetic data set and a fast config."""

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import optuna
import polars as pl
import pytest

from mloom.config import Config

N_ROWS = 120


@pytest.fixture(autouse=True)
def _quiet_optuna() -> Iterator[None]:
    level = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    optuna.logging.set_verbosity(level)


@pytest.fixture
def frame() -> pl.DataFrame:
    """Rows in reverse time order, 12 lots, 6 sensors and odd columns.

    ``y`` (numeric) and ``c`` (two classes) depend on ``s0`` and ``s1``.
    """
    rng = np.random.default_rng(0)
    sensors = rng.normal(size=(N_ROWS, 6))
    y = 2.0 * sensors[:, 0] + sensors[:, 1] + rng.normal(scale=0.1, size=N_ROWS)
    return pl.DataFrame(
        {
            "t": np.arange(N_ROWS)[::-1],
            "lot": [f"L{i % 12}" for i in range(N_ROWS)],
            **{f"s{i}": sensors[:, i] for i in range(6)},
            "const": np.ones(N_ROWS),
            "text": ["a"] * N_ROWS,
            "y": y,
            "c": np.where(y > 0, "pos", "neg"),
        }
    )


@pytest.fixture
def make_config() -> Callable[..., Config]:
    """Build a config that runs fast; keyword arguments override the defaults."""

    def make(**overrides: Any) -> Config:  # noqa: ANN401 - any Config field
        values: dict[str, Any] = {
            "target": "y",
            "task": "regression",
            "model": "ridge",
            "cv": "kfold",
            "n_splits": 3,
            "n_trials": 3,
            "exclude_columns": ("y", "c"),
            "unavailable_sensors": None,
        }
        values.update(overrides)
        return Config(**values)

    return make
