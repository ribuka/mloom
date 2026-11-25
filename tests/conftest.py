"""Shared fixtures: a small synthetic data set and a fast config."""

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
import pytest

import mloom.settings
from mloom.config import Config
from mloom.settings import SETTINGS_FILE

N_ROWS = 120


@pytest.fixture(autouse=True)
def mloom_config(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Read the defaults from a copy of config/mloom.toml, never from config/mloom.local.toml.

    A developer's local defaults must not change the tests. Return the directory
    of the copy, where a test may write a local defaults file.
    """
    directory = tmp_path_factory.mktemp("mloom-config")
    shutil.copy(Path(__file__).parents[1] / "config" / SETTINGS_FILE, directory / SETTINGS_FILE)
    monkeypatch.setattr(mloom.settings, "mloom_config_dir", lambda: directory)
    return directory


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
            "settings": None,
        }
        values.update(overrides)
        return Config(**values)

    return make
