"""Shared fixtures: a small synthetic data set, a fast config, and a base made from them."""

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
import pytest

import mloom.settings
from mloom.base import Base, load_base
from mloom.config import Config
from mloom.optimize import optimize_model
from mloom.result import Result
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
def graded() -> pl.DataFrame:
    """Rows in time order with a number ``x1``, a String ``grade`` of 4 values and a Categorical ``shift`` of 2.

    ``y`` (numeric), ``c2`` (two classes) and ``c3`` (three classes) depend on ``x1``, ``grade`` and ``shift``;
    ``grade`` drives them most. Each value of ``grade`` has about 200 rows, enough for LightGBM to split on it.
    """
    rng = np.random.default_rng(5)
    n_rows = 800
    x1 = rng.normal(size=n_rows)
    grade = rng.choice(["A", "B", "C", "D"], size=n_rows)
    shift = rng.choice(["day", "night"], size=n_rows)
    effect = np.select([grade == value for value in "ABC"], [-3.0, -1.0, 1.0], 3.0) + np.where(shift == "day", 0.5, -0.5)
    y = x1 + effect + rng.normal(scale=0.3, size=n_rows)
    return pl.DataFrame(
        {
            "t": np.arange(n_rows),
            "x1": x1,
            "grade": grade,
            "shift": pl.Series(shift, dtype=pl.Categorical),
            "y": y,
            "c2": np.where(y > 0, "pos", "neg"),
            "c3": np.select([y < -1.5, y < 1.5], ["low", "mid"], "high"),
        }
    )


@pytest.fixture
def data_path(frame: pl.DataFrame, tmp_path: Path) -> Path:
    """Write the shared frame as Parquet and return its path."""
    path = tmp_path / "data.parquet"
    frame.write_parquet(path)
    return path


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


@pytest.fixture
def make_base(
    frame: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path
) -> Callable[..., tuple[Result, Base]]:
    """Optimize on the shared frame (or ``data``), save to ``<tmp>/base`` and load it as a base.

    The other keyword arguments go to the config.
    """

    def make(data: pl.DataFrame | None = None, **overrides: object) -> tuple[Result, Base]:
        result = optimize_model(frame if data is None else data, make_config(**overrides))
        return result, load_base(result.save(tmp_path / "base"))

    return make
