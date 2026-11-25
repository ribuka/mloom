"""Tests of ``optimize_model`` and of the pipeline it saves."""

import json
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mloom import Config, InputError, optimize_model
from mloom.models import MODELS

MODEL_TASKS = [(model, task) for model, spec in MODELS.items() for task in spec.builders]


@pytest.mark.parametrize(("model", "task"), MODEL_TASKS)
def test_every_model_fits_saves_and_predicts(
    frame: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path, model: str, task: str
) -> None:
    """Each model and task saves a pipeline that predicts from the table."""
    target = "y" if task == "regression" else "c"
    result = optimize_model(frame, make_config(model=model, task=task, target=target))
    run_path = result.save(tmp_path)

    with (tmp_path / "model.pkl").open("rb") as file:
        pipeline = pickle.load(file)
    predictions = pipeline.predict(frame)
    assert len(predictions) == frame.height
    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert run["model"] == model
    assert run["task"] == task
    assert set(run["selected_columns"]) <= set(run["candidate_columns"])
    assert len(run["selected_columns"]) == run["best_params"]["k"]
    assert len(run["cv_scores"]["folds"]) == 3


@pytest.mark.parametrize(
    "overrides",
    [{"cv": "timeseries", "time_column": "t"}, {"cv": "group", "group_column": "lot"}],
)
def test_cv_methods_run(frame: pl.DataFrame, make_config: Callable[..., Config], overrides: dict) -> None:
    """The time series and group splits run end to end, and their columns are not features."""
    result = optimize_model(frame, make_config(**overrides))
    role = overrides.get("time_column") or overrides["group_column"]
    assert role not in result.columns.candidates
    assert result.cv["method"] == overrides["cv"]


def test_unavailable_sensor_is_never_selected(
    frame: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path
) -> None:
    """A sensor in ``unavailable-sensors.json`` is neither a candidate nor selected, and its reason is saved."""
    sensors = tmp_path / "unavailable-sensors.json"
    sensors.write_text('{"sensors": [{"name": "s0", "reason": "故障"}, {"name": "gone"}]}', encoding="utf-8")
    result = optimize_model(frame, make_config(unavailable_sensors=sensors))
    run = result.to_dict()
    assert "s0" not in run["candidate_columns"]
    assert "s0" not in run["selected_columns"]
    assert {"name": "s0", "reason": "unavailable_sensor", "detail": "故障"} in run["excluded_columns"]
    assert run["unavailable_sensors"]["not_in_data"] == ["gone"]


def test_parquet_path_and_dataframe_agree(
    frame: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path
) -> None:
    """A Parquet path gives the same result as the DataFrame, and its path is recorded."""
    path = tmp_path / "data.parquet"
    frame.write_parquet(path)
    from_path = optimize_model(path, make_config())
    from_frame = optimize_model(frame, make_config())
    assert from_path.data.path == str(path)
    assert from_path.best_params == from_frame.best_params


def test_same_seed_gives_same_result(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Two runs with the same data, config and seed select the same columns and parameters."""
    config = make_config(model="random_forest")
    first, second = optimize_model(frame, config), optimize_model(frame, config)
    assert first.selected_columns == second.selected_columns
    assert first.best_params == second.best_params
    assert first.cv_scores == second.cv_scores


def test_pipeline_ignores_extra_columns_and_order(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """The saved pipeline selects its columns by name from any table that has them."""
    result = optimize_model(frame, make_config())
    shuffled = frame.select(reversed(frame.columns)).with_columns(extra=pl.lit(1.0))
    np.testing.assert_allclose(result.pipeline.predict(shuffled), result.pipeline.predict(frame))


def test_rows_without_target_are_dropped(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Rows whose target is null or NaN are left out and counted."""
    data = frame.with_columns(y=pl.when(pl.col("t") < 6).then(float("nan")).otherwise(pl.col("y")))
    result = optimize_model(data, make_config())
    assert result.data.n_rows_used == frame.height - 6


@pytest.mark.parametrize(
    ("data_change", "overrides", "message"),
    [
        (None, {"target": "text"}, "needs a numeric target"),
        (pl.lit("a").alias("c"), {"task": "classification", "target": "c"}, "needs 2 or more classes"),
        (
            pl.Series("c", ["a", "b", "c"] * 40),
            {"task": "classification", "target": "c", "metric": "roc_auc"},
            "needs 2 classes",
        ),
    ],
)
def test_invalid_target_raises(
    frame: pl.DataFrame,
    make_config: Callable[..., Config],
    data_change: pl.Expr | pl.Series | None,
    overrides: dict,
    message: str,
) -> None:
    """A target that does not fit the task or the metric raises ``InputError``."""
    data = frame if data_change is None else frame.with_columns(data_change)
    with pytest.raises(InputError, match=message):
        optimize_model(data, make_config(**overrides))
