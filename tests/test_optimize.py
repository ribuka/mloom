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
    """Each model and task saves a ``model.pkl`` whose pipeline predicts from the table."""
    target = "y" if task == "regression" else "c"
    result = optimize_model(frame, make_config(model=model, task=task, target=target))
    run_path = result.save(tmp_path)

    with (tmp_path / "model.pkl").open("rb") as file:
        bundle = pickle.load(file)
    predictions = bundle["pipeline"].predict(frame)
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


def test_dropped_rows_are_recorded(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """Rows whose target is missing, or with +-inf, are left out and counted in ``run.json``."""
    data = frame.with_columns(
        y=pl.when(pl.col("t") < 6).then(float("nan")).otherwise(pl.col("y")),
        s2=pl.when(pl.col("t") == 10).then(float("inf")).otherwise(pl.col("s2")),
    )
    run = optimize_model(data, make_config()).to_dict()["data"]
    assert run["n_rows_without_target"] == 6
    assert run["n_rows_with_infinite"] == 1
    assert run["infinite_by_column"] == {"s2": 1}
    assert run["n_rows_used"] == frame.height - 7
    assert run["n_unlabeled_rows_used"] == 6


@pytest.mark.parametrize("use_unlabeled", [True, False])
def test_unlabeled_rows_join_the_preprocessing(
    frame: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path, use_unlabeled: bool
) -> None:
    """With ``use_unlabeled_rows``, the median and the scaling come from labeled and unlabeled rows."""
    # Unlabeled rows far from the labeled ones move the statistics a lot.
    unlabeled = frame.head(30).with_columns(y=pl.lit(None, dtype=pl.Float64), s0=pl.col("s0") + 100.0)
    data = pl.concat([frame, unlabeled])
    settings = tmp_path / "mloom.toml"
    settings.write_text(f"[preprocessing]\nuse_unlabeled_rows = {str(use_unlabeled).lower()}\n", encoding="utf-8")
    result = optimize_model(data, make_config(model="pls", settings=settings))

    statistic_rows = data if use_unlabeled else frame
    preprocess = result.pipeline.named_steps["preprocess"].estimator
    s0 = result.columns.candidates.index("s0")
    assert preprocess.named_steps["scale"].mean_[s0] == pytest.approx(statistic_rows["s0"].mean())
    assert preprocess.named_steps["impute"].statistics_[s0] == pytest.approx(statistic_rows["s0"].median())
    assert result.data.n_unlabeled_rows_used == (30 if use_unlabeled else 0)
    assert all(fold["n_unlabeled"] == (30 if use_unlabeled else 0) for fold in result.cv["folds"])


@pytest.mark.parametrize(("rate", "excluded"), [(0.5, False), (0.2, True)])
def test_max_missing_rate_comes_from_settings(
    frame: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path, rate: float, excluded: bool
) -> None:
    """``max_missing_rate`` in the project settings decides whether a column with 25% missing is excluded."""
    data = frame.with_columns(s2=pl.when(pl.col("t") % 4 == 0).then(None).otherwise(pl.col("s2")))
    settings = tmp_path / "mloom.toml"
    settings.write_text(f"[columns]\nmax_missing_rate = {rate}\n", encoding="utf-8")
    run = optimize_model(data, make_config(settings=settings)).to_dict()
    high_missing = {"name": "s2", "reason": "high_missing", "detail": "labeled 0.250, all 0.250"}
    assert (high_missing in run["excluded_columns"]) is excluded
    assert run["settings"]["max_missing_rate"] == rate
    assert run["settings"]["sources"]["max_missing_rate"] == str(settings)


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


def test_train_filters_leave_rows_out_of_training_and_statistics(
    frame: pl.DataFrame, make_config: Callable[..., Config]
) -> None:
    """Rows outside ``train_filters`` are neither trained on nor in the statistics, and are counted."""
    # Rows outside the filters have values far from the others, which would move the statistics.
    late = frame.with_columns(s0=pl.when(pl.col("t") >= 90).then(pl.col("s0") + 1000.0).otherwise(pl.col("s0")))
    unlabeled = late.tail(40).with_columns(y=pl.lit(None, dtype=pl.Float64), s0=pl.col("s0") + 100.0)
    data = pl.concat([late, unlabeled])
    filters = ({"column": "lot", "op": "ne", "value": "L0"},)
    train_filters = ({"column": "t", "op": "lt", "value": 90},)
    result = optimize_model(data, make_config(model="pls", filters=filters, train_filters=train_filters))

    filtered = data.filter(pl.col("lot") != "L0")
    statistic_rows = filtered.filter(pl.col("t") < 90)
    run = result.to_dict()
    assert run["filters"] == list(filters)
    assert run["train_filters"] == list(train_filters)
    assert run["data"]["n_rows"] == data.height
    assert run["data"]["n_rows_filtered_out"] == data.height - filtered.height
    assert run["data"]["n_rows_train_filtered_out"] == filtered.height - statistic_rows.height
    assert run["data"]["n_rows_used"] == statistic_rows["y"].is_not_null().sum()
    assert run["data"]["n_unlabeled_rows_used"] == statistic_rows["y"].is_null().sum()
    assert run["data"]["n_rows_without_target"] == statistic_rows["y"].is_null().sum()
    preprocess = result.pipeline.named_steps["preprocess"].estimator
    s0 = result.columns.candidates.index("s0")
    assert preprocess.named_steps["impute"].statistics_[s0] == pytest.approx(statistic_rows["s0"].median())
    assert preprocess.named_steps["scale"].mean_[s0] == pytest.approx(statistic_rows["s0"].mean())


def test_columns_of_the_filters_stay_candidates(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """A column is not excluded only because a filter uses it."""
    result = optimize_model(frame, make_config(filters=({"column": "s0", "op": "gt", "value": -1},)))
    assert "s0" in result.columns.candidates


def test_role_column_may_miss_values_in_rows_the_filters_remove(
    frame: pl.DataFrame, make_config: Callable[..., Config]
) -> None:
    """Missing values of ``group_column`` are checked only in the rows left by the filters."""
    data = frame.with_columns(lot=pl.when(pl.col("t") == 0).then(None).otherwise(pl.col("lot")))
    config = make_config(cv="group", group_column="lot")
    with pytest.raises(InputError, match="column 'lot' has missing values"):
        optimize_model(data, config)
    filters = ({"column": "lot", "op": "is_not_null"},)
    assert optimize_model(data, make_config(cv="group", group_column="lot", filters=filters)).data.n_rows_filtered_out == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"filters": ({"column": "missing", "op": "is_null"},)}, r"filters\[0\]: column 'missing' is not in the data"),
        ({"filters": ({"column": "s0", "op": "gt", "value": "0"},)}, r"filters\[0\]: value '0' does not fit"),
        ({"train_filters": ({"column": "t", "op": "gt", "value": 1000},)}, "no row is left after applying train_filters"),
        ({"cv": "timeseries", "time_column": "d"}, "column 'd' is of type Duration"),
    ],
)
def test_invalid_filters_or_unhashable_columns_raise(
    frame: pl.DataFrame, make_config: Callable[..., Config], overrides: dict, message: str
) -> None:
    """Filters that do not fit the data, or a role column the rows cannot be hashed by, raise ``InputError``."""
    data = frame.with_columns(d=pl.duration(days=pl.col("t")))
    with pytest.raises(InputError, match=message):
        optimize_model(data, make_config(exclude_columns=("c", "d"), **overrides))
