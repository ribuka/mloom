"""Tests of ``optimize_model`` and of the pipeline it saves."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import mloom.optimize
import mloom.training
from mloom import Config, InputError, optimize_model, read_run
from mloom.cv import CVPlan, plan_cv
from mloom.models import MODELS
from mloom.prediction import load_model
from mloom.selection import rank_columns

MODEL_TASKS = [(model, task) for model, spec in MODELS.items() for task in spec.builders]


@pytest.fixture
def similar() -> pl.DataFrame:
    """Independent ``a``, ``b`` and noise columns, and ``a2`` almost equal to ``a`` (|ρ| > 0.99).

    ``y`` (numeric) and ``c`` (two classes) depend on ``a`` and ``b``.
    """
    rng = np.random.default_rng(3)
    n_rows = 120
    a, b = rng.normal(size=(2, n_rows))
    noise = rng.normal(size=(n_rows, 3))
    y = 2.0 * a + b + rng.normal(scale=0.1, size=n_rows)
    return pl.DataFrame(
        {
            "a": a,
            "a2": a + rng.normal(scale=0.01, size=n_rows),
            "b": b,
            **{f"n{i}": noise[:, i] for i in range(3)},
            "y": y,
            "c": np.where(y > 0, "pos", "neg"),
        }
    )


@pytest.mark.parametrize(("model", "task"), MODEL_TASKS)
def test_every_model_fits_saves_and_predicts(
    similar: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path, model: str, task: str
) -> None:
    """Each model and task saves a ``model.pkl`` whose pipeline predicts from the table with similar columns."""
    target = "y" if task == "regression" else "c"
    result = optimize_model(similar, make_config(model=model, task=task, target=target))
    model_path = result.save(tmp_path)

    predictions = load_model(model_path)["pipeline"].predict(similar)
    assert len(predictions) == similar.height
    run = read_run(model_path)
    assert run["model"] == model
    assert run["task"] == task
    assert set(run["selected_columns"]) <= set(run["candidate_columns"])
    assert len(run["selected_columns"]) == run["best_params"]["k"]
    assert run["best_params"].get("n_components", 1) <= run["selection"]["n_clusters"]
    assert len(run["cv_scores"]["folds"]) == 3


def test_similar_columns_are_not_selected_together(similar: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """``a`` and ``a2`` form a cluster recorded with its representative; at most one of them is selected."""
    result = optimize_model(similar, make_config(n_trials=10))
    run = result.to_dict()
    assert not {"a", "a2"} <= set(run["selected_columns"])
    selection = run["selection"]
    assert selection["max_correlation"] == 0.9
    [cluster] = selection["clusters"]
    assert cluster["columns"] == ["a", "a2"]
    assert cluster["representative"] in {"a", "a2"}
    assert cluster["selected"] == (cluster["representative"] in run["selected_columns"])
    assert selection["n_clusters"] == len(run["candidate_columns"]) - 1
    assert run["best_params"]["k"] <= selection["n_clusters"]


def test_max_correlation_of_one_selects_as_before(
    similar: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path
) -> None:
    """``max_correlation = 1`` clusters nothing and selects what mloom selected before the clusters existed."""
    settings = tmp_path / "mloom.toml"
    settings.write_text("[selection]\nmax_correlation = 1\n", encoding="utf-8")
    run = optimize_model(similar, make_config(n_trials=10, settings=settings)).to_dict()
    assert run["selection"]["clusters"] == []
    assert run["selection"]["n_clusters"] == len(run["candidate_columns"])
    # Recorded with the same data, config and seed on main before the clusters were added.
    assert run["selected_columns"] == ["a", "a2", "b", "n2"]
    assert run["best_params"] == {"alpha": pytest.approx(0.00013852519879556092, rel=1e-12), "k": 4}


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


@pytest.mark.parametrize("model", list(MODELS))
def test_same_seed_gives_same_result(similar: pl.DataFrame, make_config: Callable[..., Config], model: str) -> None:
    """Two runs with the same data, config and seed give the same clusters, columns and parameters."""
    config = make_config(model=model)
    first, second = optimize_model(similar, config), optimize_model(similar, config)
    assert first.selection == second.selection
    assert first.selected_columns == second.selected_columns
    assert first.best_params == second.best_params
    assert first.cv_scores == second.cv_scores


def test_pipeline_ignores_extra_columns_and_order(frame: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """The saved pipeline selects its columns by name from any table that has them."""
    result = optimize_model(frame, make_config())
    shuffled = frame.select(reversed(frame.columns)).with_columns(extra=pl.lit(1.0))
    np.testing.assert_allclose(result.pipeline.predict(shuffled), result.pipeline.predict(frame))


@pytest.mark.parametrize("model", ["ridge", "lightgbm"])
def test_only_rows_with_an_infinite_target_are_dropped(
    frame: pl.DataFrame, make_config: Callable[..., Config], model: str
) -> None:
    """Rows whose target is missing or +-inf are left out; rows with +-inf in a feature are trained on."""
    data = frame.with_columns(
        y=pl.when(pl.col("t") < 6).then(float("nan")).when(pl.col("t") == 7).then(float("-inf")).otherwise(pl.col("y")),
        # +-inf in a labeled row (t = 10, 11) and in an unlabeled row (t = 2).
        s2=pl.when(pl.col("t").is_in([2, 10])).then(float("inf")).otherwise(pl.col("s2")),
        s3=pl.when(pl.col("t") == 11).then(float("-inf")).otherwise(pl.col("s3")),
    )
    result = optimize_model(data, make_config(model=model))
    run = result.to_dict()["data"]
    assert run["n_rows_without_target"] == 6
    assert run["n_rows_with_infinite_target"] == 1
    assert run["infinite_by_column"] == {"s2": 2, "s3": 1, "y": 1}
    assert run["n_rows_used"] == frame.height - 7
    assert run["n_unlabeled_rows_used"] == 6
    assert np.isfinite(result.pipeline.predict(data.filter(pl.col("t").is_in([10, 11])))).all()


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
    assert run["settings"]["columns"]["max_missing_rate"] == rate
    assert run["settings"]["sources"]["columns"]["max_missing_rate"] == str(settings)


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


def _settings(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "mloom.toml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def with_missing(frame: pl.DataFrame) -> pl.DataFrame:
    """Add missing values and +-inf to ``s0`` and ``s1`` of the shared frame (none to the other sensors)."""
    return frame.with_columns(
        s0=pl.when(pl.col("t") % 7 == 0).then(None).when(pl.col("t") % 11 == 0).then(float("inf")).otherwise(pl.col("s0")),
        s1=pl.when(pl.col("t") % 5 == 0).then(float("nan")).otherwise(pl.col("s1")),
    )


@pytest.mark.parametrize(
    ("method", "model", "resolved"),
    [
        ("auto", "ridge", "median"),
        ("auto", "lightgbm", "none"),
        ("auto", "random_forest", "none"),
        ("none", "lightgbm", "none"),
        ("median", "ridge", "median"),
        ("mean", "lightgbm", "mean"),
        ("most_frequent", "ridge", "most_frequent"),
        ("constant", "lightgbm", "constant"),
        ("knn", "ridge", "knn"),
        ("iterative", "lightgbm", "iterative"),
    ],
)
def test_each_imputation_fits_and_predicts(
    with_missing: pl.DataFrame,
    make_config: Callable[..., Config],
    tmp_path: Path,
    method: str,
    model: str,
    resolved: str,
) -> None:
    """Every ``[impute] method`` trains on data with missing values and +-inf, and predicts every row.

    By default (``auto``), the models that handle missing values get no imputation step. The record keeps the
    method as set in ``settings.impute`` and the method the final model used in ``preprocessing.impute``.
    """
    settings = _settings(tmp_path, f"[impute]\nmethod = '{method}'\n")
    result = optimize_model(with_missing, make_config(model=model, settings=settings))
    preprocess = result.pipeline.named_steps["preprocess"].estimator
    assert ("impute" in preprocess.named_steps) is (resolved != "none")
    assert np.isfinite(result.pipeline.predict(with_missing)).all()
    run = result.to_dict()
    assert run["settings"]["impute"]["method"] == method
    assert run["preprocessing"]["impute"] == {"method": resolved}


def test_clipping_is_recorded_in_run_json(frame: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path) -> None:
    """``preprocessing.clip`` holds the bounds of the final model and the columns without spread."""
    # flat has an IQR of 0, so it is not clipped; it is not constant, so it stays a candidate.
    data = frame.with_columns(flat=pl.when(pl.col("t") < 3).then(1.0).otherwise(0.0))
    settings = _settings(tmp_path, "[clip]\nmethod = 'iqr'\niqr_k = 1.0\n")
    run = optimize_model(data, make_config(settings=settings)).to_dict()
    clip = run["preprocessing"]["clip"]
    assert clip["method"] == "iqr"
    assert clip["unclipped_columns"] == ["flat"]
    assert set(clip["bounds"]) == set(run["candidate_columns"]) - {"flat"}
    q1, q3 = np.quantile(data["s0"].to_numpy(), [0.25, 0.75])
    assert clip["bounds"]["s0"] == pytest.approx([q1 - (q3 - q1), q3 + (q3 - q1)])
    assert run["settings"]["clip"]["iqr_k"] == 1.0


def test_column_related_to_the_target_is_selected(make_config: Callable[..., Config]) -> None:
    """A column that alone drives the target is in ``selected_columns``, whose length is ``best_params.k``."""
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(100, 8))
    signal = rng.normal(size=100)
    data = pl.DataFrame(
        {**{f"n{i}": noise[:, i] for i in range(8)}, "signal": signal, "z": 3.0 * signal + rng.normal(scale=0.1, size=100)}
    )
    result = optimize_model(data, make_config(target="z", exclude_columns=(), n_trials=5))
    assert "signal" in result.selected_columns
    assert len(result.selected_columns) == result.best_params["k"]
    assert result.selected_columns == [name for name in result.columns.candidates if name in result.selected_columns]


def test_columns_are_ranked_on_the_labeled_training_rows_only(
    frame: pl.DataFrame, make_config: Callable[..., Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each fold picks and ranks the representatives once, on its training rows (never its test rows).

    The final model does so on all rows.
    """
    data = pl.concat([frame, frame.head(20).with_columns(y=pl.lit(None, dtype=pl.Float64), t=pl.col("t") + 1000)])
    plans: list[CVPlan] = []
    ranked: list[np.ndarray] = []

    def record_plan(*args: Any) -> CVPlan:  # noqa: ANN401 - the arguments of plan_cv
        plans.append(plan_cv(*args))
        return plans[-1]

    def record_ranking(task: str, preprocess: Any, x: np.ndarray, *args: Any) -> np.ndarray:  # noqa: ANN401 - the arguments of rank_columns
        ranked.append(x)
        return rank_columns(task, preprocess, x, *args)

    monkeypatch.setattr(mloom.training, "plan_cv", record_plan)
    monkeypatch.setattr(mloom.optimize, "rank_columns", record_ranking)
    result = optimize_model(data, make_config(n_trials=4))

    t = result.columns.candidates.index("t")
    times = frame["t"].to_numpy()
    [plan] = plans
    assert len(ranked) == len(plan.folds) + 1
    for (train, _), x in zip(plan.folds, ranked, strict=False):
        np.testing.assert_array_equal(x[:, t], times[train])
    np.testing.assert_array_equal(ranked[-1][:, t], times)
