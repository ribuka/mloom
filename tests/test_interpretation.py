"""Tests of the interpretation in the record: permutation importance, coefficients and VIP."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import mloom.interpretation
import mloom.optimize
from mloom import Config, optimize_model
from mloom.interpretation import Importance
from mloom.models import MODELS

N_ROWS = 150


def _importance_settings(tmp_path: Path, n_repeats: int) -> Path:
    path = tmp_path / "mloom.toml"
    path.write_text(f"[importance]\nn_repeats = {n_repeats}\n", encoding="utf-8")
    return path


@pytest.fixture
def strong_and_weak() -> pl.DataFrame:
    """``y = 3a + b`` with a little noise, ``a2`` almost equal to ``a``, and noise columns."""
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=(2, N_ROWS))
    noise = rng.normal(size=(N_ROWS, 3))
    return pl.DataFrame(
        {
            "a": a,
            "a2": a + rng.normal(scale=0.01, size=N_ROWS),
            "b": b,
            **{f"n{i}": noise[:, i] for i in range(3)},
            "y": 3.0 * a + b + rng.normal(scale=0.1, size=N_ROWS),
        }
    )


@pytest.fixture
def scaled() -> pl.DataFrame:
    """Build a column of mean 300 and std 50 among unit columns, with numeric, 2-class and 3-class targets."""
    rng = np.random.default_rng(5)
    a, b, n = rng.normal(size=(3, N_ROWS))
    big = rng.normal(loc=300.0, scale=50.0, size=N_ROWS)
    signal = a + 0.5 * b + (big - 300.0) / 50.0
    return pl.DataFrame(
        {
            "a": a,
            "b": b,
            "n": n,
            "big": big,
            "y": signal + rng.normal(scale=0.3, size=N_ROWS),
            "c2": np.where(signal + rng.normal(scale=0.5, size=N_ROWS) > 0, "pos", "neg"),
            "c3": np.digitize(signal + rng.normal(scale=0.5, size=N_ROWS), [-0.7, 0.7]).astype(str),
        }
    )


@pytest.mark.parametrize("model", ["ridge", "lightgbm"])
def test_strong_group_is_most_important(
    strong_and_weak: pl.DataFrame, make_config: Callable[..., Config], model: str
) -> None:
    """The group of ``a`` and ``a2`` comes first, and ``b`` matters less but more than nothing."""
    run = optimize_model(strong_and_weak, make_config(model=model, n_trials=20)).to_dict()
    importance = run["interpretation"]["permutation_importance"]
    groups = importance["groups"]
    assert (importance["metric"], importance["n_repeats"]) == ("rmse", 5)
    assert len(groups) == len(run["selected_columns"])
    assert groups[0]["column"] in {"a", "a2"}
    assert groups[0]["columns"] == ["a", "a2"]
    [weak] = [group for group in groups if group["column"] == "b"]
    assert weak["columns"] == ["b"]
    assert 0 < weak["mean"] < groups[0]["mean"]
    for group in groups:
        assert len(group["folds"]) == 3
        assert group["mean"] == pytest.approx(np.mean(group["folds"]), rel=1e-12)
        assert group["std"] == pytest.approx(np.std(group["folds"]), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(("model", "task"), [(model, task) for model, spec in MODELS.items() for task in spec.builders])
def test_baselines_are_the_scores_of_the_best_trial(
    strong_and_weak: pl.DataFrame, make_config: Callable[..., Config], monkeypatch: pytest.MonkeyPatch, model: str, task: str
) -> None:
    """The models refitted on each fold score on their test side what the best trial scored."""
    frame = strong_and_weak.with_columns(c=pl.when(pl.col("y") > 0).then(pl.lit("pos")).otherwise(pl.lit("neg")))
    measured: list[Importance] = []
    original = mloom.optimize.permutation_importance

    def record(*args: Any) -> Importance:  # noqa: ANN401 - passed through
        measured.append(original(*args))
        return measured[-1]

    monkeypatch.setattr(mloom.optimize, "permutation_importance", record)
    target = "y" if task == "regression" else "c"
    config = make_config(model=model, task=task, target=target, exclude_columns=("y", "c"))
    result = optimize_model(frame, config)
    [importance] = measured
    assert importance.baselines == pytest.approx(result.cv_scores["folds"], rel=1e-12)


@pytest.mark.parametrize("model", list(MODELS))
def test_same_seed_gives_same_interpretation(
    strong_and_weak: pl.DataFrame, make_config: Callable[..., Config], model: str
) -> None:
    """Two runs with the same data, config and seed record the same interpretation."""
    config = make_config(model=model)
    first, second = optimize_model(strong_and_weak, config), optimize_model(strong_and_weak, config)
    assert first.to_dict()["interpretation"] == second.to_dict()["interpretation"]


@pytest.mark.parametrize("model", list(MODELS))
def test_zero_repeats_computes_no_importance(
    strong_and_weak: pl.DataFrame,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: str,
) -> None:
    """``n_repeats = 0`` fits no model on the folds, and the other interpretations still follow the model."""

    def fail(*args: Any) -> None:  # noqa: ANN401 - any arguments
        raise AssertionError("a fold was fitted")

    monkeypatch.setattr(mloom.interpretation, "_fit_fold", fail)
    run = optimize_model(strong_and_weak, make_config(model=model, settings=_importance_settings(tmp_path, 0))).to_dict()
    interpretation = run["interpretation"]
    assert interpretation["permutation_importance"] is None
    assert (interpretation["coefficients"] is None) == (model not in {"ridge", "pls"})
    assert (interpretation["vip"] is None) == (model != "pls")
    assert run["settings"]["importance"] == {"n_repeats": 0}
    assert run["settings"]["sources"]["importance"] == {"n_repeats": str(tmp_path / "mloom.toml")}


@pytest.mark.parametrize("model", list(MODELS))
def test_interpretations_that_do_not_apply_are_null(
    strong_and_weak: pl.DataFrame, make_config: Callable[..., Config], model: str
) -> None:
    """Coefficients are only for the linear models and VIP only for PLS; the record has no NaN."""
    run = optimize_model(strong_and_weak, make_config(model=model)).to_dict()
    interpretation = run["interpretation"]
    assert list(run).index("interpretation") == list(run).index("cv_scores") + 1
    assert interpretation["permutation_importance"] is not None
    assert (interpretation["coefficients"] is None) == (model in {"random_forest", "lightgbm"})
    assert (interpretation["vip"] is None) == (model != "pls")
    json.dumps(run, allow_nan=False)


@pytest.mark.parametrize(
    ("model", "task", "target"),
    [("ridge", "regression", "y"), ("pls", "regression", "y"), ("ridge", "classification", "c2"), ("ridge", "classification", "c3")],
)
def test_coefficients_are_the_change_of_the_output(
    scaled: pl.DataFrame, make_config: Callable[..., Config], model: str, task: str, target: str
) -> None:
    """``per_unit`` is the change of the output when a column moves by 1, and ``per_std`` is it times σ."""
    config = make_config(model=model, task=task, target=target, exclude_columns=("y", "c2", "c3"), n_trials=10)
    result = optimize_model(scaled, config)
    pipeline = result.pipeline
    coefficients = result.interpretation["coefficients"]
    assert coefficients["output"] == ("prediction" if task == "regression" else "log_odds")
    output = pipeline.decision_function if task == "classification" else pipeline.predict
    classes = [str(label) for label in pipeline.classes_] if target == "c3" else [None]
    scaler = pipeline.named_steps["preprocess"].estimator.named_steps["scale"]
    candidates = result.columns.candidates
    row = scaled[:1]
    before = np.asarray(output(row), dtype=np.float64).reshape(-1)
    assert [entry["class"] for entry in coefficients["columns"]] == [
        label for label in classes for _ in result.selected_columns
    ]
    for entry in coefficients["columns"]:
        changed = row.with_columns(pl.col(entry["column"]) + 1.0)
        after = np.asarray(output(changed), dtype=np.float64).reshape(-1)
        index = 0 if entry["class"] is None else classes.index(entry["class"])
        assert after[index] - before[index] == pytest.approx(entry["per_unit"], rel=1e-9)
        sigma = scaler.scale_[candidates.index(entry["column"])]
        assert entry["per_std"] == pytest.approx(entry["per_unit"] * sigma, rel=1e-12)
    for label in classes:
        per_std = [abs(entry["per_std"]) for entry in coefficients["columns"] if entry["class"] == label]
        assert per_std == sorted(per_std, reverse=True)


def test_vip_follows_its_definition(scaled: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """The squares of the VIP sum to the number of selected columns, and the strongest column has the largest."""
    strong = scaled.with_columns(y=5.0 * pl.col("a") + pl.col("y"))
    result = optimize_model(strong, make_config(model="pls", exclude_columns=("y", "c2", "c3"), n_trials=10))
    vip = result.interpretation["vip"]["columns"]
    values = [entry["vip"] for entry in vip]
    assert sum(value**2 for value in values) == pytest.approx(len(result.selected_columns), rel=1e-9)
    assert vip[0]["column"] == "a"
    assert values == sorted(values, reverse=True)
