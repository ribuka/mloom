"""Tests of ``optimize_model`` with a base: the parameters of the base tried first, and the base scored on the same folds."""

import json
import pickle
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mloom import Base, Result, optimize_model, read_run, refit_model
from mloom.cli import EXIT_OK, main


@pytest.fixture
def base_of(make_base: Callable[..., tuple[Result, Base]]) -> Callable[..., Base]:
    """Make a base; keyword arguments go to the config. Its unavailable sensors are not read (as in the base)."""

    def make(**overrides: object) -> Base:
        _, base = make_base(**overrides)
        return replace(base, config=replace(base.config, unavailable_sensors=None))

    return make


def _record(result: Result) -> dict:
    return result.to_dict()["base"]


@pytest.mark.parametrize("metric", ["rmse", "r2"])
def test_same_data_tries_the_base_first_and_scores_it_on_the_same_folds(
    frame: pl.DataFrame, base_of: Callable[..., Base], metric: str
) -> None:
    """With the data and the arguments of the base, trial 0 is the base, and the base is scored as a refit scores it."""
    base = base_of(metric=metric)
    base_mean = base.record["cv_scores"]["mean"]
    better = np.less_equal if metric == "rmse" else np.greater_equal

    first = optimize_model(frame, replace(base.config, n_trials=1), base=base)
    assert first.best_params == base.best_params
    assert first.cv_scores["folds"] == pytest.approx(base.record["cv_scores"]["folds"], rel=1e-12)

    result = optimize_model(frame, base.config, base=base)
    record = _record(result)
    assert better(result.cv_scores["mean"], base_mean) or result.cv_scores["mean"] == pytest.approx(base_mean)
    assert (record["mode"], record["enqueued"], record["dropped_columns"], record["not_compared"]) == (
        "optimize",
        True,
        [],
        [],
    )
    assert (record["path"], record["cv_scores"]) == (str(base.path), base.record["cv_scores"])
    # The columns and the parameters of the base, fixed on the folds of this run: what a refit on the same data scores.
    fixed = refit_model(frame, base, unavailable_sensors=None).cv_scores
    assert record["cv_scores_on_this_data"] == pytest.approx(fixed, rel=1e-12)

    ours, theirs = result.cv_scores["folds"], record["cv_scores_on_this_data"]["folds"]
    expected = [b - o for o, b in zip(ours, theirs, strict=True)]
    if metric == "r2":
        expected = [o - b for o, b in zip(ours, theirs, strict=True)]
    comparison = record["comparison"]
    assert comparison["folds"] == pytest.approx(expected, rel=1e-12)
    assert comparison["mean"] == pytest.approx(np.mean(expected), rel=1e-12)
    assert comparison["std"] == pytest.approx(np.std(expected), rel=1e-12)


def test_another_model_scores_the_base_without_trying_its_parameters(
    frame: pl.DataFrame, base_of: Callable[..., Base]
) -> None:
    """Another model searches its own space, and the model of the base is still scored on the same folds."""
    base = base_of()
    record = _record(optimize_model(frame, replace(base.config, model="random_forest", n_trials=2), base=base))
    assert record["enqueued"] is False
    assert record["not_compared"] == []
    fixed = refit_model(frame, base, unavailable_sensors=None).cv_scores
    assert record["cv_scores_on_this_data"] == pytest.approx(fixed, rel=1e-12)


@pytest.mark.parametrize(
    ("change", "reasons"),
    [
        ({"metric": "mae"}, ["metric"]),
        ({"target": "s5"}, ["target"]),
    ],
)
def test_another_target_or_metric_is_not_compared(
    frame: pl.DataFrame, base_of: Callable[..., Base], change: dict, reasons: list[str]
) -> None:
    """Scores of another target or metric mean something else, so the base is not scored."""
    base = base_of()
    record = _record(optimize_model(frame, replace(base.config, n_trials=1, **change), base=base))
    assert (record["cv_scores_on_this_data"], record["comparison"], record["not_compared"]) == (None, None, reasons)


def test_parameters_out_of_the_search_space_are_not_tried(frame: pl.DataFrame, base_of: Callable[..., Base]) -> None:
    """A parameter of the base out of the search space of this run is not enqueued, with a warning."""
    base = base_of()
    outside = replace(base, best_params={**base.best_params, "alpha": 1e9})

    with pytest.warns(UserWarning, match=r"not tried first: alpha=1000000000\.0 not in the search space"):
        result = optimize_model(frame, replace(base.config, n_trials=1), base=outside)

    assert _record(result)["enqueued"] is False
    assert result.best_params["alpha"] != 1e9


def test_k_is_kept_within_the_columns_of_this_run(frame: pl.DataFrame, base_of: Callable[..., Base]) -> None:
    """``k`` of the base above the number of clusters of this run is lowered to it."""
    base = base_of()
    many = replace(base, best_params={**base.best_params, "k": 1000})
    result = optimize_model(frame, replace(base.config, n_trials=1), base=many)
    assert _record(result)["enqueued"] is True
    assert result.best_params["k"] == result.to_dict()["selection"]["n_clusters"]


def test_columns_no_longer_usable_are_dropped_from_the_score_of_the_base(
    frame: pl.DataFrame, base_of: Callable[..., Base], tmp_path: Path
) -> None:
    """A selected column that is now an unavailable sensor is dropped from the base, which is still scored."""
    base = base_of()
    broken = base.selected_columns[0]
    sensors = tmp_path / "unavailable-sensors.json"
    sensors.write_text(json.dumps({"sensors": [{"name": broken}]}), encoding="utf-8")
    config = replace(base.config, unavailable_sensors=sensors, n_trials=1)

    with pytest.warns(UserWarning, match=f"column {broken!r} of the base is dropped: unavailable_sensor"):
        record = _record(optimize_model(frame, config, base=base))

    assert record["dropped_columns"] == [{"name": broken, "reason": "unavailable_sensor"}]
    assert record["cv_scores_on_this_data"] is not None
    assert record["not_compared"] == []


def test_no_column_of_the_base_left_still_searches(
    data_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When every selected column of the base is unavailable, the search still ends with exit code 0, not compared."""
    base_dir = tmp_path / "base"
    common = ["--target", "y", "--task", "regression", "--model", "ridge", "--cv", "kfold", "--n-splits", "3"]
    assert main(["optimize", str(data_path), *common, "--n-trials", "2", "--output", str(base_dir)]) == EXIT_OK
    selected = read_run(base_dir / "model.pkl")["selected_columns"]
    sensors = tmp_path / "sensors.json"
    sensors.write_text(json.dumps({"sensors": [{"name": name} for name in selected]}), encoding="utf-8")
    capsys.readouterr()

    args = ["optimize", str(data_path), "--base", str(base_dir / "model.pkl"), "--output", str(tmp_path / "new")]
    assert main([*args, "--unavailable-sensors", str(sensors)]) == EXIT_OK

    err = capsys.readouterr().err
    assert all(f"warning: column {name!r} of the base is dropped: unavailable_sensor" in err for name in selected)
    record = read_run(tmp_path / "new" / "model.pkl")["base"]
    assert (record["cv_scores_on_this_data"], record["comparison"]) == (None, None)
    assert "columns" in record["not_compared"]


def test_cli_warns_of_parameters_out_of_the_search_space(
    data_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``n_components`` of PLS above the columns of this run is not tried first, with a ``warning:`` line."""
    base_dir = tmp_path / "base"
    common = ["--target", "y", "--task", "regression", "--model", "pls", "--cv", "kfold", "--n-splits", "3"]
    assert main(["optimize", str(data_path), *common, "--n-trials", "2", "--output", str(base_dir)]) == EXIT_OK
    record = read_run(base_dir / "model.pkl")
    # load_base accepts it: the bound that depends on the columns is checked by the search only.
    record["best_params"]["n_components"] = 20
    (base_dir / "model.pkl").write_bytes(pickle.dumps(record))
    capsys.readouterr()

    args = ["optimize", str(data_path), "--base", str(base_dir / "model.pkl"), "--output", str(tmp_path / "new")]
    assert main([*args, "--n-trials", "1"]) == EXIT_OK

    assert "warning: the parameters of the base are not tried first: n_components=20" in capsys.readouterr().err
    run = read_run(tmp_path / "new" / "model.pkl")
    assert run["base"]["enqueued"] is False
    assert run["best_params"]["n_components"] != 20


def test_settings_of_the_base_are_kept_by_replace(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]], tmp_path: Path
) -> None:
    """``replace(base.config, ...)`` keeps the settings of the base, without reading the file again."""
    settings = tmp_path / "mloom.toml"
    settings.write_text("[clip]\nmethod = 'iqr'\n", encoding="utf-8")
    _, base = make_base(settings=settings)
    settings.unlink()

    result = optimize_model(frame, replace(base.config, n_trials=3, unavailable_sensors=None), base=base)

    recorded = result.to_dict()["settings"]
    assert recorded["clip"]["method"] == "iqr"
    assert [file["role"] for file in recorded["files"]] == ["base"]
