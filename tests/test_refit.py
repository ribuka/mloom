"""Tests of ``refit_model``: the model of a base fitted again, without a search."""

import pickle
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest

from mloom import (
    Base,
    Config,
    InputError,
    Result,
    load_base,
    optimize_model,
    read_run,
    refit_model,
)
from mloom.models import MODELS

MODEL_TASKS = [(model, task) for model, spec in MODELS.items() for task in spec.builders]


@pytest.fixture
def make_base(
    frame: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path
) -> Callable[..., tuple[Result, Base]]:
    """Optimize on the shared frame, save to ``<tmp>/base`` and load it as a base; keyword arguments go to the config."""

    def make(**overrides: object) -> tuple[Result, Base]:
        result = optimize_model(frame, make_config(**overrides))
        return result, load_base(result.save(tmp_path / "base"))

    return make


def _predictions(result: Result, frame: pl.DataFrame) -> np.ndarray:
    pipeline = result.pipeline
    return pipeline.predict_proba(frame) if hasattr(pipeline, "predict_proba") else pipeline.predict(frame)


@pytest.mark.parametrize(("model", "task"), MODEL_TASKS)
def test_same_data_gives_the_same_model(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]], model: str, task: str
) -> None:
    """On the data of the base, the refit uses the same columns and parameters, and predicts the same."""
    optimized, base = make_base(model=model, task=task, target="y" if task == "regression" else "c")

    result = refit_model(frame, base, unavailable_sensors=None)

    assert result.selected_columns == optimized.selected_columns
    assert result.best_params == optimized.best_params
    np.testing.assert_array_equal(_predictions(result, frame), _predictions(optimized, frame))
    run = result.to_dict()
    assert (run["hpo"], run["selection"]) == (None, None)
    assert run["base"] == {
        "path": str(base.path),
        "mode": "refit",
        "created_at": optimized.created_at,
        "mloom_version": base.record["mloom_version"],
        "cv_scores": optimized.cv_scores,
        "dropped_columns": [],
    }
    groups = run["interpretation"]["permutation_importance"]["groups"]
    assert sorted(group["column"] for group in groups) == sorted(result.selected_columns)
    assert all(group["columns"] == [group["column"]] for group in groups)
    # The CV scores of the base select the columns on each fold; those of a refit do not, so only a refit repeats.
    assert refit_model(frame, base, unavailable_sensors=None).cv_scores == result.cv_scores


def test_base_without_its_model_is_refitted(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]], tmp_path: Path
) -> None:
    """A base whose ``model.pkl`` holds only the record (its model unreadable) refits the same."""
    optimized, base = make_base(model="random_forest")
    record_only = tmp_path / "record-only" / "model.pkl"
    record_only.parent.mkdir()
    record_only.write_bytes(pickle.dumps(read_run(base.path)))

    result = refit_model(frame, load_base(record_only), unavailable_sensors=None)

    assert result.selected_columns == optimized.selected_columns
    assert result.best_params == optimized.best_params


def test_new_rows_are_fitted_without_a_search(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """More rows refit the fixed columns and parameters; no study is created."""
    optimized, base = make_base()
    rng = np.random.default_rng(1)
    added = frame.head(30).with_columns(
        t=pl.col("t") + 1000, s0=pl.col("s0") + rng.normal(scale=0.1, size=30), y=pl.col("y") + 0.05
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("a study was created")

    monkeypatch.setattr(optuna, "create_study", fail)
    result = refit_model(pl.concat([frame, added]), base, unavailable_sensors=None)

    assert result.selected_columns == optimized.selected_columns
    assert {**result.best_params, "k": None} == {**optimized.best_params, "k": None}
    assert result.data.n_rows_used == optimized.data.n_rows_used + 30


def test_columns_no_longer_usable_are_dropped_with_their_reason(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]], tmp_path: Path
) -> None:
    """A selected column that is now an unavailable sensor, or missing from the data, is dropped and recorded."""
    optimized, base = make_base()
    broken, removed, *kept = optimized.selected_columns
    sensors = tmp_path / "unavailable-sensors.json"
    sensors.write_text(f'{{"sensors": [{{"name": "{broken}", "reason": "故障"}}]}}', encoding="utf-8")

    with pytest.warns(UserWarning) as warned:
        result = refit_model(frame.drop(removed), base, unavailable_sensors=sensors)

    assert result.selected_columns == kept
    assert result.best_params["k"] == len(kept)
    assert result.to_dict()["base"]["dropped_columns"] == [
        {"name": broken, "reason": "unavailable_sensor"},
        {"name": removed, "reason": "not_in_data"},
    ]
    assert [str(warning.message) for warning in warned] == [
        f"column {broken!r} of the base is dropped: unavailable_sensor",
        f"column {removed!r} of the base is dropped: not_in_data",
    ]


def test_no_usable_column_raises(frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]]) -> None:
    """When no selected column is left, ``InputError`` names them and their reasons."""
    optimized, base = make_base()
    with pytest.raises(InputError, match="no column the base selected is left") as error:
        refit_model(frame.drop(optimized.selected_columns), base, unavailable_sensors=None)
    assert all(f"{name} (not_in_data)" in str(error.value) for name in optimized.selected_columns)


def test_no_candidate_at_all_raises_with_the_dropped_columns(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]]
) -> None:
    """When the data leaves no candidate at all, the error still names each selected column and its reason."""
    optimized, base = make_base()
    selected = optimized.selected_columns
    data = frame.select("y", *selected).with_columns(pl.lit(None, dtype=pl.Float64).alias(name) for name in selected)
    with pytest.raises(InputError, match="no column the base selected is left") as error:
        refit_model(data, base, unavailable_sensors=None)
    assert all(f"{name} (all_null)" in str(error.value) for name in selected)


def test_pls_components_are_limited_to_the_columns_left(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]]
) -> None:
    """PLS keeps its ``n_components`` in the record, and fits no more components than columns."""
    optimized, base = make_base(model="pls")
    kept = optimized.selected_columns[-1]
    with pytest.warns(UserWarning):
        result = refit_model(frame.drop(optimized.selected_columns[:-1]), base, unavailable_sensors=None)
    assert result.selected_columns == [kept]
    assert result.best_params["n_components"] == optimized.best_params["n_components"]
    assert result.pipeline[-1].n_components == 1


def test_settings_come_from_the_base_unless_a_file_is_given(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]], tmp_path: Path
) -> None:
    """Without ``settings``, those recorded in the base are used; with a file, it is read as optimize reads it."""
    settings = tmp_path / "mloom.toml"
    settings.write_text("[clip]\nmethod = 'iqr'\n", encoding="utf-8")
    _, base = make_base(settings=settings)
    settings.unlink()

    recorded = refit_model(frame, base, unavailable_sensors=None).to_dict()["settings"]
    assert recorded["clip"]["method"] == "iqr"
    assert {path for table in recorded["sources"].values() for path in table.values()} == {str(base.path)}
    assert [file["role"] for file in recorded["files"]] == ["base"]

    settings.write_text("[clip]\nmethod = 'none'\n", encoding="utf-8")
    recorded = refit_model(frame, base, unavailable_sensors=None, settings=settings).to_dict()["settings"]
    assert recorded["clip"]["method"] == "none"
    assert [file["role"] for file in recorded["files"]] == ["project", "local_defaults", "defaults"]


def test_filters_come_from_the_base_unless_given(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]]
) -> None:
    """Without filters, those of the base are kept; given ones replace them."""
    optimized, base = make_base(train_filters=({"column": "t", "op": "lt", "value": 90},))

    kept = refit_model(frame, base, unavailable_sensors=None)
    assert (kept.config.filters, kept.config.train_filters) == (optimized.config.filters, optimized.config.train_filters)
    assert kept.data.n_rows_train_filtered_out == optimized.data.n_rows_train_filtered_out == 30

    wider = ({"column": "t", "op": "lt", "value": 110},)
    replaced = refit_model(frame, base, unavailable_sensors=None, filters=(), train_filters=wider)
    assert (replaced.config.filters, replaced.config.train_filters) == ((), wider)
    assert replaced.data.n_rows_train_filtered_out == 10


def test_refit_is_never_saved_over_its_base(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]], tmp_path: Path
) -> None:
    """Saving a refit in the directory of its base raises ``InputError`` and leaves the base as it was."""
    _, base = make_base()
    before = (Path(base.path).stat().st_mtime_ns, Path(base.path).read_bytes())
    result = refit_model(frame, base, unavailable_sensors=None)

    with pytest.raises(InputError, match="would overwrite the base"):
        result.save(tmp_path / "base")
    assert (Path(base.path).stat().st_mtime_ns, Path(base.path).read_bytes()) == before
    assert read_run(result.save(tmp_path / "refit"))["base"]["path"] == str(base.path)


_WITHOUT_MLOOM = """
import pickle
import sys

sys.modules["mloom"] = None  # any import of mloom fails
import numpy as np
import polars as pl

with open(sys.argv[1], "rb") as file:
    run = pickle.load(file)
    bundle = pickle.load(file)
assert run["base"]["mode"] == "refit"
np.save(sys.argv[3], bundle["pipeline"].predict(pl.read_parquet(sys.argv[2])))
"""


def test_refit_is_read_and_predicts_without_mloom(
    frame: pl.DataFrame, make_base: Callable[..., tuple[Result, Base]], tmp_path: Path
) -> None:
    """A Python that cannot import mloom reads the ``model.pkl`` of a refit and predicts what mloom predicts."""
    _, base = make_base(model="lightgbm")
    result = refit_model(frame, base, unavailable_sensors=None)
    paths = [str(result.save(tmp_path / "refit")), str(tmp_path / "data.parquet"), str(tmp_path / "predictions.npy")]
    frame.write_parquet(paths[1])

    completed = subprocess.run(
        [sys.executable, "-I", "-c", _WITHOUT_MLOOM, *paths], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(np.load(paths[2]), result.pipeline.predict(frame))
