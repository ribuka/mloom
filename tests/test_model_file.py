"""Tests of the format of ``model.pkl``: the record of the run, then the model."""

import json
import pickle
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import polars as pl
import pytest

import mloom
from mloom import Config, InputError, Result, optimize_model, read_run
from mloom.prediction import load_model

BUNDLE_KEYS = {"pipeline", "filters", "train_filters", "training_rows", "shap_background"}


@pytest.fixture
def result(frame: pl.DataFrame, make_config: Callable[..., Config]) -> Result:
    """Optimize a fast model on the shared frame."""
    return optimize_model(frame, make_config())


def _write(path: Path, *pickles: object) -> Path:
    path.write_bytes(b"".join(pickle.dumps(content) for content in pickles))
    return path


def test_save_writes_only_model_pkl_with_the_record_then_the_model(result: Result, tmp_path: Path) -> None:
    """``save`` writes ``model.pkl`` alone: the record (plain values) is the first pickle, the model the second."""
    path = result.save(tmp_path / "out")

    assert path == tmp_path / "out" / "model.pkl"
    assert [child.name for child in path.parent.iterdir()] == ["model.pkl"]
    with path.open("rb") as file:
        record = pickle.load(file)
        bundle = pickle.load(file)
    assert record == json.loads(json.dumps(result.to_dict()))
    assert isinstance(bundle, dict)
    assert set(bundle) == BUNDLE_KEYS
    assert read_run(path) == record
    assert mloom.read_run is read_run


_RECORD_ONLY = """
import json
import pickle
import sys

for name in ("mloom", "sklearn", "numpy", "polars"):
    sys.modules[name] = None  # any import of them fails
with open(sys.argv[1], "rb") as file:
    print(json.dumps(pickle.load(file)))
"""


def test_record_is_read_without_scikit_learn_or_numpy(result: Result, tmp_path: Path) -> None:
    """A Python that cannot import mloom, scikit-learn, numpy or polars reads the record."""
    path = result.save(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _RECORD_ONLY, str(path)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == read_run(path)


def test_read_run_does_not_read_the_model(tmp_path: Path) -> None:
    """``read_run`` reads a file with only the record; ``load_model`` does not."""
    path = _write(tmp_path / "model.pkl", {"mloom_version": "0.0.0", "name": "記録"})
    assert read_run(path) == {"mloom_version": "0.0.0", "name": "記録"}
    with pytest.raises(InputError, match="has no model after the record of the run"):
        load_model(path)


@pytest.mark.parametrize(
    ("pickles", "message"),
    [
        (None, "model file not found"),
        ((), "is not a model.pkl of mloom"),
        ((dict.fromkeys(BUNDLE_KEYS),), "make it again with mloom optimize"),
        (([1],), "does not start with the record of the run"),
        (({"created_at": "2026-11-26"},), "does not start with the record of the run"),
    ],
    ids=["missing", "empty", "without a record", "not a dict", "no mloom_version"],
)
def test_file_without_a_record_raises(tmp_path: Path, pickles: tuple | None, message: str) -> None:
    """``read_run`` and ``load_model`` raise ``InputError`` for a file that does not start with a record."""
    path = tmp_path / "model.pkl"
    if pickles is not None:
        _write(path, *pickles)
    with pytest.raises(InputError, match=message):
        read_run(path)
    with pytest.raises(InputError, match=message):
        load_model(path)


def test_model_that_is_not_the_bundle_raises(tmp_path: Path) -> None:
    """``load_model`` raises ``InputError`` when the second pickle is not the dict of the model."""
    path = _write(tmp_path / "model.pkl", {"mloom_version": "0.0.0"}, {"pipeline": None})
    with pytest.raises(InputError, match="must be a dict with the keys"):
        load_model(path)


def test_failed_save_keeps_the_previous_model_pkl(
    result: Result, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When writing the model fails, the previous ``model.pkl`` is unchanged and no temporary file is left."""
    path = result.save(tmp_path)
    before = path.read_bytes()
    dump = pickle.dump
    calls = []

    def fail_on_the_model(obj: object, file: BinaryIO) -> None:
        calls.append(obj)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        dump(obj, file)

    monkeypatch.setattr(pickle, "dump", fail_on_the_model)
    with pytest.raises(RuntimeError, match="disk full"):
        result.save(tmp_path)

    assert len(calls) == 2
    assert path.read_bytes() == before
    assert [child.name for child in tmp_path.iterdir()] == ["model.pkl"]
