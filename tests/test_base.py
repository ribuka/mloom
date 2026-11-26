"""Tests of reading a base ``model.pkl`` (``load_base``) and of keeping it from being overwritten."""

import os
import pickle
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mloom import Config, InputError, load_base, optimize_model, read_run
from mloom.base import check_output
from mloom.settings import settings_from_record


@pytest.fixture
def base_path(frame: pl.DataFrame, make_config: Callable[..., Config], tmp_path: Path) -> Path:
    """Save a model with filters, a time split and settings of a project file, and return its path."""
    settings = tmp_path / "mloom.toml"
    settings.write_text("[clip]\nmethod = 'iqr'\n", encoding="utf-8")
    config = make_config(
        cv="timeseries",
        time_column="t",
        metric="mae",
        seed=3,
        exclude_columns=("c", "absent"),
        filters=({"column": "s5", "op": "lt", "value": 2},),
        train_filters=({"column": "t", "op": "lt", "value": 100},),
        settings=settings,
    )
    return optimize_model(frame, config).save(tmp_path / "base")


def _write_record(path: Path, record: dict[str, Any]) -> Path:
    """Write a ``model.pkl`` that holds only a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(record))
    return path


def test_load_base_reads_how_the_base_was_made(base_path: Path, make_config: Callable[..., Config]) -> None:
    """The config, the settings, the columns and the parameters come from the record."""
    base = load_base(base_path)

    record = read_run(base_path)
    assert base.path == base_path
    assert base.record == record
    expected = make_config(
        cv="timeseries",
        time_column="t",
        metric="mae",
        seed=3,
        exclude_columns=("c", "absent"),
        filters=({"column": "s5", "op": "lt", "value": 2},),
        train_filters=({"column": "t", "op": "lt", "value": 100},),
    )
    assert base.config == replace(expected, unavailable_sensors=Config.unavailable_sensors, settings=Config.settings)
    # A user-excluded column that is not in the data is kept in the config.
    assert record["exclude_columns"] == ["c", "absent"]
    assert base.settings.clip.method == "iqr"
    assert base.settings.to_dict()["columns"] == record["settings"]["columns"]
    assert base.selected_columns == record["selected_columns"]
    assert base.best_params == record["best_params"]


def test_settings_of_the_record_are_read_from_the_base(base_path: Path) -> None:
    """Every key of the settings is recorded as read from the base, the only file looked up."""
    settings = load_base(base_path).settings

    assert {path for table in settings.sources.values() for path in table.values()} == {str(base_path)}
    assert [(file.role, file.path, file.found) for file in settings.files] == [("base", str(base_path), True)]
    assert settings_from_record(settings.to_dict(), "other").clip == settings.clip


def test_base_without_a_search_takes_the_default_trials(base_path: Path, tmp_path: Path) -> None:
    """A base whose ``hpo`` is null (a refit) has the default ``n_trials`` and ``timeout``."""
    record = {**read_run(base_path), "hpo": None}
    base = load_base(_write_record(tmp_path / "refit" / "model.pkl", record))
    assert (base.config.n_trials, base.config.timeout) == (Config.n_trials, Config.timeout)


def test_missing_file_or_record_raises(tmp_path: Path) -> None:
    """A path that does not exist, or a model.pkl without a record, raises ``InputError``."""
    with pytest.raises(InputError, match="model file not found"):
        load_base(tmp_path / "missing" / "model.pkl")
    with pytest.raises(InputError, match="make it again with mloom optimize"):
        load_base(_write_record(tmp_path / "old" / "model.pkl", {"pipeline": None}))


def _without(record: dict[str, Any], key: str) -> dict[str, Any]:
    *parents, last = key.split(".")
    copy = pickle.loads(pickle.dumps(record))
    table = copy
    for part in parents:
        table = table[part]
    del table[last]
    return copy


@pytest.mark.parametrize(
    "key",
    [
        "target",
        "task",
        "model",
        "metric",
        "seed",
        "cv.method",
        "cv.n_splits",
        "cv.time_column",
        "cv.group_column",
        "exclude_columns",
        "filters",
        "train_filters",
        "hpo",
        "hpo.n_trials",
        "hpo.timeout",
        "selected_columns",
        "best_params",
        "created_at",
        "cv_scores",
        "settings",
        "settings.clip",
        "settings.clip.method",
        "settings.importance.n_repeats",
    ],
)
def test_missing_key_raises_with_the_key_and_the_version(base_path: Path, tmp_path: Path, key: str) -> None:
    """A record without a key the base needs raises ``InputError`` naming the key and who made the record."""
    path = _write_record(tmp_path / "broken" / "model.pkl", _without(read_run(base_path), key))
    with pytest.raises(InputError) as error:
        load_base(path)
    message = str(error.value)
    assert f"{key} " in message
    assert f"mloom {version('mloom')}" in message
    assert "make the base again with mloom optimize" in message


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("task", "foo", "task must be one of"),
        ("filters", [{"column": "s0", "op": "foo", "value": 1}], r"filters\[0\]"),
        ("seed", True, "seed must be int"),
        ("selected_columns", ["s0", 1], "selected_columns must be a list of column names"),
        ("settings", {"clip": {"method": "foo"}}, "settings.columns is missing"),
    ],
)
def test_invalid_value_raises_with_the_version(
    base_path: Path, tmp_path: Path, key: str, value: object, match: str
) -> None:
    """A record whose value is invalid raises ``InputError`` naming it and who made the record."""
    path = _write_record(tmp_path / "broken" / "model.pkl", {**read_run(base_path), key: value})
    with pytest.raises(InputError, match=match) as error:
        load_base(path)
    assert f"mloom {version('mloom')}" in str(error.value)


def test_invalid_setting_raises_with_its_key(base_path: Path, tmp_path: Path) -> None:
    """A setting that ``mloom.toml`` would not accept raises ``InputError`` naming its key."""
    record = read_run(base_path)
    record["settings"]["clip"]["iqr_k"] = -1
    path = _write_record(tmp_path / "broken" / "model.pkl", record)
    with pytest.raises(InputError, match="settings.clip.iqr_k must be a positive number"):
        load_base(path)


def test_output_must_not_be_the_base(base_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The directory of the base, however it is written, is refused; another directory is not."""
    monkeypatch.chdir(tmp_path)
    for output in (base_path.parent, Path("base"), "./base", "base/", f"base{os.sep}", "other/../base"):
        with pytest.raises(InputError, match="would overwrite the base"):
            check_output(output, base_path)
    with pytest.raises(InputError, match="would overwrite the base"):
        check_output(base_path.parent, Path("base") / "model.pkl")
    check_output(tmp_path / "other", base_path)
