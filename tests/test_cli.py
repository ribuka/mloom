"""Tests of the ``mloom`` command line."""

import json
from pathlib import Path

import polars as pl
import pytest

from mloom.cli import EXIT_INPUT, EXIT_OK, main


@pytest.fixture
def data_path(frame: pl.DataFrame, tmp_path: Path) -> Path:
    """Write the shared frame as Parquet and return its path."""
    path = tmp_path / "data.parquet"
    frame.write_parquet(path)
    return path


def _args(data_path: Path, output: Path, *extra: str) -> list[str]:
    return [
        "optimize",
        str(data_path),
        "--target", "y",
        "--task", "regression",
        "--model", "ridge",
        "--cv", "kfold",
        "--n-splits", "3",
        "--n-trials", "2",
        "--exclude-columns", "c",
        "--output", str(output),
        *extra,
    ]  # fmt: skip


def test_optimize_saves_and_prints_run_json(
    data_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On success, model.pkl and run.json are saved and stdout ends with the path of run.json."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "unavailable-sensors.json").write_text('{"sensors": [{"name": "s0"}]}', encoding="utf-8")
    (tmp_path / "config" / "mloom.toml").write_text("[columns]\nmax_missing_rate = 0.3\n", encoding="utf-8")
    output = tmp_path / "out"

    assert main(_args(data_path, output)) == EXIT_OK

    run_path = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert run_path == output / "run.json"
    assert (output / "model.pkl").is_file()
    # The default unavailable-sensors.json and mloom.toml are read from the current directory.
    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert run["unavailable_sensors"]["found"]
    assert "s0" not in run["candidate_columns"]
    assert run["settings"]["max_missing_rate"] == 0.3


def test_invalid_input_exits_with_2(data_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An input error prints a message to stderr and returns exit code 2."""
    assert main(_args(data_path, tmp_path / "out", "--target", "missing")) == EXIT_INPUT
    assert "error: column 'missing' is not in the data" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_invalid_arguments_exit_with_2(data_path: Path, tmp_path: Path) -> None:
    """An argument outside its choices exits with code 2 (argparse)."""
    with pytest.raises(SystemExit) as raised:
        main(_args(data_path, tmp_path / "out", "--model", "svm"))
    assert raised.value.code == EXIT_INPUT
