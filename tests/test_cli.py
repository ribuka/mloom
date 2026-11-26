"""Tests of the ``mloom`` command line."""

import json
from pathlib import Path

import polars as pl
import pytest

import mloom.optimize
from mloom.cli import EXIT_INPUT, EXIT_OK, main
from mloom.prediction import predict


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
    assert run["settings"]["columns"]["max_missing_rate"] == 0.3


def test_invalid_input_exits_with_2(data_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An input error prints a message to stderr and returns exit code 2."""
    assert main(_args(data_path, tmp_path / "out", "--target", "missing")) == EXIT_INPUT
    assert "error: column 'missing' is not in the data" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_no_imputation_with_ridge_exits_with_2_before_training(
    data_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[impute] method = "none"`` with ``ridge`` stops with exit code 2 before anything is fitted."""

    def fail(*args: object) -> None:
        raise AssertionError("the preprocessing was fitted")

    monkeypatch.setattr(mloom.optimize, "fit_preprocess", fail)
    settings = tmp_path / "mloom.toml"
    settings.write_text("[impute]\nmethod = 'none'\n", encoding="utf-8")
    assert main(_args(data_path, tmp_path / "out", "--settings", str(settings))) == EXIT_INPUT
    assert "[impute] method 'none' needs a model that handles missing values; 'ridge' does not" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_invalid_arguments_exit_with_2(data_path: Path, tmp_path: Path) -> None:
    """An argument outside its choices exits with code 2 (argparse)."""
    with pytest.raises(SystemExit) as raised:
        main(_args(data_path, tmp_path / "out", "--model", "svm"))
    assert raised.value.code == EXIT_INPUT


@pytest.fixture
def filters_path(tmp_path: Path) -> Path:
    """Write a filters file with both lists and return its path."""
    path = tmp_path / "filters.json"
    path.write_text(
        json.dumps(
            {
                "filters": [{"column": "s0", "op": "gt", "value": -1}],
                "train_filters": [{"column": "lot", "op": "str_contains_any", "value": ["L1", "L2"]}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_predict_saves_what_the_api_returns(
    data_path: Path, filters_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``optimize --filters`` then ``predict`` save the result of ``mloom.predict``; stdout ends with its path."""
    model_dir = tmp_path / "model"
    assert main(_args(data_path, model_dir, "--filters", str(filters_path))) == EXIT_OK
    run = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
    assert run["filters"] == json.loads(filters_path.read_text(encoding="utf-8"))["filters"]
    assert run["data"]["n_rows_train_filtered_out"] > 0
    capsys.readouterr()

    output = tmp_path / "out" / "predictions.parquet"
    command = ["predict", str(model_dir / "model.pkl"), str(data_path), "--output", str(output)]
    assert main([*command, "--keep-columns", "t", "lot"]) == EXIT_OK

    assert Path(capsys.readouterr().out.strip().splitlines()[-1]) == output
    expected = predict(model_dir / "model.pkl", data_path, keep_columns=("t", "lot"))
    assert pl.read_parquet(output).equals(expected)
    assert expected["used_for_training"].sum() == run["data"]["n_rows_used"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{", "is not valid JSON"),
        ('{"filters": [{"column": "s0", "op": "above", "value": 0}]}', "unknown op"),
        ('{"filters": [{"column": "nope", "op": "is_null"}]}', "column 'nope' is not in the data"),
        ('{"train_filters": [{"column": "s0", "op": "lt", "value": "0"}]}', "does not fit the column"),
        ('{"filters": [], "where": []}', "unknown keys ['where']"),
        ('{"filters": [{"column": "s0", "op": "is_between", "value": [0]}]}', "needs a list [lower, upper]"),
        ('{"filters": [{"column": "s0", "op": "gt", "value": 100}]}', "no row is left after applying filters"),
    ],
)
def test_invalid_filters_exit_with_2(
    data_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, message: str
) -> None:
    """An invalid filters file stops before training with exit code 2."""
    path = tmp_path / "filters.json"
    path.write_text(content, encoding="utf-8")
    assert main(_args(data_path, tmp_path / "out", "--filters", str(path))) == EXIT_INPUT
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_invalid_predict_input_exits_with_2(
    data_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing model or a column of ``--keep-columns`` not in the data exits with code 2."""
    output = tmp_path / "predictions.parquet"
    assert main(["predict", str(tmp_path / "none.pkl"), str(data_path), "--output", str(output)]) == EXIT_INPUT
    assert "model file not found" in capsys.readouterr().err
    assert main(_args(data_path, tmp_path / "model")) == EXIT_OK
    command = ["predict", str(tmp_path / "model" / "model.pkl"), str(data_path), "--output", str(output)]
    assert main([*command, "--keep-columns", "nope"]) == EXIT_INPUT
    assert "['nope'] not in the data" in capsys.readouterr().err
    assert not output.exists()


@pytest.fixture
def first_half_model(data_path: Path, tmp_path: Path) -> Path:
    """Fit a time series model on the first half of the time (``train_filters``) and return its model.pkl."""
    filters = tmp_path / "first-half.json"
    filters.write_text(json.dumps({"train_filters": [{"column": "t", "op": "lt", "value": 60}]}), encoding="utf-8")
    model_dir = tmp_path / "model"
    extra = ("--cv", "timeseries", "--time-column", "t", "--filters", str(filters))
    assert main(_args(data_path, model_dir, *extra)) == EXIT_OK
    return model_dir / "model.pkl"


def test_predict_with_scores_and_shap_saves_what_the_api_returns(
    data_path: Path, first_half_model: Path, tmp_path: Path
) -> None:
    """``predict`` with the thresholds and ``--shap`` saves the columns and values of ``mloom.predict``."""
    output = tmp_path / "predictions.parquet"
    command = ["predict", str(first_half_model), str(data_path), "--output", str(output), "--shap"]
    thresholds = ["--cm-actual-threshold", "0", "--cm-predicted-threshold", "0.5", "--cm-positive", "below"]
    assert main([*command, *thresholds]) == EXIT_OK

    expected = predict(
        first_half_model, data_path, shap=True, cm_actual_threshold=0.0, cm_predicted_threshold=0.5, cm_positive="below"
    )
    saved = pl.read_parquet(output)
    assert saved.equals(expected)
    assert saved["rmse"].is_not_null().all()
    assert saved["cm"].is_not_null().any()
    assert "shap_base" in saved.columns


@pytest.mark.parametrize(
    ("task", "extra", "message"),
    [
        ("regression", ["--cm-actual-threshold", "0", "--cm-positive", "above"], "cm_predicted_threshold"),
        ("classification", ["--positive-class", "bad"], "'bad' is not a class of the model"),
        ("regression", ["--positive-class", "pos"], "positive_class is for classification"),
    ],
    ids=["some thresholds", "unknown class", "class for regression"],
)
def test_invalid_confusion_arguments_exit_with_2(
    data_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], task: str, extra: list[str], message: str
) -> None:
    """Arguments of the confusion matrix that do not fit the model exit with code 2 and save nothing."""
    model_dir = tmp_path / "model"
    target = ["--task", task, "--target", "y" if task == "regression" else "c", "--exclude-columns", "y", "c"]
    assert main([*_args(data_path, model_dir), *target]) == EXIT_OK
    capsys.readouterr()
    output = tmp_path / "predictions.parquet"
    command = ["predict", str(model_dir / "model.pkl"), str(data_path), "--output", str(output)]
    assert main([*command, *extra]) == EXIT_INPUT
    assert message in capsys.readouterr().err
    assert not output.exists()
