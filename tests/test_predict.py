"""Tests of ``mloom.predict`` and of reading ``model.pkl`` without mloom."""

import pickle
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mloom
from mloom import Config, InputError, optimize_model, predict
from mloom.result import Result

FILTERS = ({"column": "mode", "op": "str_contains_any", "value": ["steady"]},)
TRAIN_FILTERS = ({"column": "lot", "op": "ne", "value": "L3"},)


@pytest.fixture
def data(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a ``mode`` (``start`` every 10th row) to the shared frame, and copies of 9 rows without a target.

    The copies have the same features, time and lot as rows that are trained on.
    """
    with_mode = frame.with_columns(mode=pl.when(pl.col("t") % 10 == 0).then(pl.lit("start")).otherwise(pl.lit("steady")))
    copies = with_mode.filter(pl.col("t").is_between(1, 9)).with_columns(y=pl.lit(None, dtype=pl.Float64))
    return pl.concat([with_mode, copies])


@pytest.fixture
def trained(data: pl.DataFrame, make_config: Callable[..., Config]) -> Result:
    """Fit a ridge regression on ``data`` with ``filters``, ``train_filters`` and a time series split."""
    config = make_config(cv="timeseries", time_column="t", filters=FILTERS, train_filters=TRAIN_FILTERS)
    return optimize_model(data, config)


def _passed(data: pl.DataFrame) -> pl.Series:
    return data.select(pl.col("mode") == "steady").to_series()


def test_regression_output_has_the_rows_and_columns_of_the_spec(data: pl.DataFrame, trained: Result) -> None:
    """One row per input row, in order; rows outside ``filters`` are NaN and the others are predicted."""
    output = predict(trained.model_bundle(), data, keep_columns=("t", "lot"))
    assert output.columns == ["t", "lot", "prediction", "passed_filter", "used_for_training"]
    assert output.height == data.height
    assert output["t"].to_list() == data["t"].to_list()
    passed = _passed(data)
    assert output["passed_filter"].to_list() == passed.to_list()
    assert output.filter(~passed)["prediction"].is_nan().all()
    np.testing.assert_allclose(
        output.filter(passed)["prediction"].to_numpy(), trained.pipeline.predict(data.filter(passed))
    )


def test_classification_output_has_null_predictions_and_nan_probabilities(
    data: pl.DataFrame, make_config: Callable[..., Config]
) -> None:
    """Classification adds ``proba_<class>``; rows outside ``filters`` have a null class and NaN probabilities."""
    result = optimize_model(data, make_config(task="classification", target="c", filters=FILTERS))
    output = predict(result.model_bundle(), data)
    assert output.columns == ["prediction", "passed_filter", "used_for_training", "proba_neg", "proba_pos"]
    passed = _passed(data)
    outside, inside = output.filter(~passed), output.filter(passed)
    assert outside["prediction"].is_null().all()
    assert outside["proba_neg"].is_nan().all()
    assert inside["prediction"].to_list() == result.pipeline.predict(data.filter(passed)).tolist()
    np.testing.assert_allclose(inside["proba_neg"] + inside["proba_pos"], 1.0)
    # Every row has the target c, so every row that passes the filters is trained on.
    assert output["used_for_training"].to_list() == passed.to_list()


def test_no_row_passing_the_filters_is_not_predicted(data: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """When no row passes ``filters``, every prediction is null or NaN (``predict`` is not called on 0 rows)."""
    result = optimize_model(data, make_config(task="classification", target="c"))
    bundle = {**result.model_bundle(), "filters": [{"column": "s0", "op": "gt", "value": 100}]}
    output = predict(bundle, data)
    assert output.height == data.height
    assert not output["passed_filter"].any()
    assert output["prediction"].is_null().all()
    assert output["prediction"].dtype == pl.String
    assert output["proba_pos"].is_nan().all()
    assert not output["used_for_training"].any()


def test_used_for_training_marks_exactly_the_rows_trained_on(data: pl.DataFrame, trained: Result) -> None:
    """Rows without a target, rows outside ``train_filters`` and rows outside ``filters`` are false."""
    output = predict(trained.model_bundle(), data)
    trained_on = _passed(data) & (data["lot"] != "L3") & data["y"].is_not_null()
    assert output["used_for_training"].to_list() == trained_on.to_list()
    assert trained.data.n_rows_used == trained_on.sum()
    # The copies without a target have the features, time and lot of rows trained on, and are false.
    originals = output.head(data.height - 9).filter(data.head(data.height - 9)["t"].is_between(1, 9))
    assert originals["used_for_training"].sum() == 8  # t = 1..9 but the lot L3 (t = 3)
    assert not output.tail(9)["used_for_training"].any()


def test_changed_value_is_not_used_for_training(data: pl.DataFrame, trained: Result) -> None:
    """A row whose value was changed after training does not match its hash."""
    changed = data.with_columns(s5=pl.col("s5") + 1e-9)
    assert not predict(trained.model_bundle(), changed)["used_for_training"].any()


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.drop("y"),
        lambda data: data.drop("lot"),
        lambda data: data.with_columns(lot=pl.col("t") % 12),
    ],
    ids=["no target", "no train_filters column", "train_filters value of another type"],
)
def test_used_for_training_is_null_when_it_cannot_be_told(
    data: pl.DataFrame, trained: Result, change: Callable[[pl.DataFrame], pl.DataFrame]
) -> None:
    """Without the target, or without a usable ``train_filters`` column, it is null and the rows are predicted."""
    output = predict(trained.model_bundle(), change(data))
    assert output["used_for_training"].is_null().all()
    assert output["used_for_training"].dtype == pl.Boolean
    assert output.filter(pl.col("passed_filter"))["prediction"].is_not_nan().all()


@pytest.mark.parametrize(
    ("change", "keep_columns", "message"),
    [
        (lambda data: data.drop("s0"), (), r"candidate column \['s0'\] not in the data"),
        (lambda data: data.drop("mode"), (), r"filters\[0\]: column 'mode' is not in the data"),
        (lambda data: data.with_columns(mode=pl.lit(1)), (), r"filters\[0\]: .* needs a string column"),
        (lambda data: data.drop("t"), (), r"column the training rows are matched by \['t'\]"),
        (lambda data: data, ("nope",), r"column of keep_columns \['nope'\] not in the data"),
        (lambda data: data.with_columns(prediction=pl.lit(0)), ("prediction",), "names of output columns"),
        (lambda data: data, ("t", "t"), "more than once"),
    ],
)
def test_invalid_input_raises(
    data: pl.DataFrame,
    trained: Result,
    change: Callable[[pl.DataFrame], pl.DataFrame],
    keep_columns: tuple[str, ...],
    message: str,
) -> None:
    """A column the model needs, or a column of ``keep_columns``, that is missing or clashes raises ``InputError``."""
    with pytest.raises(InputError, match=message):
        predict(trained.model_bundle(), change(data), keep_columns=keep_columns)


def test_model_path_and_dict_agree(data: pl.DataFrame, trained: Result, tmp_path: Path) -> None:
    """``model`` may be the path of ``model.pkl`` or its dict, and ``data`` a Parquet path or a DataFrame."""
    trained.save(tmp_path)
    data.write_parquet(tmp_path / "data.parquet")
    from_paths = predict(tmp_path / "model.pkl", tmp_path / "data.parquet")
    assert from_paths.equals(predict(trained.model_bundle(), data))
    assert mloom.predict is predict


@pytest.mark.parametrize(("content", "message"), [(None, "model file not found"), ([1], "must be a dict")])
def test_invalid_model_raises(data: pl.DataFrame, tmp_path: Path, content: object, message: str) -> None:
    """A missing file, or a pickle that is not a model of mloom, raises ``InputError``."""
    path = tmp_path / "model.pkl"
    if content is not None:
        path.write_bytes(pickle.dumps(content))
    with pytest.raises(InputError, match=message):
        predict(path, data)


_WITHOUT_MLOOM = """
import pickle
import sys

sys.modules["mloom"] = None  # any import of mloom fails
import numpy as np
import polars as pl

with open(sys.argv[1], "rb") as file:
    bundle = pickle.load(file)
np.save(sys.argv[3], bundle["pipeline"].predict(pl.read_parquet(sys.argv[2])))
"""


@pytest.mark.parametrize(("model", "task", "target"), [("ridge", "regression", "y"), ("random_forest", "classification", "c")])
def test_model_pkl_is_read_and_predicts_without_mloom(
    frame: pl.DataFrame,
    make_config: Callable[..., Config],
    tmp_path: Path,
    model: str,
    task: str,
    target: str,
) -> None:
    """A Python that cannot import mloom reads ``model.pkl`` and predicts what mloom predicts."""
    result = optimize_model(frame, make_config(model=model, task=task, target=target))
    result.save(tmp_path)
    frame.write_parquet(tmp_path / "data.parquet")
    assert b"mloom" not in (tmp_path / "model.pkl").read_bytes()

    out = tmp_path / "predictions.npy"
    paths = [str(tmp_path / "model.pkl"), str(tmp_path / "data.parquet"), str(out)]
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _WITHOUT_MLOOM, *paths], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(np.load(out, allow_pickle=True), result.pipeline.predict(frame))
