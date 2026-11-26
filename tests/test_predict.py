"""Tests of ``mloom.predict`` and of reading ``model.pkl`` without mloom."""

import pickle
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.base import is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import mloom
from mloom import Config, InputError, optimize_model, predict
from mloom.pipeline import estimator_input
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
    assert output.columns == ["t", "lot", "prediction", "passed_filter", "used_for_training", "error", "rmse", "r2"]
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


@pytest.mark.parametrize("model", ["ridge", "hist_gradient_boosting"])
def test_rows_with_infinite_features_are_predicted(
    data: pl.DataFrame, make_config: Callable[..., Config], model: str
) -> None:
    """A row with +inf or -inf in a feature gets a finite prediction, as with a missing value."""
    result = optimize_model(data, make_config(model=model))
    changed = data.with_columns(
        s0=pl.when(pl.col("t") == 1).then(float("inf")).otherwise(pl.col("s0")),
        s1=pl.when(pl.col("t") == 2).then(float("-inf")).otherwise(pl.col("s1")),
    )
    output = predict(result.model_bundle(), changed)
    rows = changed["t"].is_in([1, 2])
    assert rows.sum() == 4  # the rows and their copies without a target
    assert output.filter(rows)["prediction"].is_finite().all()


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
sys.modules["shap"] = None  # and of shap
import numpy as np
import polars as pl

with open(sys.argv[1], "rb") as file:
    bundle = pickle.load(file)
np.save(sys.argv[3], bundle["pipeline"].predict(pl.read_parquet(sys.argv[2])))
"""


@pytest.mark.parametrize(
    ("model", "task", "target", "settings"),
    [
        ("ridge", "regression", "y", ""),
        ("random_forest", "classification", "c", ""),
        ("ridge", "regression", "y", "[clip]\nmethod = 'iqr'\n[impute]\nmethod = 'knn'\n"),
        ("hist_gradient_boosting", "classification", "c", "[clip]\nmethod = 'iqr'\n[impute]\nmethod = 'iterative'\n"),
    ],
)
def test_model_pkl_is_read_and_predicts_without_mloom(
    frame: pl.DataFrame,
    make_config: Callable[..., Config],
    tmp_path: Path,
    model: str,
    task: str,
    target: str,
    settings: str,
) -> None:
    """A Python that cannot import mloom reads ``model.pkl`` and predicts what mloom predicts.

    It holds with the clipping and the imputations that use the other columns, and with +-inf and NaN.
    """
    data = frame.with_columns(
        s0=pl.when(pl.col("t") % 9 == 0).then(float("-inf")).otherwise(pl.col("s0")),
        s1=pl.when(pl.col("t") % 8 == 0).then(None).otherwise(pl.col("s1")),
    )
    settings_path = tmp_path / "mloom.toml"
    settings_path.write_text(settings, encoding="utf-8")
    result = optimize_model(data, make_config(model=model, task=task, target=target, settings=settings_path))
    result.save(tmp_path)
    data.write_parquet(tmp_path / "data.parquet")
    assert b"mloom" not in (tmp_path / "model.pkl").read_bytes()

    out = tmp_path / "predictions.npy"
    paths = [str(tmp_path / "model.pkl"), str(tmp_path / "data.parquet"), str(out)]
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _WITHOUT_MLOOM, *paths], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(np.load(out, allow_pickle=True), result.pipeline.predict(data))


_SHAP_LESS = """
import sys

import mloom

mloom.predict(sys.argv[1], sys.argv[2], cm_actual_threshold=0, cm_predicted_threshold=0, cm_positive="above")
assert "shap" not in sys.modules
"""


def test_shap_is_not_imported_without_shap(data: pl.DataFrame, trained: Result, tmp_path: Path) -> None:
    """Predicting without ``shap`` (with the scores and the confusion matrix) does not import shap."""
    trained.save(tmp_path)
    data.write_parquet(tmp_path / "data.parquet")
    paths = [str(tmp_path / "model.pkl"), str(tmp_path / "data.parquet")]
    completed = subprocess.run([sys.executable, "-c", _SHAP_LESS, *paths], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


# The scores and the confusion matrix (see "スコア" and "混同行列" in docs/spec.md).

FIRST_HALF = ({"column": "t", "op": "lt", "value": 60},)


def _three_classes() -> pl.Expr:
    return (
        pl.when(pl.col("y") > 1)
        .then(pl.lit("high"))
        .when(pl.col("y") > -1)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("low"))
    )


@pytest.fixture
def scored_data(data: pl.DataFrame) -> pl.DataFrame:
    """Put a NaN, an infinite and a missing target into ``data`` after the first half."""
    return data.with_columns(
        y=pl.when(pl.col("t") == 71)
        .then(float("nan"))
        .when(pl.col("t") == 72)
        .then(float("inf"))
        .when(pl.col("t") == 73)
        .then(None)
        .otherwise(pl.col("y"))
    )


@pytest.fixture
def first_half(scored_data: pl.DataFrame, make_config: Callable[..., Config]) -> Result:
    """Fit a ridge regression on the first half of the time (``train_filters``), with ``filters``."""
    config = make_config(cv="timeseries", time_column="t", filters=FILTERS, train_filters=FIRST_HALF)
    return optimize_model(scored_data, config)


def test_regression_scores_are_those_of_the_rows_after_training(scored_data: pl.DataFrame, first_half: Result) -> None:
    """``error`` is on the rows after ``last_time`` with a target and a prediction; ``rmse`` and ``r2`` are theirs."""
    bundle = first_half.model_bundle()
    last_time = bundle["training_rows"]["last_time"]
    trained_on = scored_data.filter((pl.col("t") < 60) & (pl.col("mode") == "steady") & pl.col("y").is_not_null())
    assert last_time == trained_on["t"].max() == 59
    output = predict(bundle, scored_data)

    expected = scored_data.select(
        (pl.col("t") > last_time) & pl.col("y").is_finite().fill_null(value=False) & (pl.col("mode") == "steady")
    ).to_series()
    assert output["error"].is_not_null().to_list() == expected.to_list()
    # Not scored: up to last_time, a NaN, infinite or missing target, and the rows outside filters.
    assert not expected.filter(scored_data["t"].is_in([59, 70, 71, 72, 73])).any()
    assert expected.sum() == 60 - 6 - 3  # t = 60..119 but 6 rows outside filters and 3 targets
    actual = scored_data.filter(expected)["y"].to_numpy()
    prediction = output.filter(expected)["prediction"].to_numpy()
    np.testing.assert_allclose(output.filter(expected)["error"].to_numpy(), prediction - actual, rtol=0, atol=1e-9)
    assert output["rmse"].unique().to_list() == pytest.approx([root_mean_squared_error(actual, prediction)], abs=1e-9)
    assert output["r2"].unique().to_list() == pytest.approx([r2_score(actual, prediction)], abs=1e-9)


@pytest.mark.parametrize(
    ("time_column", "make_data"),
    [(None, lambda data: data), ("t", lambda data: data.drop("y"))],
    ids=["no time_column", "no target"],
)
def test_scores_are_null_when_they_cannot_be_told(
    scored_data: pl.DataFrame,
    make_config: Callable[..., Config],
    time_column: str | None,
    make_data: Callable[[pl.DataFrame], pl.DataFrame],
) -> None:
    """A model without ``time_column``, or data without the target, gives null scores and confusion matrix."""
    cv = "kfold" if time_column is None else "timeseries"
    result = optimize_model(scored_data, make_config(cv=cv, time_column=time_column, train_filters=FIRST_HALF))
    assert (result.model_bundle()["training_rows"]["last_time"] is None) == (time_column is None)
    output = predict(
        result.model_bundle(),
        make_data(scored_data),
        cm_actual_threshold=0.0,
        cm_predicted_threshold=0.0,
        cm_positive="above",
    )
    scores = ["error", "rmse", "r2", "cm", "accuracy", "precision", "recall"]
    assert output.columns[-7:] == scores
    assert all(output[name].is_null().all() for name in scores)
    assert output["cm"].dtype == pl.String
    assert output["prediction"].is_not_nan().all()


def _identity_bundle(x: list[float], y: list[float], last_time: int) -> tuple[dict[str, Any], pl.DataFrame]:
    """Build a regression model that predicts ``x`` exactly, and the table of ``x``, ``y`` and ``t`` = 1, 2, ..."""
    pipeline = Pipeline(
        [
            ("columns", ColumnTransformer([("keep", "passthrough", ["x"])], remainder="drop")),
            ("preprocess", FunctionTransformer()),
            ("select", ColumnTransformer([("keep", "passthrough", [0])], remainder="drop")),
            ("model", KNeighborsRegressor(n_neighbors=1)),
        ]
    )
    pipeline.fit(pl.DataFrame({"x": x}), x)
    training_rows = {
        "columns": ["x", "y", "t"],
        "target": "y",
        "time_column": "t",
        "group_column": None,
        "hashes": np.array([], dtype=np.uint64),
        "last_time": last_time,
    }
    bundle = {
        "pipeline": pipeline,
        "filters": [],
        "train_filters": [],
        "training_rows": training_rows,
        "shap_background": np.zeros(1),
    }
    return bundle, pl.DataFrame({"x": x, "y": y, "t": range(1, len(x) + 1)})


# Prediction and target of each row: (3, 2), (2, 2), (3, 1), (1, 0), (2, 1); the times are 1 to 5.
_CM_X = [3.0, 2.0, 3.0, 1.0, 2.0]
_CM_Y = [2.0, 2.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("positive", "predicted_threshold", "last_time", "cm", "rates"),
    [
        # Above 1 (target) and 2 (prediction); a value equal to a threshold is negative.
        ("above", 2.0, 0, ["tp", "fn", "fp", "tn", "tn"], (3 / 5, 1 / 2, 1 / 2)),
        # Below 1 and 2: only (1, 0) is positive at all.
        ("below", 2.0, 0, ["tn", "tn", "tn", "tp", "tn"], (1.0, 1.0, 1.0)),
        # Below 1 and 0: nothing is predicted positive, so precision has no denominator.
        ("below", 0.0, 0, ["tn", "tn", "tn", "fn", "tn"], (4 / 5, None, 0.0)),
        # Above, with the first 2 rows up to last_time: no positive target is left, so recall has none.
        ("above", 2.0, 2, [None, None, "fp", "tn", "tn"], (2 / 3, 0.0, None)),
        # No row after last_time: every rate has no denominator.
        ("above", 2.0, 5, [None] * 5, (None, None, None)),
    ],
)
def test_regression_confusion_matrix_counts_by_hand(
    positive: str, predicted_threshold: float, last_time: int, cm: list[str | None], rates: tuple[float | None, ...]
) -> None:
    """``tp``, ``fp``, ``fn`` and ``tn`` follow the thresholds and the side; the rates come from their counts."""
    bundle, table = _identity_bundle(_CM_X, _CM_Y, last_time)
    output = predict(
        bundle, table, cm_actual_threshold=1.0, cm_predicted_threshold=predicted_threshold, cm_positive=positive
    )
    assert output["prediction"].to_list() == _CM_X
    assert output["cm"].to_list() == cm
    for name, rate in zip(("accuracy", "precision", "recall"), rates, strict=True):
        assert output[name].to_list() == pytest.approx([rate] * 5)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classification_confusion_matrix_follows_the_probability_threshold(
    scored_data: pl.DataFrame, make_config: Callable[..., Config], n_classes: int
) -> None:
    """A row is predicted positive when the probability of the positive class is above the threshold.

    With 3 classes, the positive class is told from all the others.
    """
    labels = _three_classes() if n_classes == 3 else pl.col("c")
    table = scored_data.with_columns(c=pl.when(pl.col("t") == 75).then(None).otherwise(labels))
    positive = "high" if n_classes == 3 else "pos"
    config = make_config(task="classification", target="c", cv="timeseries", time_column="t", train_filters=FIRST_HALF)
    bundle = optimize_model(table, config).model_bundle()
    assert len(bundle["pipeline"].classes_) == n_classes
    scored = table.select((pl.col("t") > 59) & pl.col("c").is_not_null()).to_series()
    actual = (table["c"] == positive).fill_null(value=False)

    cms = {}
    for threshold in (0.5, 0.9):
        output = predict(bundle, table, positive_class=positive, cm_predicted_threshold=threshold)
        assert output.columns[-4:] == ["cm", "accuracy", "precision", "recall"]
        assert "error" not in output.columns
        predicted = output[f"proba_{positive}"] > threshold
        expected = [
            (("t" if is_positive == predicted_positive else "f") + ("p" if predicted_positive else "n")) if is_scored else None
            for is_scored, is_positive, predicted_positive in zip(scored, actual, predicted, strict=True)
        ]
        assert output["cm"].to_list() == expected
        counts = {cell: expected.count(cell) for cell in ("tp", "fp", "fn", "tn")}
        assert output["accuracy"][0] == pytest.approx((counts["tp"] + counts["tn"]) / scored.sum())
        cms[threshold] = expected
    assert cms[0.5] != cms[0.9]
    # The default threshold is 0.5.
    assert predict(bundle, table, positive_class=positive)["cm"].to_list() == cms[0.5]


@pytest.mark.parametrize(
    ("task", "arguments", "message"),
    [
        ("regression", {"cm_actual_threshold": 0.0, "cm_positive": "above"}, r"\['cm_predicted_threshold'\] not given"),
        ("regression", {"cm_actual_threshold": 0, "cm_predicted_threshold": 0, "cm_positive": "over"}, "one of above"),
        ("regression", {"positive_class": "pos"}, "positive_class is for classification"),
        ("classification", {"positive_class": "bad"}, "'bad' is not a class of the model"),
        ("classification", {"positive_class": "pos", "cm_actual_threshold": 0.0}, "are for regression"),
        ("classification", {"positive_class": "pos", "cm_positive": "above"}, "are for regression"),
        ("classification", {"cm_predicted_threshold": 0.5}, "needs positive_class"),
    ],
)
def test_invalid_confusion_arguments_raise(
    data: pl.DataFrame, make_config: Callable[..., Config], task: str, arguments: dict[str, Any], message: str
) -> None:
    """Arguments of the confusion matrix that do not fit the model raise ``InputError``."""
    target = "y" if task == "regression" else "c"
    bundle = optimize_model(data, make_config(task=task, target=target)).model_bundle()
    with pytest.raises(InputError, match=message):
        predict(bundle, data, **arguments)


@pytest.mark.parametrize(
    ("keep", "arguments"),
    [
        ("error", {}),
        ("recall", {"cm_actual_threshold": 0.0, "cm_predicted_threshold": 0.0, "cm_positive": "above"}),
        ("shap_base", {"shap": True}),
        ("selected", {"shap": True}),
    ],
)
def test_keep_columns_may_not_have_the_names_of_the_scores(
    data: pl.DataFrame, trained: Result, keep: str, arguments: dict[str, Any]
) -> None:
    """A column of ``keep_columns`` named as a column of the scores, the matrix or SHAP raises ``InputError``."""
    name = f"shap_{trained.selected_columns[0]}" if keep == "selected" else keep
    with pytest.raises(InputError, match="names of output columns"):
        predict(trained.model_bundle(), data.with_columns(pl.lit(0).alias(name)), keep_columns=(name,), **arguments)


def _time_of_kind(kind: str) -> pl.Expr:
    if kind == "string":
        return pl.format("T{}", pl.col("t").cast(pl.String).str.zfill(3))
    if kind == "decimal":
        return pl.col("t").cast(pl.Decimal(10, 1))
    time = pl.lit("2026-01-01").str.to_datetime(time_unit="ns") + pl.duration(microseconds=pl.col("t"), nanoseconds=500)
    return time.dt.replace_time_zone("Asia/Tokyo") if kind.endswith("Tokyo") else time


@pytest.mark.parametrize("kind", ["datetime ns", "datetime ns Asia/Tokyo", "string", "decimal"])
def test_last_time_compares_the_rows_as_the_training_sorts_them(
    data: pl.DataFrame, make_config: Callable[..., Config], kind: str
) -> None:
    """The last row trained on is not scored, even with nanoseconds below the microseconds of ``last_time``."""
    table = data.with_columns(time=_time_of_kind(kind))
    config = make_config(
        cv="timeseries", time_column="time", train_filters=FIRST_HALF, exclude_columns=("y", "c", "t")
    )
    bundle = optimize_model(table, config).model_bundle()
    last_time = bundle["training_rows"]["last_time"]
    if kind == "string":
        assert last_time == "T059"
    elif kind == "decimal":
        assert last_time == Decimal("59.0")
    else:
        assert last_time.microsecond == 59
        assert last_time.tzinfo == (UTC if kind.endswith("Tokyo") else None)
    scored = ((table["t"] > 59) & table["y"].is_not_null()).to_list()
    assert predict(bundle, table)["error"].is_not_null().to_list() == scored
    if kind == "decimal":
        # A number of another type is compared too.
        as_float = table.with_columns(pl.col("time").cast(pl.Float64))
        assert predict(bundle, as_float)["error"].is_not_null().to_list() == scored


def test_selected_column_named_base_clashes_with_shap_base(data: pl.DataFrame, make_config: Callable[..., Config]) -> None:
    """A selected column ``base`` would give a second ``shap_base``: ``shap`` raises ``InputError``, the rest predicts."""
    table = data.rename({"s0": "base"})
    result = optimize_model(table, make_config())
    assert "base" in result.selected_columns
    with pytest.raises(InputError, match=r"more than one column named \['shap_base'\]"):
        predict(result.model_bundle(), table, shap=True)
    assert predict(result.model_bundle(), table)["prediction"].is_not_nan().all()


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.with_columns(pl.col("time").dt.replace_time_zone("UTC")),
        lambda data: data.with_columns(pl.col("time").cast(pl.String)),
    ],
    ids=["time zone added", "string"],
)
def test_time_column_of_another_kind_raises(
    data: pl.DataFrame, make_config: Callable[..., Config], change: Callable[[pl.DataFrame], pl.DataFrame]
) -> None:
    """With the target, a ``time_column`` that cannot be compared with ``last_time`` raises ``InputError``."""
    table = data.with_columns(time=pl.lit("2026-01-01").str.to_datetime() + pl.duration(seconds=pl.col("t")))
    config = make_config(cv="timeseries", time_column="time", exclude_columns=("y", "c", "t"))
    bundle = optimize_model(table, config).model_bundle()
    with pytest.raises(InputError, match="cannot be compared"):
        predict(bundle, change(table))


# The SHAP contributions (see "SHAP の寄与" in docs/spec.md).


def _model_output(estimator: Any, x: np.ndarray) -> np.ndarray:  # noqa: ANN401 - any estimator
    """Compute the output the contributions add up to: prediction, log odds, or probability (random forest)."""
    if not is_classifier(estimator):
        return np.asarray(estimator.predict(x)).reshape(len(x), 1)
    if isinstance(estimator, RandomForestClassifier):
        output = estimator.predict_proba(x)
        return output[:, 1:] if output.shape[1] == 2 else output
    return np.asarray(estimator.decision_function(x)).reshape(len(x), -1)


@pytest.mark.parametrize(
    ("model", "task", "n_classes"),
    [
        ("ridge", "regression", 0),
        ("pls", "regression", 0),
        ("random_forest", "regression", 0),
        ("hist_gradient_boosting", "regression", 0),
        ("ridge", "classification", 2),
        ("random_forest", "classification", 2),
        ("hist_gradient_boosting", "classification", 2),
        ("ridge", "classification", 3),
        ("random_forest", "classification", 3),
        ("hist_gradient_boosting", "classification", 3),
    ],
)
def test_shap_contributions_add_up_to_the_output(
    data: pl.DataFrame, make_config: Callable[..., Config], model: str, task: str, n_classes: int
) -> None:
    """``shap_base`` plus the contributions is the output of the model, for each selected column and class."""
    table = data.with_columns(c=_three_classes()) if n_classes == 3 else data
    target = "y" if task == "regression" else "c"
    result = optimize_model(table, make_config(model=model, task=task, target=target, filters=FILTERS))
    output = predict(result.model_bundle(), table, shap=True)

    selected = result.selected_columns
    classes = [None] if n_classes < 3 else [str(label) for label in result.pipeline.classes_]
    names = [
        name
        for label in classes
        for name in (
            "shap_base" if label is None else f"shap_base_{label}",
            *(f"shap_{column}" if label is None else f"shap_{label}_{column}" for column in selected),
        )
    ]
    assert [name for name in output.columns if name.startswith("shap_")] == names
    assert len(names) == len(classes) * (len(selected) + 1)

    passed = _passed(table)
    assert all(output.filter(~passed)[name].is_nan().all() for name in names)
    rows = output.filter(passed)
    expected = _model_output(result.pipeline[-1], estimator_input(result.pipeline, table.filter(passed)))
    width = len(selected) + 1
    for index in range(len(classes)):
        total = rows.select(names[index * width : (index + 1) * width]).to_numpy().sum(axis=1)
        np.testing.assert_allclose(total, expected[:, index], rtol=0, atol=1e-6)
