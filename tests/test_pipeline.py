"""Tests of the preprocessing: +-inf as missing, the clipping bounds, and the choice of the imputation."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from mloom.categorical import encode, fit_encoder
from mloom.clip import describe_clip
from mloom.config import Config
from mloom.errors import InputError
from mloom.models import MODELS
from mloom.pipeline import (
    Preprocessing,
    build_pipeline,
    fit_fold_preprocess,
    fit_preprocess,
    fitted_clip_bounds,
    numeric_preprocess,
    plan_preprocessing,
    select_columns,
    to_matrix,
)
from mloom.settings import ClipSettings, load_settings

INF = float("inf")
NAN = float("nan")
COLUMNS = ["a", "constant", "empty", "d", "e"]
# The last row holds +-inf or NaN, which no bound may use.
X = np.array(
    [
        [1.0, 5.0, NAN, 0.0, 0.0],
        [2.0, 5.0, INF, 0.0, 0.0],
        [3.0, 5.0, NAN, 0.0, 0.0],
        [4.0, 5.0, -INF, 1.0, 0.0],
        [100.0, 5.0, NAN, 2.0, 10.0],
        [INF, NAN, NAN, -INF, NAN],
    ]
)
UNCLIPPED = (-INF, INF)


def _clip(method: str) -> ClipSettings:
    return ClipSettings(method=method, lower_quantile=0.25, upper_quantile=0.75, iqr_k=1.5, mad_k=3.0, sigma_k=2.0)


def _preprocessing(clip: ClipSettings, impute: str = "none") -> Preprocessing:
    return Preprocessing(clip=clip, impute=impute, fill_value=0.0, n_neighbors=2, scale=False, seed=0)


# Hand computed from the finite values; "constant" and "empty" have no spread or no value.
MAD_A = 3.0 * 1.4826 * 1.0
SIGMA_A = 2.0 * np.sqrt(1522.0)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("quantile", [(2.0, 4.0), UNCLIPPED, UNCLIPPED, (0.0, 1.0), UNCLIPPED]),
        ("iqr", [(-1.0, 7.0), UNCLIPPED, UNCLIPPED, (-1.5, 2.5), UNCLIPPED]),
        ("mad", [(3.0 - MAD_A, 3.0 + MAD_A), UNCLIPPED, UNCLIPPED, UNCLIPPED, UNCLIPPED]),
        ("sigma", [(22.0 - SIGMA_A, 22.0 + SIGMA_A), UNCLIPPED, UNCLIPPED, (-1.0, 2.2), (-6.0, 10.0)]),
    ],
)
def test_clip_bounds_match_the_hand_computed_ones(method: str, expected: list[tuple[float, float]]) -> None:
    """Each method bounds the finite values as the spec says; a column without spread or values is not clipped."""
    preprocess = fit_preprocess(_preprocessing(_clip(method)), X, X.shape[1])
    bounds = fitted_clip_bounds(preprocess)
    assert bounds is not None
    np.testing.assert_allclose(np.column_stack(bounds), np.array(expected))

    out = preprocess.transform(X)
    # +-inf became NaN before the clipping, and NaN stays NaN.
    assert np.isnan(out[5]).all()
    np.testing.assert_array_equal(np.isnan(out), np.isnan(X) | np.isinf(X))
    lower, upper = bounds
    finite = ~np.isnan(out)
    assert ((out >= lower) & (out <= upper))[finite].all()
    record = describe_clip(method, bounds, COLUMNS)
    clipped = [name for name, bound in zip(COLUMNS, expected, strict=True) if bound != UNCLIPPED]
    assert list(record["bounds"]) == clipped
    assert record["unclipped_columns"] == [name for name in COLUMNS if name not in clipped]


def test_no_clip_step_without_a_method() -> None:
    """``none`` adds no clipping step; its record has no bounds."""
    preprocess = fit_preprocess(_preprocessing(_clip("none")), X, X.shape[1])
    assert list(numeric_preprocess(preprocess).named_steps) == ["infinite"]
    assert fitted_clip_bounds(preprocess) is None
    assert describe_clip("none", None, COLUMNS) == {"method": "none", "bounds": {}, "unclipped_columns": []}


@pytest.mark.parametrize(
    ("impute", "expected"),
    [
        ("median", [3.0, 5.0, 0.0, 0.0, 0.0]),
        ("mean", [22.0, 5.0, 0.0, 0.6, 2.0]),
        ("constant", [0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_imputation_fills_the_infinite_values_too(impute: str, expected: list[float]) -> None:
    """+-inf is filled like a missing value, from the finite values only; a column without values keeps 0."""
    preprocess = fit_preprocess(_preprocessing(_clip("none"), impute), X[:5], X.shape[1])
    out = preprocess.transform(np.array([[INF, NAN, -INF, NAN, INF]]))
    np.testing.assert_allclose(out[0], expected)


@pytest.mark.parametrize("impute", ["most_frequent", "knn", "iterative"])
def test_other_imputations_leave_no_missing_value(impute: str) -> None:
    """``most_frequent``, ``knn`` and ``iterative`` fill every column, keeping their number."""
    preprocess = fit_preprocess(_preprocessing(_clip("iqr"), impute), X, X.shape[1])
    out = preprocess.transform(X)
    assert out.shape == X.shape
    assert not np.isnan(out).any()


@pytest.mark.parametrize(
    ("model", "method", "expected"),
    [
        ("ridge", "auto", "median"),
        ("pls", "auto", "median"),
        ("random_forest", "auto", "none"),
        ("lightgbm", "auto", "none"),
        ("lightgbm", "knn", "knn"),
        ("ridge", "iterative", "iterative"),
    ],
)
def test_auto_imputation_depends_on_the_model(
    make_config: Callable[..., Config], tmp_path: Path, model: str, method: str, expected: str
) -> None:
    """``auto`` is ``median`` for the models that cannot handle missing values and ``none`` for the others."""
    settings = tmp_path / "mloom.toml"
    settings.write_text(f"[impute]\nmethod = '{method}'\n", encoding="utf-8")
    preprocessing = plan_preprocessing(make_config(model=model), load_settings(settings))
    assert preprocessing.impute == expected


@pytest.mark.parametrize("model", ["ridge", "pls"])
def test_no_imputation_needs_a_model_that_handles_missing_values(
    make_config: Callable[..., Config], tmp_path: Path, model: str
) -> None:
    """``none`` with ``ridge`` or ``pls`` raises ``InputError``."""
    settings = tmp_path / "mloom.toml"
    settings.write_text("[impute]\nmethod = 'none'\n", encoding="utf-8")
    with pytest.raises(InputError, match=f"'none' needs a model that handles missing values; '{model}' does not"):
        plan_preprocessing(make_config(model=model), load_settings(settings))


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_lightgbm_estimator_has_the_fixed_arguments(task: str) -> None:
    """LightGBM runs on 1 thread with the seed, samples rows at every iteration and logs nothing."""
    params = MODELS["lightgbm"].builders[task]({"num_leaves": 16}, 7).get_params()
    assert {key: params[key] for key in ("n_jobs", "random_state", "subsample_freq", "verbose", "num_leaves")} == {
        "n_jobs": 1,
        "random_state": 7,
        "subsample_freq": 1,
        "verbose": -1,
        "num_leaves": 16,
    }


def _categorical_pipeline(frame: pl.DataFrame, config: Config) -> Pipeline:
    # x1 and grade, with grade encoded on the rows of frame, fitted on them.
    encoder = fit_encoder([frame], ["grade"])
    x = np.hstack([to_matrix(frame, ["x1"]), encode(encoder, frame, ["grade"])])
    preprocess = fit_preprocess(plan_preprocessing(config, load_settings(None)), x, 1)
    params = {"num_leaves": 8, "learning_rate": 0.1, "n_estimators": 50, "min_child_samples": 5, "subsample": 1.0,
              "colsample_bytree": 1.0, "reg_lambda": 1e-3, "cat_smooth": 1.0}  # fmt: skip
    pipeline = build_pipeline(config, params, preprocess, [0, 1], select_columns(["x1"], ["grade"], encoder))
    return pipeline.fit(frame, frame["y"].to_numpy())


@pytest.mark.parametrize(
    "dtype", [pl.String, pl.Categorical, pl.Enum(["A", "B", "C", "D"])], ids=["string", "categorical", "enum"]
)
def test_categorical_pipeline_fits_and_predicts_each_string_type(
    graded: pl.DataFrame, make_config: Callable[..., Config], dtype: pl.DataType
) -> None:
    """A pipeline with a categorical column fits and predicts with it as String, Categorical or Enum.

    The column is a categorical feature of LightGBM, and a model fitted on String predicts the same from Categorical.
    """
    config = make_config(model="lightgbm")
    frame = graded.with_columns(pl.col("grade").cast(dtype))
    pipeline = _categorical_pipeline(frame, config)
    assert pipeline[-1].get_params()["categorical_column"] == [1]
    assert np.isfinite(pipeline.predict(frame)).all()

    from_strings = _categorical_pipeline(graded, config)
    np.testing.assert_array_equal(from_strings.predict(frame), from_strings.predict(graded))


def test_fold_encoding_is_that_of_an_encoder_of_the_strings_of_the_fold(make_config: Callable[..., Config]) -> None:
    """Encoding the codes again with the categories of a fold gives the codes of an encoder fitted on its strings.

    A value not on the rows of the fold, and a column without a value there, become NaN.
    """
    frame = pl.DataFrame({"g": ["d", "a", "c", "b", None, "a"], "h": ["x", "y", "x", None, "y", "x"]})
    columns = ["g", "h"]
    codes = encode(fit_encoder([frame], columns), frame, columns)
    fold = frame.head(3).with_columns(h=pl.lit(None, dtype=pl.String))
    fold_codes = codes[:3].copy()
    fold_codes[:, 1] = np.nan
    preprocessing = plan_preprocessing(make_config(model="lightgbm"), load_settings(None))
    preprocess = fit_fold_preprocess(preprocessing, fold_codes, 0)

    expected = encode(fit_encoder([fold], ["g"]), frame, ["g"])
    out = preprocess.transform(codes)
    np.testing.assert_array_equal(out[:, 0], expected[:, 0])
    assert np.isnan(out[[3, 4], 0]).all()  # b is not on the rows of the fold, and null
    assert np.isnan(out[:, 1]).all()
