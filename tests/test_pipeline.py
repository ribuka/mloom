"""Tests of the preprocessing: +-inf as missing, the clipping bounds, and the choice of the imputation."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from mloom.clip import describe_clip
from mloom.config import Config
from mloom.errors import InputError
from mloom.models import MODELS
from mloom.pipeline import (
    Preprocessing,
    fit_preprocess,
    fitted_clip_bounds,
    plan_preprocessing,
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
    preprocess = fit_preprocess(_preprocessing(_clip(method)), X)
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
    preprocess = fit_preprocess(_preprocessing(_clip("none")), X)
    assert list(preprocess.named_steps) == ["infinite"]
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
    preprocess = fit_preprocess(_preprocessing(_clip("none"), impute), X[:5])
    out = preprocess.transform(np.array([[INF, NAN, -INF, NAN, INF]]))
    np.testing.assert_allclose(out[0], expected)


@pytest.mark.parametrize("impute", ["most_frequent", "knn", "iterative"])
def test_other_imputations_leave_no_missing_value(impute: str) -> None:
    """``most_frequent``, ``knn`` and ``iterative`` fill every column, keeping their number."""
    preprocess = fit_preprocess(_preprocessing(_clip("iqr"), impute), X)
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
