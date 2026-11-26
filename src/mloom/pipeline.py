"""Building the scikit-learn Pipeline that is tuned, fitted, and pickled.

The steps are those of "前処理" in ``docs/spec.md``: column selection (a
``ColumnTransformer``, only in the saved pipeline) → preprocessing (+inf and
-inf to NaN, clipping, imputation, and standardization for some models) →
selection of the kept columns (a ``ColumnTransformer`` by position) → the
estimator.

The preprocessing uses no target, so it is fitted beforehand on the rows the
statistics come from (which may include unlabeled rows) and frozen. The
columns are ranked outside the pipeline (``mloom.selection``); the estimator
is fitted with the pipeline, so CV fits it on the training folds only.

The saved pipeline holds only parts of scikit-learn (and functions of numpy),
never a class or function of mloom, so that ``model.pkl`` can be read without
mloom (see "``model.pkl``" in ``docs/spec.md``).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.experimental import (
    enable_iterative_imputer,  # noqa: F401 - makes IterativeImputer importable
)
from sklearn.frozen import FrozenEstimator
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mloom.clip import Bounds, clip_bounds
from mloom.config import Config
from mloom.errors import InputError
from mloom.models import Params
from mloom.settings import ClipSettings, Settings

_INFINITE_AS_NAN = {"nan": np.nan, "posinf": np.nan, "neginf": np.nan}


@dataclass(frozen=True)
class Preprocessing:
    """How the features are preprocessed, from the config and the settings.

    Parameters
    ----------
    clip : ClipSettings
        ``[clip]``.
    impute : str
        The imputation, with ``"auto"`` resolved for the model.
    fill_value : float
        ``[impute] fill_value``.
    n_neighbors : int
        ``[impute] n_neighbors``.
    scale : bool
        Whether the features are standardized.
    seed : int
        Seed of ``IterativeImputer``.
    """

    clip: ClipSettings
    impute: str
    fill_value: float
    n_neighbors: int
    scale: bool
    seed: int


def as_float(name: str) -> pl.Expr:
    """Read a numeric or boolean column as a feature: Float64, NaN as missing.

    Parameters
    ----------
    name : str
        Column name.

    Returns
    -------
    pl.Expr
        The column as Float64 where NaN is null.
    """
    return pl.col(name).cast(pl.Float64).fill_nan(None)


def to_matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Turn the feature columns into a float matrix (missing values as NaN).

    +inf and -inf are kept: the pipeline turns them into NaN, as it does when
    it predicts.

    Parameters
    ----------
    frame : pl.DataFrame
        Data that has every column of ``columns``.
    columns : list[str]
        Feature columns, in the order of the matrix.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(frame.height, len(columns))``.
    """
    return frame.select([as_float(name) for name in columns]).to_numpy().astype(np.float64)


def plan_preprocessing(config: Config, settings: Settings) -> Preprocessing:
    """Decide the preprocessing of a run, before any data is fitted.

    Parameters
    ----------
    config : Config
        The config of the run.
    settings : Settings
        The settings of the run.

    Returns
    -------
    Preprocessing
        The preprocessing.

    Raises
    ------
    InputError
        When ``[impute] method`` is ``"none"`` and the model cannot handle
        missing values.
    """
    spec = config.model_spec
    impute = settings.impute.method
    if impute == "auto":
        impute = spec.default_impute
    if impute == "none" and not spec.handles_missing:
        raise InputError(f"[impute] method 'none' needs a model that handles missing values; {spec.name!r} does not")
    return Preprocessing(
        clip=settings.clip,
        impute=impute,
        fill_value=settings.impute.fill_value,
        n_neighbors=settings.impute.n_neighbors,
        scale=spec.scale,
        seed=config.seed,
    )


def fit_preprocess(preprocessing: Preprocessing, x: np.ndarray) -> Pipeline:
    """Compute the clipping bounds and fit the imputation and the standardization.

    Parameters
    ----------
    preprocessing : Preprocessing
        The preprocessing of the run.
    x : np.ndarray
        Feature matrix of the rows the statistics come from: the labeled
        training rows and the unlabeled rows on the training side.

    Returns
    -------
    Pipeline
        The fitted preprocessing. Its steps are ``infinite``, then ``clip``,
        ``impute`` and ``scale`` when they are used.
    """
    steps: list[tuple[str, Any]] = [("infinite", FunctionTransformer(np.nan_to_num, kw_args=_INFINITE_AS_NAN))]
    if preprocessing.clip.method != "none":
        lower, upper = clip_bounds(np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan), preprocessing.clip)
        steps.append(("clip", FunctionTransformer(np.clip, kw_args={"a_min": lower, "a_max": upper})))
    if preprocessing.impute != "none":
        steps.append(("impute", _imputer(preprocessing)))
    if preprocessing.scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).fit(x)


def _imputer(preprocessing: Preprocessing) -> Any:  # noqa: ANN401 - a scikit-learn imputer
    # keep_empty_features keeps the number of columns, so that the kept positions stay right.
    if preprocessing.impute == "knn":
        return KNNImputer(n_neighbors=preprocessing.n_neighbors, keep_empty_features=True)
    if preprocessing.impute == "iterative":
        return IterativeImputer(random_state=preprocessing.seed, keep_empty_features=True)
    if preprocessing.impute == "constant":
        return SimpleImputer(strategy="constant", fill_value=preprocessing.fill_value, keep_empty_features=True)
    return SimpleImputer(strategy=preprocessing.impute, keep_empty_features=True)


def fitted_clip_bounds(preprocess: Pipeline) -> Bounds | None:
    """Read the clipping bounds of a fitted preprocessing.

    Parameters
    ----------
    preprocess : Pipeline
        A preprocessing from ``fit_preprocess``.

    Returns
    -------
    Bounds | None
        Lower and upper bounds, or ``None`` when it does not clip.
    """
    if "clip" not in preprocess.named_steps:
        return None
    kw_args = preprocess.named_steps["clip"].kw_args
    return kw_args["a_min"], kw_args["a_max"]


def build_pipeline(
    config: Config, params: Params, preprocess: Pipeline, keep: Sequence[int], columns: list[str] | None = None
) -> Pipeline:
    """Build a pipeline whose preprocessing is already fitted.

    Parameters
    ----------
    config : Config
        The config of the run.
    params : Params
        Hyperparameters of the model.
    preprocess : Pipeline
        Fitted preprocessing from ``fit_preprocess``. It is frozen: fitting
        the pipeline fits only the model.
    keep : Sequence[int]
        Positions, among the candidate columns, of the columns the model uses.
    columns : list[str] | None
        Feature columns. When given, the pipeline starts with a
        ``ColumnTransformer`` that selects them by name, and takes a table
        (polars or pandas) with them in any order; other columns are dropped.
        Otherwise it takes a matrix.

    Returns
    -------
    Pipeline
        The pipeline.
    """
    if "n_components" in params:
        # PLS cannot have more components than features.
        params = {**params, "n_components": min(params["n_components"], len(keep))}
    steps: list[tuple[str, Any]] = [
        ("preprocess", FrozenEstimator(preprocess)),
        ("select", ColumnTransformer([("keep", "passthrough", [int(i) for i in keep])], remainder="drop")),
        ("model", config.model_spec.build(config.task, params, config.seed)),
    ]
    if columns is not None:
        steps.insert(0, ("columns", ColumnTransformer([("keep", "passthrough", columns)], remainder="drop")))
    return Pipeline(steps)


def candidate_columns(pipeline: Pipeline) -> list[str]:
    """Tell the columns a saved pipeline selects from its input table.

    Parameters
    ----------
    pipeline : Pipeline
        A pipeline built with ``columns``.

    Returns
    -------
    list[str]
        The feature candidates, in the order of the matrix.
    """
    return list(pipeline.named_steps["columns"].transformers[0][2])
