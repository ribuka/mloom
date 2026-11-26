"""Building the scikit-learn Pipeline that is tuned, fitted, and pickled.

The steps are those of "前処理" in ``docs/spec.md``: column selection (a
``ColumnTransformer``, only in the saved pipeline, that also encodes the
categorical candidates as integers) → preprocessing (+inf and -inf to NaN,
clipping, imputation, and standardization for some models, of the numeric
candidates only) → selection of the kept columns (a ``ColumnTransformer`` by
position) → the estimator. The numeric candidates come before the
categorical ones in every matrix.

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
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

from mloom.categorical import fold_encoder
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
    if not columns:  # polars selects no row with no column
        return np.empty((frame.height, 0))
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


def fit_preprocess(
    preprocessing: Preprocessing,
    x: np.ndarray,
    n_numeric: int,
    categorical: Any = "passthrough",  # noqa: ANN401 - a scikit-learn transformer
) -> ColumnTransformer:
    """Compute the clipping bounds and fit the imputation and the standardization.

    Parameters
    ----------
    preprocessing : Preprocessing
        The preprocessing of the run.
    x : np.ndarray
        Feature matrix of the rows the statistics come from: the labeled
        training rows and the unlabeled rows on the training side. Its first
        ``n_numeric`` columns are the numeric candidates, and the others the
        codes of the categorical candidates (``mloom.categorical.encode``).
    n_numeric : int
        Number of numeric candidate columns.
    categorical : Any
        What is done to the codes: ``"passthrough"`` for the final model,
        whose codes are those of its encoder, and
        ``mloom.categorical.fold_encoder`` for a CV fold.

    Returns
    -------
    ColumnTransformer
        The fitted preprocessing, whose output has the columns of ``x`` in
        their order. Its ``"numeric"`` transformer is a ``Pipeline`` whose
        steps are ``infinite``, then ``clip``, ``impute`` and ``scale`` when
        they are used; ``"categorical"`` is ``categorical``.
    """
    numeric_x = x[:, :n_numeric]
    steps: list[tuple[str, Any]] = [("infinite", FunctionTransformer(np.nan_to_num, kw_args=_INFINITE_AS_NAN))]
    if preprocessing.clip.method != "none":
        lower, upper = clip_bounds(np.nan_to_num(numeric_x, nan=np.nan, posinf=np.nan, neginf=np.nan), preprocessing.clip)
        steps.append(("clip", FunctionTransformer(np.clip, kw_args={"a_min": lower, "a_max": upper})))
    if preprocessing.impute != "none":
        steps.append(("impute", _imputer(preprocessing)))
    if preprocessing.scale:
        steps.append(("scale", StandardScaler()))
    transformer = ColumnTransformer(
        [
            ("numeric", Pipeline(steps), list(range(n_numeric))),
            ("categorical", categorical, list(range(n_numeric, x.shape[1]))),
        ],
        remainder="drop",
    )
    return transformer.fit(x)


def fit_fold_preprocess(preprocessing: Preprocessing, x: np.ndarray, n_numeric: int) -> ColumnTransformer:
    """Fit the preprocessing of a CV fold, which encodes the codes again with the categories of the fold.

    Parameters
    ----------
    preprocessing : Preprocessing
        The preprocessing of the run.
    x : np.ndarray
        Feature matrix of the statistic rows of the fold (``fit_preprocess``).
    n_numeric : int
        Number of numeric candidate columns.

    Returns
    -------
    ColumnTransformer
        The fitted preprocessing (``fit_preprocess``); its ``"categorical"``
        transformer is a ``mloom.categorical.fold_encoder``.
    """
    return fit_preprocess(preprocessing, x, n_numeric, fold_encoder(x[:, n_numeric:]))


def numeric_preprocess(preprocess: ColumnTransformer) -> Pipeline:
    """Take the preprocessing of the numeric candidate columns.

    Parameters
    ----------
    preprocess : ColumnTransformer
        A preprocessing from ``fit_preprocess``.

    Returns
    -------
    Pipeline
        Its ``"numeric"`` transformer, whose steps work on the numeric
        candidate columns only (``fit_preprocess``).
    """
    return preprocess.named_transformers_["numeric"]


def n_numeric_columns(preprocess: ColumnTransformer) -> int:
    """Tell the number of numeric candidate columns of a preprocessing.

    Parameters
    ----------
    preprocess : ColumnTransformer
        A preprocessing from ``fit_preprocess``.

    Returns
    -------
    int
        Number of numeric candidate columns, which come first in its input
        and output.
    """
    return len(preprocess.transformers[0][2])


def _imputer(preprocessing: Preprocessing) -> Any:  # noqa: ANN401 - a scikit-learn imputer
    # keep_empty_features keeps the number of columns, so that the kept positions stay right.
    if preprocessing.impute == "knn":
        return KNNImputer(n_neighbors=preprocessing.n_neighbors, keep_empty_features=True)
    if preprocessing.impute == "iterative":
        return IterativeImputer(random_state=preprocessing.seed, keep_empty_features=True)
    if preprocessing.impute == "constant":
        return SimpleImputer(strategy="constant", fill_value=preprocessing.fill_value, keep_empty_features=True)
    return SimpleImputer(strategy=preprocessing.impute, keep_empty_features=True)


def fitted_clip_bounds(preprocess: ColumnTransformer) -> Bounds | None:
    """Read the clipping bounds of a fitted preprocessing.

    Parameters
    ----------
    preprocess : ColumnTransformer
        A preprocessing from ``fit_preprocess``.

    Returns
    -------
    Bounds | None
        Lower and upper bounds of the numeric candidate columns, or ``None``
        when it does not clip.
    """
    numeric = numeric_preprocess(preprocess)
    if "clip" not in numeric.named_steps:
        return None
    kw_args = numeric.named_steps["clip"].kw_args
    return kw_args["a_min"], kw_args["a_max"]


def select_columns(numeric: list[str], categorical: list[str], encoder: OrdinalEncoder | None) -> ColumnTransformer:
    """Build the first step of a saved pipeline, which takes the candidate columns from a table.

    Parameters
    ----------
    numeric : list[str]
        The numeric candidate columns.
    categorical : list[str]
        The categorical candidate columns.
    encoder : OrdinalEncoder | None
        The fitted encoder of ``categorical`` (``mloom.categorical.fit_encoder``);
        ``None`` when ``categorical`` is empty.

    Returns
    -------
    ColumnTransformer
        It takes a table (polars or pandas) with the candidate columns in any
        order and drops other columns. Its output is the numeric columns as
        they are (``"numeric"``), then the codes of the categorical columns
        (``"categorical"``, only when there are any).
    """
    transformers: list[tuple[str, Any, list[str]]] = [("numeric", "passthrough", numeric)]
    if encoder is not None:
        # Frozen, so that fitting the pipeline (which clones the transformers) keeps the fitted categories.
        transformers.append(("categorical", FrozenEstimator(encoder), categorical))
    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(
    config: Config,
    params: Params,
    preprocess: ColumnTransformer,
    keep: Sequence[int],
    columns: ColumnTransformer | None = None,
) -> Pipeline:
    """Build a pipeline whose preprocessing is already fitted.

    Parameters
    ----------
    config : Config
        The config of the run.
    params : Params
        Hyperparameters of the model.
    preprocess : ColumnTransformer
        Fitted preprocessing from ``fit_preprocess``. It is frozen: fitting
        the pipeline fits only the model.
    keep : Sequence[int]
        Positions, among the candidate columns, of the columns the model
        uses. Those of the categorical candidates are given to the model as
        its categorical features.
    columns : ColumnTransformer | None
        The step that takes the candidate columns from a table
        (``select_columns``). When given, the pipeline takes a table;
        otherwise a matrix of the candidate columns (``to_matrix`` and
        ``mloom.categorical.encode``).

    Returns
    -------
    Pipeline
        The pipeline.
    """
    if "n_components" in params:
        # PLS cannot have more components than features.
        params = {**params, "n_components": min(params["n_components"], len(keep))}
    n_numeric = n_numeric_columns(preprocess)
    categorical = [position for position, i in enumerate(keep) if i >= n_numeric]
    steps: list[tuple[str, Any]] = [
        ("preprocess", FrozenEstimator(preprocess)),
        ("select", ColumnTransformer([("keep", "passthrough", [int(i) for i in keep])], remainder="drop")),
        ("model", config.model_spec.build(config.task, params, config.seed, categorical)),
    ]
    if columns is not None:
        steps.insert(0, ("columns", columns))
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
        The feature candidates, in the order of the matrix: the numeric
        ones, then the categorical ones.
    """
    return [name for _, _, names in pipeline.named_steps["columns"].transformers for name in names]


def selected_columns(pipeline: Pipeline) -> list[str]:
    """Tell the columns the estimator of a saved pipeline takes.

    Parameters
    ----------
    pipeline : Pipeline
        A pipeline built with ``columns``.

    Returns
    -------
    list[str]
        The kept candidate columns, in the order of the estimator's input.
    """
    candidates = candidate_columns(pipeline)
    return [candidates[i] for i in pipeline.named_steps["select"].transformers[0][2]]


def estimator_input(pipeline: Pipeline, frame: pl.DataFrame) -> np.ndarray:
    """Turn a table into the input of the estimator of a saved pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted pipeline built with ``columns``.
    frame : pl.DataFrame
        Data that has every candidate column.

    Returns
    -------
    np.ndarray
        The values after the preprocessing and the selection of the kept
        columns, of shape ``(frame.height, len(selected_columns(pipeline)))``.
    """
    return np.asarray(pipeline[:-1].transform(frame), dtype=np.float64)
