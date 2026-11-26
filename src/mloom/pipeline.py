"""Building the scikit-learn Pipeline that is tuned, fitted, and pickled.

The steps are: column selection (a ``ColumnTransformer``, only in the saved
pipeline) → preprocessing (imputation, and standardization for some models)
→ feature selection (``SelectKBest``) → the estimator.

The preprocessing uses no target, so it is fitted beforehand on the rows the
statistics come from (which may include unlabeled rows) and frozen. The
feature selection and the estimator are fitted with the pipeline, so CV fits
them on the training folds only.

The saved pipeline holds only parts of scikit-learn (and of the model's
library), never a class or function of mloom, so that ``model.pkl`` can be
read without mloom (see "``model.pkl``" in ``docs/spec.md``).
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mloom.config import Config
from mloom.models import Params


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


def fit_preprocess(config: Config, x: np.ndarray) -> Pipeline:
    """Fit the imputation (and the standardization, for some models).

    Parameters
    ----------
    config : Config
        The config of the run.
    x : np.ndarray
        Feature matrix of the rows the statistics come from: the labeled
        training rows and the unlabeled rows on the training side.

    Returns
    -------
    Pipeline
        The fitted preprocessing.
    """
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if config.model_spec.scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).fit(x)


def build_pipeline(
    config: Config, params: Params, k: int, preprocess: Pipeline, columns: list[str] | None = None
) -> Pipeline:
    """Build a pipeline whose preprocessing is already fitted.

    Parameters
    ----------
    config : Config
        The config of the run.
    params : Params
        Hyperparameters of the model.
    k : int
        Number of features ``SelectKBest`` keeps.
    preprocess : Pipeline
        Fitted preprocessing from ``fit_preprocess``. It is frozen: fitting
        the pipeline fits only the feature selection and the model.
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
        params = {**params, "n_components": min(params["n_components"], k)}
    score_func = f_regression if config.task == "regression" else f_classif
    steps: list[tuple[str, Any]] = [
        ("preprocess", FrozenEstimator(preprocess)),
        ("select", SelectKBest(score_func=score_func, k=k)),
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
