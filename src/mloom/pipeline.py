"""Building the scikit-learn Pipeline that is tuned, fitted, and pickled.

The steps are: column selection (``ColumnSelector``, only in the saved
pipeline) → preprocessing (imputation, and standardization for some models)
→ feature selection (``SelectKBest``) → the estimator.

The preprocessing uses no target, so it is fitted beforehand on the rows the
statistics come from (which may include unlabeled rows) and frozen. The
feature selection and the estimator are fitted with the pipeline, so CV fits
them on the training folds only.
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
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


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Select the feature columns of a table by name and return a matrix.

    It lets the saved pipeline take the whole table: columns it does not use
    are ignored.

    Parameters
    ----------
    columns : list[str]
        Feature columns, in the order the following steps expect.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> "ColumnSelector":  # noqa: ANN401, ARG002 - scikit-learn signature
        """Do nothing; the columns are fixed at construction.

        Parameters
        ----------
        X : Any
            Ignored.
        y : Any
            Ignored.

        Returns
        -------
        ColumnSelector
            This transformer.
        """
        return self

    def transform(self, X: Any) -> np.ndarray:  # noqa: ANN401 - scikit-learn signature
        """Select the columns.

        Parameters
        ----------
        X : Any
            A polars DataFrame, or a table indexable by a list of column names
            (such as a pandas DataFrame).

        Returns
        -------
        np.ndarray
            Float matrix of the selected columns.
        """
        if isinstance(X, pl.DataFrame):
            return to_matrix(X, self.columns)
        return np.asarray(X[self.columns], dtype=np.float64)

    def __sklearn_is_fitted__(self) -> bool:
        """Report that the transformer is always ready.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True


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
        ``ColumnSelector`` and takes a table; otherwise it takes a matrix.

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
        steps.insert(0, ("columns", ColumnSelector(columns)))
    return Pipeline(steps)
