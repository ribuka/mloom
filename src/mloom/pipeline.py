"""Building the scikit-learn Pipeline that is tuned, fitted, and pickled.

The steps are: column selection (``ColumnSelector``, only in the saved
pipeline) → imputation → standardization (for some models) → feature
selection (``SelectKBest``) → the estimator. Every supervised step sits
inside the pipeline, so CV fits it on the training folds only.
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
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


def _model_steps(config: Config, params: Params, k: int) -> list[tuple[str, Any]]:
    """Build the unfitted steps that follow the column selection.

    Parameters
    ----------
    config : Config
        The config of the run.
    params : Params
        Hyperparameters of the model.
    k : int
        Number of features ``SelectKBest`` keeps.

    Returns
    -------
    list[tuple[str, Any]]
        Named pipeline steps.
    """
    spec = config.model_spec
    if "n_components" in params:
        # PLS cannot have more components than features.
        params = {**params, "n_components": min(params["n_components"], k)}
    score_func = f_regression if config.task == "regression" else f_classif
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if spec.scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("select", SelectKBest(score_func=score_func, k=k)))
    steps.append(("model", spec.build(config.task, params, config.seed)))
    return steps


def build_pipeline(config: Config, params: Params, k: int, columns: list[str] | None = None) -> Pipeline:
    """Build an unfitted pipeline.

    Parameters
    ----------
    config : Config
        The config of the run.
    params : Params
        Hyperparameters of the model.
    k : int
        Number of features ``SelectKBest`` keeps.
    columns : list[str] | None
        Feature columns. When given, the pipeline starts with a
        ``ColumnSelector`` and takes a table; otherwise it takes a matrix.

    Returns
    -------
    Pipeline
        The pipeline.
    """
    steps = _model_steps(config, params, k)
    if columns is not None:
        steps.insert(0, ("columns", ColumnSelector(columns)))
    return Pipeline(steps)
