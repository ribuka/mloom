"""SHAP contributions of the selected columns to the output of the estimator.

See "SHAP の寄与" in ``docs/spec.md``. The contributions are computed on the
input of the estimator (after the preprocessing and the selection of the
kept columns) and named by the selected columns. Tree models use
``shap.TreeExplainer``; the linear models (``ridge`` and ``pls``) use the
linear SHAP with independent features, from the mean of the input on the
rows trained on (``shap_background``). shap is imported only when the
contributions are computed.
"""

import warnings
from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from mloom.pipeline import estimator_input, selected_columns

SHAP_PREFIX = "shap_"
_TREES = (LGBMClassifier, LGBMRegressor, RandomForestClassifier, RandomForestRegressor)


def shap_background(pipeline: Pipeline, rows: pl.DataFrame) -> np.ndarray:
    """Compute the mean of the input of the estimator, the base of the linear SHAP.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted pipeline.
    rows : pl.DataFrame
        The rows the final model was fitted on.

    Returns
    -------
    np.ndarray
        Mean of each column of the input of the estimator, missing values
        left out (NaN for a column with none; the linear models impute, so
        their input has no missing value).
    """
    x = estimator_input(pipeline, rows)
    known = ~np.isnan(x)
    counts = known.sum(axis=0)
    totals = np.where(known, x, 0.0).sum(axis=0)
    return np.divide(totals, counts, out=np.full(x.shape[1], np.nan), where=counts > 0)


def shap_names(pipeline: Pipeline) -> list[str]:
    """Name the SHAP columns of a model.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted pipeline.

    Returns
    -------
    list[str]
        ``shap_base`` and ``shap_<column>`` for each selected column; for
        more than 2 classes, ``shap_base_<class>`` and
        ``shap_<class>_<column>`` for each class in turn.
    """
    columns = selected_columns(pipeline)
    if not is_classifier(pipeline) or len(pipeline.classes_) == 2:
        return [f"{SHAP_PREFIX}base", *(f"{SHAP_PREFIX}{name}" for name in columns)]
    return [
        name
        for label in pipeline.classes_
        for name in (f"{SHAP_PREFIX}base_{label}", *(f"{SHAP_PREFIX}{label}_{column}" for column in columns))
    ]


def shap_columns(
    pipeline: Pipeline, frame: pl.DataFrame, predicted: np.ndarray, background: np.ndarray
) -> list[pl.Series]:
    """Compute the SHAP columns.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted pipeline.
    frame : pl.DataFrame
        The data to predict, with every candidate column.
    predicted : np.ndarray
        Boolean mask of the rows that were predicted (``passed_filter``);
        the other rows are NaN.
    background : np.ndarray
        ``shap_background`` of ``model.pkl``.

    Returns
    -------
    list[pl.Series]
        The columns of ``shap_names``. Their sum over a class (the base and
        the contributions) is the output of the model: the prediction for
        regression; for classification, the log odds of ``LGBMClassifier``
        (``predict(x, raw_score=True)``) and of ``LogisticRegression``
        (``decision_function``), and the probability of
        ``RandomForestClassifier``. For 2 classes, they are those of
        ``classes_[1]``.
    """
    n_columns = len(selected_columns(pipeline))
    n_outputs = len(pipeline.classes_) if is_classifier(pipeline) and len(pipeline.classes_) > 2 else 1
    values = np.full((frame.height, n_outputs, n_columns + 1), np.nan)
    if predicted.any():
        x = estimator_input(pipeline, frame.filter(pl.Series(predicted)))
        base, contributions = _explain(pipeline, x, background)
        values[predicted, :, 0] = base
        values[predicted, :, 1:] = np.moveaxis(contributions, 2, 1)
    flat = values.reshape(frame.height, -1)
    return [pl.Series(name, flat[:, index]) for index, name in enumerate(shap_names(pipeline))]


def _explain(pipeline: Pipeline, x: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The base of shape (outputs,) and the contributions of shape (rows, columns, outputs).
    estimator = pipeline[-1]
    if isinstance(estimator, _TREES):
        return _tree(estimator, x)
    return _linear(_output(estimator), x, background)


def _tree(estimator: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - a tree model
    import shap

    explainer = shap.TreeExplainer(estimator)
    with warnings.catch_warnings():
        # shap warns of a list for a 2-class LGBMClassifier, but returns one (rows, columns) array.
        warnings.filterwarnings("ignore", "LightGBM binary classifier with TreeExplainer", UserWarning)
        contributions = np.asarray(explainer.shap_values(x), dtype=np.float64)
    base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=np.float64))
    if contributions.ndim == 2:
        contributions = contributions[:, :, None]
    if is_classifier(estimator) and len(estimator.classes_) == 2 and base.size == 2:
        # RandomForestClassifier explains both probabilities; keep those of classes_[1].
        return base[1:], contributions[:, :, 1:]
    return base, contributions


def _output(estimator: Any) -> Any:  # noqa: ANN401 - an estimator and its output function
    return estimator.decision_function if is_classifier(estimator) else estimator.predict


def _linear(output: Any, x: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - the output function of a linear model
    # The output is affine in x: its weights are its changes from the background along each column.
    probes = np.vstack([background, background + np.eye(background.size)])
    outputs = np.asarray(output(probes), dtype=np.float64).reshape(probes.shape[0], -1)
    weights = outputs[1:] - outputs[0]
    return outputs[0], (x - background)[:, :, None] * weights[None, :, :]
