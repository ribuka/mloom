"""Ranking the candidate columns for feature selection, outside the pipeline.

See "特徴量選択" in ``docs/spec.md``. The ranking uses the target, so it is
computed on the labeled training rows only: once for each CV fold (and reused
by every trial), and once for the final model. A trial keeps the top ``k``
columns of the ranking.
"""

import warnings

import numpy as np
from sklearn.feature_selection import f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def rank_columns(task: str, preprocess: Pipeline, x: np.ndarray, y: np.ndarray, x_statistics: np.ndarray) -> np.ndarray:
    """Rank the columns by their relation to the target, best first.

    Parameters
    ----------
    task : str
        ``"regression"`` (``f_regression``) or ``"classification"``
        (``f_classif``).
    preprocess : Pipeline
        The fitted preprocessing; the columns are ranked on its output.
    x : np.ndarray
        Features of the labeled training rows.
    y : np.ndarray
        Target of those rows.
    x_statistics : np.ndarray
        Features of the statistic rows. When the preprocessing leaves missing
        values, a copy of its output for ``x`` is filled with their median for
        the ranking only.

    Returns
    -------
    np.ndarray
        Positions of the columns, best first. A column with a NaN score (such
        as a constant one) comes last; columns with the same score keep their
        order.
    """
    z = preprocess.transform(x)
    if np.isnan(z).any():
        median = SimpleImputer(strategy="median", keep_empty_features=True).fit(preprocess.transform(x_statistics))
        z = median.transform(z)
    score_func = f_regression if task == "regression" else f_classif
    with warnings.catch_warnings():
        # A constant column gives a NaN score, which ranks it last below.
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        scores, _ = score_func(z, y)
    scores = np.where(np.isnan(scores), -np.inf, scores)
    return np.argsort(-scores, kind="stable")


def top_columns(ranking: np.ndarray, k: int) -> list[int]:
    """Keep the top ``k`` columns of a ranking.

    Parameters
    ----------
    ranking : np.ndarray
        Positions of the columns, best first (``rank_columns``).
    k : int
        Number of columns to keep.

    Returns
    -------
    list[int]
        The kept positions, in the order of the candidate columns.
    """
    return sorted(int(i) for i in ranking[:k])
