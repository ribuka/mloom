"""Feature selection outside the pipeline: clusters of similar columns, and the ranking of their representatives.

See "特徴量選択" in ``docs/spec.md``. The clusters do not use the target, so
they are computed once, before the CV split, on the statistic rows. The
ranking uses the target, so it is computed on the labeled training rows only:
once for each CV fold (and reused by every trial), and once for the final
model. It keeps one representative of each cluster, and a trial keeps the top
``k`` representatives. Only the numeric candidates are clustered and ranked;
a trial keeps each categorical candidate or not by its ``use:<column>``.
"""

import warnings
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_classif, f_regression
from sklearn.impute import SimpleImputer

from mloom.models import K_PARAM, Params, use_param
from mloom.pipeline import n_numeric_columns

# A pair is correlated only when both columns have a value on this many rows.
MIN_PAIRED_ROWS = 3


def abs_spearman(x: np.ndarray) -> np.ndarray:
    """Compute the absolute Spearman correlation of every pair of columns, with missing values.

    Each column is ranked among its own missing-free values (ties get their
    average rank), then the Pearson correlation of the ranks is taken on the
    rows where both columns have a value.

    Parameters
    ----------
    x : np.ndarray
        Matrix of shape ``(rows, columns)``; NaN, +inf and -inf are missing.

    Returns
    -------
    np.ndarray
        Symmetric matrix of |ρ| from 0 to 1. A pair with fewer than
        ``MIN_PAIRED_ROWS`` such rows, or on which either column is constant,
        is 0; so is the diagonal of such a column.
    """
    observed = np.isfinite(x)
    ranks = np.zeros(x.shape)
    for j in range(x.shape[1]):
        values = x[observed[:, j], j]
        # Centered average ranks are multiples of 0.5, so the sums below are exact up to ~1e5 rows.
        ranks[observed[:, j], j] = rankdata(values, method="average") - (values.size + 1) / 2
    mask = observed.astype(np.float64)
    n = mask.T @ mask
    sums = ranks.T @ mask  # [i, j]: sum of the ranks of i on the rows where both have a value
    squares = (ranks * ranks).T @ mask
    products = ranks.T @ ranks
    # n times the sums of squared deviations; exactly 0 for a constant column when the sums are exact.
    spread = n * squares - sums * sums
    covariance = n * products - sums * sums.T
    valid = (n >= MIN_PAIRED_ROWS) & (spread > 0) & (spread.T > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(valid, covariance / np.sqrt(spread * spread.T), 0.0)
    return np.clip(np.abs(rho), 0.0, 1.0)


def correlation_clusters(x: np.ndarray, max_correlation: float) -> list[list[int]]:
    """Cluster the columns whose |Spearman correlation| is above ``max_correlation``.

    Complete linkage on the distance ``1 - |ρ|`` builds the tree; it is cut
    into the largest subtrees in which every two columns have |ρ| above
    ``max_correlation`` (``abs_spearman``), compared on |ρ| itself.

    Parameters
    ----------
    x : np.ndarray
        The numeric candidate columns on the statistic rows, before the
        preprocessing.
    max_correlation : float
        From 0 to 1; 1 clusters nothing.

    Returns
    -------
    list[list[int]]
        Positions of the columns of each cluster, every column in one
        cluster. The clusters, and the columns in each, are in the order of
        the columns.
    """
    n_columns = x.shape[1]
    if n_columns < 2:  # linkage needs two columns
        return [[i] for i in range(n_columns)]
    rho = abs_spearman(x)
    distance = 1.0 - rho
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="complete")
    members, lowest = _subtrees(tree, rho)
    # 1 - |ρ| can round two |ρ| to one distance, so the cut compares |ρ| with max_correlation instead of
    # the heights of the tree with 1 - max_correlation.
    clusters = []
    nodes = [len(members) - 1]
    while nodes:
        node = nodes.pop()
        if lowest[node] > max_correlation:
            clusters.append(sorted(members[node]))
        else:
            nodes.extend(int(child) for child in tree[node - n_columns, :2])
    return sorted(clusters)


def _subtrees(tree: np.ndarray, rho: np.ndarray) -> tuple[list[list[int]], list[float]]:
    # The columns under each node of the tree (leaves first, as scipy numbers them), and the lowest |ρ|
    # of a pair among them (inf for a single column).
    members: list[list[int]] = [[i] for i in range(rho.shape[0])]
    lowest = [np.inf] * rho.shape[0]
    for left, right, *_ in tree:
        a, b = members[int(left)], members[int(right)]
        members.append(a + b)
        lowest.append(min(lowest[int(left)], lowest[int(right)], float(rho[np.ix_(a, b)].min())))
    return members, lowest


def column_scores(
    task: str, preprocess: ColumnTransformer, x: np.ndarray, y: np.ndarray, x_statistics: np.ndarray
) -> np.ndarray:
    """Score each numeric candidate column by its relation to the target, the higher the better.

    Parameters
    ----------
    task : str
        ``"regression"`` (``f_regression``) or ``"classification"``
        (``f_classif``).
    preprocess : ColumnTransformer
        The fitted preprocessing; the columns are scored on its output.
    x : np.ndarray
        Features of the labeled training rows (every candidate column).
    y : np.ndarray
        Target of those rows.
    x_statistics : np.ndarray
        Features of the statistic rows. When the preprocessing leaves missing
        values, a copy of its output for ``x`` is filled with their median for
        the scores only.

    Returns
    -------
    np.ndarray
        The F statistic of each numeric candidate column; ``-inf`` for a NaN
        score (such as a constant column). The codes of the categorical
        candidates are not scored: their order means nothing.
    """
    numeric = slice(n_numeric_columns(preprocess))
    z = preprocess.transform(x)[:, numeric]
    if z.shape[1] == 0:  # no numeric candidate; the F statistics need a column
        return np.empty(0)
    if np.isnan(z).any():
        median = SimpleImputer(strategy="median", keep_empty_features=True)
        z = median.fit(preprocess.transform(x_statistics)[:, numeric]).transform(z)
    score_func = f_regression if task == "regression" else f_classif
    with warnings.catch_warnings():
        # A constant column gives a NaN score, which scores it lowest below.
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        scores, _ = score_func(z, y)
    return np.where(np.isnan(scores), -np.inf, scores)


def rank_representatives(scores: np.ndarray, clusters: list[list[int]]) -> np.ndarray:
    """Pick the best column of each cluster and rank those representatives.

    Parameters
    ----------
    scores : np.ndarray
        Score of each column (``column_scores``).
    clusters : list[list[int]]
        Positions of the columns of each cluster, in the order of the columns
        (``correlation_clusters``).

    Returns
    -------
    np.ndarray
        Positions of the representatives, best first. The representative of
        a cluster is its column with the highest score, the first one on a
        tie; representatives with the same score keep the order of the
        columns. The other columns are not ranked.
    """
    representatives = np.array(sorted(cluster[int(np.argmax(scores[cluster]))] for cluster in clusters), dtype=np.intp)
    return representatives[np.argsort(-scores[representatives], kind="stable")]


def rank_columns(
    task: str,
    preprocess: ColumnTransformer,
    x: np.ndarray,
    y: np.ndarray,
    x_statistics: np.ndarray,
    clusters: list[list[int]],
) -> np.ndarray:
    """Rank the representatives of the clusters of the numeric candidates by their relation to the target, best first.

    Parameters
    ----------
    task : str
        See ``column_scores``.
    preprocess : Pipeline
        See ``column_scores``.
    x : np.ndarray
        Features of the labeled training rows.
    y : np.ndarray
        Target of those rows.
    x_statistics : np.ndarray
        See ``column_scores``.
    clusters : list[list[int]]
        See ``rank_representatives``.

    Returns
    -------
    np.ndarray
        See ``rank_representatives``.
    """
    return rank_representatives(column_scores(task, preprocess, x, y, x_statistics), clusters)


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


def keep_columns(params: Params, ranking: np.ndarray, categorical: list[str], n_numeric: int) -> list[int]:
    """Tell the columns a trial or the final model keeps.

    Parameters
    ----------
    params : Params
        Parameters of the trial: ``k``, and ``use:<column>`` of each
        categorical candidate column. Without them (no numeric candidate),
        no numeric column and every categorical column is kept.
    ranking : np.ndarray
        The ranking of the numeric candidates (``rank_columns``).
    categorical : list[str]
        The categorical candidate columns.
    n_numeric : int
        Number of numeric candidate columns.

    Returns
    -------
    list[int]
        The positions, among the candidate columns, of the top ``k`` numeric
        columns and of the categorical columns whose ``use:<column>`` is
        true, in the order of the candidate columns.
    """
    used = [n_numeric + j for j, name in enumerate(categorical) if params.get(use_param(name), True)]
    return [*top_columns(ranking, params.get(K_PARAM, 0)), *used]


def selection_params(keep: list[int], categorical: list[str], n_numeric: int) -> Params:
    """Tell the parameters of the feature selection that keep given columns, as ``keep_columns`` reads them.

    Parameters
    ----------
    keep : list[int]
        Positions, among the candidate columns, of the kept columns.
    categorical : list[str]
        The categorical candidate columns.
    n_numeric : int
        Number of numeric candidate columns.

    Returns
    -------
    Params
        ``k``, the number of kept numeric columns (left out when it is 0),
        and ``use:<column>`` of each categorical column; nothing without a
        numeric candidate, as the search then has neither.
    """
    if n_numeric == 0:
        return {}
    kept = set(keep)
    params: Params = {use_param(name): n_numeric + j in kept for j, name in enumerate(categorical)}
    n_kept = sum(i < n_numeric for i in kept)
    return {K_PARAM: n_kept, **params} if n_kept else params


def describe_selection(
    clusters: list[list[int]], ranking: np.ndarray, keep: list[int], candidates: list[str], max_correlation: float
) -> dict[str, Any]:
    """Describe the clusters and their representatives for the record of the run.

    Parameters
    ----------
    clusters : list[list[int]]
        The clusters (``correlation_clusters``).
    ranking : np.ndarray
        The ranking of the final model (``rank_columns``).
    keep : list[int]
        The positions the final model selected (``top_columns``).
    candidates : list[str]
        Names of the candidate columns.
    max_correlation : float
        ``[selection] max_correlation``.

    Returns
    -------
    dict[str, Any]
        ``max_correlation``, ``n_clusters``, and ``clusters``: for each
        cluster of two or more columns, its ``columns``, its
        ``representative`` in the final model, and whether that one is
        ``selected``.
    """
    representatives = {int(i) for i in ranking}
    kept = set(keep)
    described = []
    for cluster in clusters:
        if len(cluster) < 2:  # a column alone is not worth recording
            continue
        [representative] = [i for i in cluster if i in representatives]
        described.append(
            {
                "columns": [candidates[i] for i in cluster],
                "representative": candidates[representative],
                "selected": representative in kept,
            }
        )
    return {"max_correlation": max_correlation, "n_clusters": len(clusters), "clusters": described}
