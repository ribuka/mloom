"""Tests of the clusters of similar columns and of the ranking of their representatives."""

from itertools import combinations

import numpy as np
import pytest
from scipy.stats import spearmanr

from mloom.selection import (
    abs_spearman,
    correlation_clusters,
    describe_selection,
    rank_representatives,
)

N_ROWS = 200


def _assert_correlated_within(x: np.ndarray, clusters: list[list[int]], max_correlation: float) -> None:
    rho = abs_spearman(x)
    for cluster in clusters:
        for i, j in combinations(cluster, 2):
            assert rho[i, j] > max_correlation


def test_monotone_columns_are_clustered_and_independent_ones_are_not() -> None:
    """A monotone transform of a column joins its cluster; an independent column stays alone."""
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(2, N_ROWS))
    x = np.column_stack([u, v, np.exp(u)])
    clusters = correlation_clusters(x, 0.9)
    assert clusters == [[0, 2], [1]]
    _assert_correlated_within(x, clusters, 0.9)


def test_a_pair_exactly_at_max_correlation_is_not_clustered() -> None:
    """Only |ρ| above ``max_correlation`` clusters: a pair at the boundary stays apart, just below it joins."""
    rng = np.random.default_rng(1)
    u = rng.normal(size=N_ROWS)
    x = np.column_stack([u, u + rng.normal(size=N_ROWS)])
    r = abs_spearman(x)[0, 1]
    assert 0.3 < r < 0.9
    assert correlation_clusters(x, r) == [[0], [1]]
    assert correlation_clusters(x, r - 1e-9) == [[0, 1]]


def test_a_chain_of_correlations_is_not_one_cluster() -> None:
    """With a–b and b–c above the threshold but a–c below it, a and c never share a cluster."""
    rng = np.random.default_rng(2)
    a, c = rng.normal(size=(2, N_ROWS))
    x = np.column_stack([a, (a + c) / np.sqrt(2), c])
    rho = abs_spearman(x)
    assert min(rho[0, 1], rho[1, 2]) > 0.6 > rho[0, 2]
    clusters = correlation_clusters(x, 0.6)
    assert not any({0, 2} <= set(cluster) for cluster in clusters)
    _assert_correlated_within(x, clusters, 0.6)


def test_max_correlation_of_one_clusters_nothing() -> None:
    """``max_correlation = 1`` leaves every column alone, even identical ones."""
    rng = np.random.default_rng(3)
    u = rng.normal(size=N_ROWS)
    assert correlation_clusters(np.column_stack([u, u, 2 * u]), 1.0) == [[0], [1], [2]]


def test_one_column_is_one_cluster() -> None:
    """A single candidate column is a cluster of its own."""
    assert correlation_clusters(np.ones((5, 1)), 0.9) == [[0]]


def test_without_missing_values_it_is_spearman() -> None:
    """Without missing values, |ρ| equals ``scipy.stats.spearmanr``, ties included."""
    rng = np.random.default_rng(4)
    x = np.round(rng.normal(size=(N_ROWS, 5)), 1)
    x[:, 1] += x[:, 0]
    expected = np.abs(spearmanr(x).statistic)
    np.testing.assert_allclose(abs_spearman(x), expected, rtol=1e-12)


def test_pairs_with_too_few_rows_or_a_constant_column_are_not_correlated() -> None:
    """A pair with 2 rows where both have a value, or on whose rows a column is constant, has |ρ| of 0."""
    rng = np.random.default_rng(5)
    base = rng.normal(size=20)
    two_rows = np.full(20, np.nan)
    two_rows[:2] = base[:2]
    constant_on_shared = base.copy()
    constant_on_shared[:10] = 0.0
    shared_only = np.full(20, np.nan)
    shared_only[:10] = base[:10]
    x = np.column_stack([base, two_rows, constant_on_shared, shared_only])
    rho = abs_spearman(x)
    assert rho[0, 1] == 0.0
    assert rho[2, 3] == 0.0
    # 0 and 3 are identical where both have values; 2 joins neither, since its pair with 3 is not correlated.
    assert correlation_clusters(x, 0.0) == [[0, 3], [1], [2]]


def test_missing_values_drop_only_the_rows_of_the_pair() -> None:
    """Each column is ranked on its own values; the pair is correlated on the rows where both have one."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=(30, 2))
    x[:5, 1] = np.nan
    ranks = np.column_stack([np.argsort(np.argsort(x[:, 0])), np.full(30, np.nan)])
    ranks[5:, 1] = np.argsort(np.argsort(x[5:, 1]))
    expected = abs(np.corrcoef(ranks[5:, 0], ranks[5:, 1])[0, 1])
    assert abs_spearman(x)[0, 1] == pytest.approx(expected, rel=1e-12)


def test_infinite_values_are_missing() -> None:
    """+inf and -inf are missing, like NaN."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 3))
    with_nan, with_inf = x.copy(), x.copy()
    with_nan[[1, 2], 0] = np.nan
    with_inf[1, 0], with_inf[2, 0] = np.inf, -np.inf
    np.testing.assert_array_equal(abs_spearman(with_inf), abs_spearman(with_nan))


def test_representative_is_the_best_column_of_its_cluster() -> None:
    """The representative has the highest score, the first column on a tie; the others are not ranked."""
    scores = np.array([1.0, 5.0, 3.0, 3.0, 4.0, -np.inf, -np.inf])
    clusters = [[0, 1], [2, 3], [4], [5, 6]]
    np.testing.assert_array_equal(rank_representatives(scores, clusters), [1, 4, 2, 5])


def test_representatives_with_the_same_score_keep_the_column_order() -> None:
    """Representatives with the same score are ranked in the order of the columns."""
    scores = np.array([2.0, 1.0, 2.0])
    np.testing.assert_array_equal(rank_representatives(scores, [[0], [1, 2]]), [0, 2])


def test_description_lists_clusters_of_two_or_more_columns() -> None:
    """Only clusters of two or more columns are described, with their representative and whether it is selected."""
    clusters = [[0, 2], [1], [3, 4]]
    ranking = np.array([2, 1, 3])
    described = describe_selection(clusters, ranking, [1, 2], ["a", "b", "c", "d", "e"], 0.9)
    assert described == {
        "max_correlation": 0.9,
        "n_clusters": 3,
        "clusters": [
            {"columns": ["a", "c"], "representative": "c", "selected": True},
            {"columns": ["d", "e"], "representative": "d", "selected": False},
        ],
    }
