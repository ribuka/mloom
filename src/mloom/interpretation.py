"""Interpretation of the model for the record of the run: importance, coefficients and VIP.

See "解釈" in ``docs/spec.md``. The permutation importance of each group of
similar columns is measured on the test side of each CV fold, with the model
of the best trial refitted on that fold; the coefficients of the linear models
and the VIP of PLS come from the final model. Every value is a plain Python
value, and none is NaN, so that the record stays valid JSON.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import is_classifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

from mloom.config import Config
from mloom.cv import Split
from mloom.explain import is_linear, linear_weights
from mloom.models import Params
from mloom.pipeline import build_pipeline, estimator_input, selected_columns
from mloom.selection import top_columns


@dataclass(frozen=True)
class FoldModels:
    """How to rebuild the model of the best trial on each CV fold.

    Parameters
    ----------
    config : Config
        The config of the run.
    params : Params
        Hyperparameters of the best trial, without ``k``.
    k : int
        ``k`` of the best trial.
    folds : list[Split]
        Train and test indices of the labeled rows of each fold.
    prepared : list[tuple[Pipeline, np.ndarray]]
        The fitted preprocessing and the ranking of each fold, as the search
        used them.
    """

    config: Config
    params: Params
    k: int
    folds: list[Split]
    prepared: list[tuple[Pipeline, np.ndarray]]


@dataclass(frozen=True)
class Importance:
    """The permutation importance and the scores it is measured from.

    Parameters
    ----------
    record : dict[str, Any] | None
        ``interpretation.permutation_importance`` of the record; ``None`` when
        it is not computed.
    baselines : list[float]
        The metric of each fold on its test side before any permutation;
        empty when it is not computed.
    """

    record: dict[str, Any] | None
    baselines: list[float]


def importance_groups(keep: list[int], clusters: list[list[int]]) -> list[tuple[int, list[int]]]:
    """Group each selected column with the columns of its cluster.

    Parameters
    ----------
    keep : list[int]
        Positions of the columns the final model selected (``top_columns``).
    clusters : list[list[int]]
        The clusters of similar columns (``correlation_clusters``).

    Returns
    -------
    list[tuple[int, list[int]]]
        For each selected column in the order of ``keep``, its position and
        the positions of every column of its cluster, in the order of the
        candidate columns.
    """
    cluster_of = {i: cluster for cluster in clusters for i in cluster}
    return [(i, cluster_of[i]) for i in keep]


def permutation_importance(
    models: FoldModels,
    data: tuple[np.ndarray, np.ndarray],
    groups: list[tuple[int, list[int]]],
    candidates: list[str],
    n_repeats: int,
) -> Importance:
    """Measure how much the score of each fold worsens when a group of columns is permuted.

    Parameters
    ----------
    models : FoldModels
        The models of the best trial on the CV folds.
    data : tuple[np.ndarray, np.ndarray]
        The candidate columns of the labeled rows before the preprocessing
        (``to_matrix``), and their target.
    groups : list[tuple[int, list[int]]]
        The groups (``importance_groups``).
    candidates : list[str]
        Names of the candidate columns.
    n_repeats : int
        How many times each group is permuted on each fold; 0 computes
        nothing and fits no model.

    Returns
    -------
    Importance
        The record holds the ``metric``, ``n_repeats``, and ``groups``, by
        their ``mean`` over the folds, largest first: the selected ``column``,
        the permuted ``columns``, and the ``mean``, ``std`` and ``folds`` of
        the worsening. A positive value means the score worsens.
    """
    if n_repeats == 0:
        return Importance(record=None, baselines=[])
    x, y = data
    config = models.config
    metric = config.metric_spec
    scorer = get_scorer(metric.scorer)
    sign = 1.0 if metric.direction == "minimize" else -1.0
    rng = np.random.default_rng(config.seed)
    baselines = []
    worsening = np.zeros((len(groups), len(models.folds)))
    for fold, ((train, test), (preprocess, ranking)) in enumerate(zip(models.folds, models.prepared, strict=True)):
        pipeline = _fit_fold(models, preprocess, ranking, (x[train], y[train]))
        x_test, y_test = x[test], y[test]
        baseline = metric.report_value(float(scorer(pipeline, x_test, y_test)))
        baselines.append(baseline)
        for g, (_, columns) in enumerate(groups):
            values = []
            for _ in range(n_repeats):
                permuted = x_test.copy()
                permuted[:, columns] = x_test[rng.permutation(len(test))][:, columns]
                values.append(sign * (metric.report_value(float(scorer(pipeline, permuted, y_test))) - baseline))
            worsening[g, fold] = np.mean(values)
    means = worsening.mean(axis=1)
    described = [
        {
            "column": candidates[groups[g][0]],
            "columns": [candidates[i] for i in groups[g][1]],
            "mean": float(means[g]),
            "std": float(np.std(worsening[g])),
            "folds": [float(value) for value in worsening[g]],
        }
        for g in np.argsort(-means, kind="stable")
    ]
    return Importance(
        record={"metric": config.metric_name, "n_repeats": n_repeats, "groups": described}, baselines=baselines
    )


def _fit_fold(
    models: FoldModels, preprocess: Pipeline, ranking: np.ndarray, train: tuple[np.ndarray, np.ndarray]
) -> Pipeline:
    # The same pipeline the best trial fitted on this fold; an error is not expected, so it is raised.
    pipeline = build_pipeline(models.config, models.params, preprocess, top_columns(ranking, models.k))
    return pipeline.fit(*train)


def linear_coefficients(pipeline: Pipeline, background: np.ndarray) -> dict[str, Any] | None:
    """Describe how the output of a linear model changes with each selected column.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted final pipeline.
    background : np.ndarray
        ``shap_background`` of the final model.

    Returns
    -------
    dict[str, Any] | None
        ``None`` for the tree models. Otherwise the ``output``
        (``"prediction"`` or ``"log_odds"``) and, for each ``class``
        (``None`` for regression; ``str`` of ``classes_[1]`` for 2 classes,
        whose log odds are those of ``classes_[1]``; ``str`` of each class
        otherwise) and each
        selected ``column``, the change of the output when the column moves
        by 1 in its original unit (``per_unit``) and by 1 standard deviation
        (``per_std``). Within a class, the largest ``abs(per_std)`` comes
        first.
    """
    estimator = pipeline[-1]
    if not is_linear(estimator):
        return None
    _, weights = linear_weights(estimator, background)
    keep = pipeline.named_steps["select"].transformers[0][2]
    scale = pipeline.named_steps["preprocess"].estimator.named_steps["scale"].scale_[keep]
    names = selected_columns(pipeline)
    classifier = is_classifier(estimator)
    # One output for regression and for 2 classes (those of classes_[1]), one for each class otherwise.
    classes = [str(label) for label in estimator.classes_[-weights.shape[1] :]] if classifier else [None]
    described = []
    for output, label in enumerate(classes):
        order = np.argsort(-np.abs(weights[:, output]), kind="stable")
        described.extend(
            {
                "class": label,
                "column": names[j],
                "per_unit": float(weights[j, output] / scale[j]),
                "per_std": float(weights[j, output]),
            }
            for j in order
        )
    return {"output": "log_odds" if classifier else "prediction", "columns": described}


def pls_vip(pipeline: Pipeline, rows: pl.DataFrame) -> dict[str, Any] | None:
    """Compute the VIP (variable importance in projection) of each selected column of PLS.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted final pipeline.
    rows : pl.DataFrame
        The rows the final model was fitted on.

    Returns
    -------
    dict[str, Any] | None
        ``None`` for a model other than PLS. Otherwise ``columns``: each
        selected ``column`` and its ``vip``, largest first. Every ``vip`` is
        ``None`` when the components explain nothing of the target.
    """
    estimator = pipeline[-1]
    if not isinstance(estimator, PLSRegression):
        return None
    scores = estimator.transform(estimator_input(pipeline, rows))
    weights = estimator.x_weights_
    explained = estimator.y_loadings_[0] ** 2 * np.sum(scores**2, axis=0)
    names = selected_columns(pipeline)
    total = float(np.sum(explained))
    if total == 0:
        return {"columns": [{"column": name, "vip": None} for name in names]}
    norms = np.linalg.norm(weights, axis=0)
    # A component left after the target is fully explained has zero weights; it adds nothing.
    shares = np.divide(weights, norms, out=np.zeros_like(weights), where=norms > 0) ** 2
    vip = np.sqrt(len(names) * (shares @ explained) / total)
    order = np.argsort(-vip, kind="stable")
    return {"columns": [{"column": names[j], "vip": float(vip[j])} for j in order]}
