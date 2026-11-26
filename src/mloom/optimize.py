"""``optimize_model``: feature selection and hyperparameter search, then the final fit.

See "処理の流れ" in ``docs/spec.md``.
"""

import warnings
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

from mloom.columns import require_candidates
from mloom.config import Config
from mloom.data import Data
from mloom.errors import OptimizationError
from mloom.interpretation import FoldModels, importance_groups
from mloom.models import K_PARAM
from mloom.pipeline import Preprocessing, build_pipeline, fit_preprocess
from mloom.result import Result
from mloom.selection import (
    correlation_clusters,
    describe_selection,
    rank_columns,
    top_columns,
)
from mloom.settings import load_settings
from mloom.training import FinalModel, TrainingData, fit_final, prepare_training


def optimize_model(data: Data, config: Config) -> Result:
    """Exclude columns, search features and hyperparameters, and fit the best pipeline.

    Parameters
    ----------
    data : Data
        Path of a Parquet file, or a polars DataFrame.
    config : Config
        What to optimize and how.

    Returns
    -------
    Result
        The fitted pipeline and the record of the run; ``Result.save`` writes
        both as ``model.pkl``.

    Raises
    ------
    InputError
        When the data, ``unavailable-sensors.json`` or ``mloom.toml`` is
        invalid or does not fit the config, including the row filters and the
        imputation the model cannot do without, or no row is left after the
        filters.
        Raised before any model is fitted.
    OptimizationError
        When no trial of the search finished with a score.
    """
    training = prepare_training(data, config, load_settings(config.settings), require_candidates)
    # The clusters do not use the target, so they come from the statistic rows of the whole period.
    clusters = correlation_clusters(training.x_statistics, training.settings.selection.max_correlation)

    study, prepared = _search(config, training, clusters)
    best = study.best_trial
    k = best.params[K_PARAM]
    params = {name: value for name, value in best.params.items() if name != K_PARAM}
    preprocess = fit_preprocess(training.preprocessing, training.x_statistics)
    ranking = rank_columns(config.task, preprocess, training.x, training.y, training.x_statistics, clusters)
    keep = top_columns(ranking, k)
    folds = [(fold_preprocess, top_columns(fold_ranking, k)) for fold_preprocess, fold_ranking in prepared]
    candidates = training.columns.candidates
    return fit_final(
        config,
        training,
        FinalModel(
            preprocess=preprocess,
            keep=keep,
            fold_models=FoldModels(config, params, training.cv_plan.folds, folds),
            importance_groups=importance_groups(keep, clusters),
        ),
        best.user_attrs["fold_scores"],
        selection=describe_selection(
            clusters,
            ranking,
            keep,
            candidates,
            training.settings.selection.max_correlation,
        ),
        best_params=best.params,
        hpo=_describe_search(study, config),
        base=None,
    )


def _search(
    config: Config, training: TrainingData, clusters: list[list[int]]
) -> tuple[optuna.Study, list[tuple[Pipeline, np.ndarray]]]:
    # Also returns the fitted preprocessing and the ranking of each fold (``_prepare_fold``).
    x, y, cv_plan = training.x, training.y, training.cv_plan
    # Only one representative of each cluster can be selected.
    n_clusters = len(clusters)
    metric = config.metric_spec
    scorer = get_scorer(metric.scorer)
    # Each fold fits the preprocessing and ranks the columns once for every trial.
    prepared = [
        _prepare_fold(
            config,
            training.preprocessing,
            (x[train], y[train]),
            training.fold_statistics(train, unlabeled),
            clusters,
        )
        for (train, _), unlabeled in zip(cv_plan.folds, cv_plan.unlabeled_folds, strict=True)
    ]

    def objective(trial: optuna.Trial) -> float:
        params = config.model_spec.suggest(trial, n_clusters)
        k = trial.suggest_int(K_PARAM, 1, n_clusters, log=True)
        fold_scores = []
        for (train, test), (preprocess, ranking) in zip(cv_plan.folds, prepared, strict=True):
            pipeline = build_pipeline(config, params, preprocess, top_columns(ranking, k))
            score = _score_fold(pipeline, scorer, (x[train], y[train]), (x[test], y[test]))
            fold_scores.append(metric.report_value(score))
        trial.set_user_attr("fold_scores", fold_scores)
        # A NaN (a fold failed) makes Optuna record the trial as failed.
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction=metric.direction, sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout)
    if not _completed(study):
        raise OptimizationError("no trial finished with a score; see the warnings of the failed trials")
    return study, prepared


def _prepare_fold(
    config: Config,
    preprocessing: Preprocessing,
    train: tuple[np.ndarray, np.ndarray],
    x_statistics: np.ndarray,
    clusters: list[list[int]],
) -> tuple[Pipeline, np.ndarray]:
    # The statistics come from the training side; the representatives and their ranking, from its labeled rows only.
    preprocess = fit_preprocess(preprocessing, x_statistics)
    return preprocess, rank_columns(config.task, preprocess, *train, x_statistics, clusters)


def _score_fold(
    pipeline: Pipeline,
    scorer: Any,  # noqa: ANN401 - a scikit-learn scorer
    train: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
) -> float:
    try:
        pipeline.fit(*train)
        return float(scorer(pipeline, *test))
    except Exception as error:  # noqa: BLE001 - a failed fold fails the trial, not the run
        warnings.warn(f"a CV fold failed: {error!r}", stacklevel=2)
        return float("nan")


def _completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]


def _describe_search(study: optuna.Study, config: Config) -> dict[str, Any]:
    return {
        "sampler": "TPESampler",
        "n_trials": config.n_trials,
        "n_trials_completed": len(_completed(study)),
        "n_trials_failed": sum(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials),
        "timeout": config.timeout,
        "best_trial": study.best_trial.number,
    }
