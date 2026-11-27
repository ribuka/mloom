"""``optimize_model``: feature selection and hyperparameter search, then the final fit.

See "処理の流れ" in ``docs/spec.md``. With a base (``mloom optimize --base``),
the search tries the parameters of the base first, and how the base was made
is scored on the same CV folds (see "ベースのモデル").
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

from mloom.base import Base, base_columns, describe_base, warn_dropped
from mloom.columns import require_candidates
from mloom.config import Config
from mloom.data import Data
from mloom.errors import OptimizationError
from mloom.fixed_cv import fixed_fold_models, fixed_fold_scores
from mloom.interpretation import FoldModels, importance_groups
from mloom.models import K_PARAM, Params, model_params, use_param
from mloom.pipeline import (
    build_pipeline,
    fit_fold_preprocess,
    fit_preprocess,
    plan_preprocessing,
)
from mloom.result import Result
from mloom.selection import (
    correlation_clusters,
    describe_selection,
    keep_columns,
    rank_columns,
)
from mloom.settings import load_settings
from mloom.training import (
    FinalModel,
    TrainingData,
    describe_scores,
    fit_final,
    prepare_training,
)

MODE = "optimize"


@dataclass(frozen=True)
class _BaseScores:
    # How the base was made, scored on the CV folds of this run (``_score_base``); folds is None when not compared.
    dropped: list[dict[str, Any]]
    folds: list[float] | None
    not_compared: list[str]


def optimize_model(data: Data, config: Config, *, base: Base | None = None) -> Result:
    """Exclude columns, search features and hyperparameters, and fit the best pipeline.

    Parameters
    ----------
    data : Data
        Path of a Parquet file, or a polars DataFrame.
    config : Config
        What to optimize and how. To start from a base, build it with
        ``dataclasses.replace(base.config, ...)``.
    base : Base | None
        A base (``mloom.load_base``). When given, the search tries its
        parameters first if ``config.model`` is its model and they are in
        the search space of this run (otherwise, with a warning), how it was
        made is scored on the CV folds of this run, and the record gets
        ``base``. The columns of the base that are no longer candidates are
        dropped from that score, each with a warning (``warnings.warn``).
        The saved model is always that of this search.

    Returns
    -------
    Result
        The fitted pipeline and the record of the run; ``Result.save`` writes
        both as ``model.pkl``, and refuses to overwrite the base.

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
    plan = training.columns
    n_numeric = len(plan.numeric)
    # The clusters do not use the target, so they come from the statistic rows of the whole period.
    clusters = correlation_clusters(training.x_statistics[:, :n_numeric], training.settings.selection.max_correlation)
    # The base is scored before the search, so that an error in it comes before the long part of the run.
    base_scores = None if base is None else _score_base(base, config, training)
    first = None if base is None else _first_trial(base, config, len(clusters), plan.categorical)

    study, prepared = _search(config, training, clusters, first)
    best = study.best_trial
    params = model_params(best.params)
    preprocess = fit_preprocess(training.preprocessing, training.x_statistics, n_numeric)
    ranking = rank_columns(config.task, preprocess, training.x, training.y, training.x_statistics, clusters)
    keep = keep_columns(best.params, ranking, plan.categorical, n_numeric)
    folds = [
        (fold_preprocess, keep_columns(best.params, fold_ranking, plan.categorical, n_numeric))
        for fold_preprocess, fold_ranking in prepared
    ]
    candidates = plan.candidates
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
        base=None
        if base is None or base_scores is None
        else _describe_base(base, base_scores, first is not None, best.user_attrs["fold_scores"], config),
    )


def _score_base(base: Base, config: Config, training: TrainingData) -> _BaseScores:
    # The model, parameters, columns still candidates and preprocessing of the base, on the folds of this run.
    keep, dropped = base_columns(base, training.columns)
    warn_dropped(dropped)
    differs = {
        "target": config.target != base.config.target,
        "task": config.task != base.config.task,
        "metric": config.metric_name != base.config.metric_name,
        "columns": not keep,
    }
    not_compared = [reason for reason, differ in differs.items() if differ]
    if not_compared:
        return _BaseScores(dropped, None, not_compared)
    params = model_params(base.best_params)
    preprocessing = plan_preprocessing(base.config, base.settings)
    models = fixed_fold_models(base.config, params, preprocessing, keep, training)
    return _BaseScores(dropped, fixed_fold_scores(models, training), [])


def _first_trial(base: Base, config: Config, n_clusters: int, categorical: list[str]) -> Params | None:
    # The parameters of the base, when they are in the search space of this run, with k kept within its range.
    if config.model != base.config.model:
        return None
    params = dict(base.best_params)
    if K_PARAM in params and n_clusters:
        params[K_PARAM] = min(params[K_PARAM], n_clusters)
    space = _search_space(config, n_clusters, categorical)
    outside = sorted(
        name for name in params.keys() | space.keys() if name not in space or not _contains(space[name], params.get(name))
    )
    if outside:
        described = ", ".join(f"{name}={base.best_params.get(name)!r}" for name in outside)
        warnings.warn(
            f"the parameters of the base are not tried first: {described} not in the search space of this run",
            UserWarning,
            stacklevel=3,
        )
        return None
    return params


def _search_space(config: Config, n_clusters: int, categorical: list[str]) -> dict[str, BaseDistribution]:
    # The distributions a trial of this run samples from; the values of this throwaway trial are not used.
    trial = optuna.create_study().ask()
    _suggest(trial, config, n_clusters, categorical)
    return trial.distributions


def _suggest(trial: optuna.Trial, config: Config, n_clusters: int, categorical: list[str]) -> tuple[Params, Params]:
    # The hyperparameters of the model, and the parameters of the feature selection (keep_columns).
    params = config.model_spec.suggest(trial, n_clusters, bool(categorical))
    if n_clusters == 0:
        # No numeric candidate: k cannot range from 1 to 0, and every categorical column is kept.
        return params, {}
    selection: Params = {K_PARAM: trial.suggest_int(K_PARAM, 1, n_clusters, log=True)}
    for name in categorical:
        selection[use_param(name)] = trial.suggest_categorical(use_param(name), [True, False])
    return params, selection


def _contains(distribution: BaseDistribution, value: object) -> bool:
    # Optuna 5.1.0 uses an enqueued value out of its distribution as it is (2026-11-26), so it is checked here.
    if isinstance(distribution, CategoricalDistribution):
        # 1 == True in Python, so the type is compared too.
        return any(type(value) is type(choice) and value == choice for choice in distribution.choices)
    if not isinstance(distribution, FloatDistribution | IntDistribution):
        return False
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if not distribution.low <= value <= distribution.high:
        return False
    if distribution.step is None:
        return True
    n_steps = (value - distribution.low) / distribution.step
    return math.isclose(n_steps, round(n_steps))


def _describe_base(
    base: Base, scores: _BaseScores, enqueued: bool, best_folds: list[float], config: Config
) -> dict[str, Any]:
    folds = scores.folds
    return {
        **describe_base(base, MODE, scores.dropped),
        "enqueued": enqueued,
        "cv_scores_on_this_data": None if folds is None else describe_scores(folds),
        "comparison": None if folds is None else describe_scores(_differences(best_folds, folds, config)),
        "not_compared": scores.not_compared,
    }


def _differences(ours: list[float], theirs: list[float], config: Config) -> list[float]:
    # Positive where this run is better than the base.
    if config.metric_spec.direction == "minimize":
        return [base - this for this, base in zip(ours, theirs, strict=True)]
    return [this - base for this, base in zip(ours, theirs, strict=True)]


def _search(
    config: Config, training: TrainingData, clusters: list[list[int]], first: Params | None
) -> tuple[optuna.Study, list[tuple[ColumnTransformer, np.ndarray]]]:
    # Also returns the fitted preprocessing and the ranking of each fold (``_prepare_fold``).
    x, y, cv_plan = training.x, training.y, training.cv_plan
    categorical = training.columns.categorical
    n_numeric = len(training.columns.numeric)
    # Only one representative of each cluster can be selected.
    n_clusters = len(clusters)
    metric = config.metric_spec
    scorer = get_scorer(metric.scorer)
    # Each fold fits the preprocessing and ranks the columns once for every trial.
    prepared = [
        _prepare_fold(config, training, train, unlabeled, clusters)
        for (train, _), unlabeled in zip(cv_plan.folds, cv_plan.unlabeled_folds, strict=True)
    ]

    def objective(trial: optuna.Trial) -> float:
        params, selection = _suggest(trial, config, n_clusters, categorical)
        fold_scores = []
        fold_errors = []
        for fold, ((train, test), (preprocess, ranking)) in enumerate(zip(cv_plan.folds, prepared, strict=True)):
            keep = keep_columns(selection, ranking, categorical, n_numeric)
            pipeline = build_pipeline(config, params, preprocess, keep)
            score, error = _score_fold(pipeline, scorer, (x[train], y[train]), (x[test], y[test]))
            fold_scores.append(metric.report_value(score))
            if error is not None:
                fold_errors.append({"fold": fold, "error": error})
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_errors", fold_errors)
        # A NaN (a fold failed) makes Optuna record the trial as failed.
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction=metric.direction, sampler=optuna.samplers.TPESampler(seed=config.seed))
    if first is not None:
        # Tried first, and compared with the other trials like any of them.
        study.enqueue_trial(first)
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout)
    if not _completed(study):
        raise OptimizationError("no trial finished with a score; see the warnings of the failed trials")
    return study, prepared


def _prepare_fold(
    config: Config, training: TrainingData, train: np.ndarray, unlabeled: np.ndarray, clusters: list[list[int]]
) -> tuple[ColumnTransformer, np.ndarray]:
    # The statistics come from the training side; the representatives and their ranking, from its labeled rows only.
    x_statistics = training.fold_statistics(train, unlabeled)
    preprocess = fit_fold_preprocess(training.preprocessing, x_statistics, len(training.columns.numeric))
    return preprocess, rank_columns(
        config.task, preprocess, training.x[train], training.y[train], x_statistics, clusters
    )


def _score_fold(
    pipeline: Pipeline,
    scorer: Any,  # noqa: ANN401 - a scikit-learn scorer
    train: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
) -> tuple[float, str | None]:
    # The score, or NaN with the repr of the error when the fold failed.
    try:
        pipeline.fit(*train)
        return float(scorer(pipeline, *test)), None
    except Exception as error:  # noqa: BLE001 - a failed fold fails the trial, not the run
        warnings.warn(f"a CV fold failed: {error!r}", stacklevel=2)
        return float("nan"), repr(error)


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
        "trials": [_describe_trial(trial) for trial in study.trials],
    }


def _describe_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    # A failed fold is null: a NaN in the record would make the JSON of mloom show invalid.
    completed = trial.state == optuna.trial.TrialState.COMPLETE
    return {
        "number": trial.number,
        "state": trial.state.name.lower(),
        "value": trial.value if completed else None,
        "folds": [None if math.isnan(score) else score for score in trial.user_attrs["fold_scores"]],
        "params": trial.params,
        "seconds": None if trial.duration is None else trial.duration.total_seconds(),
        "errors": trial.user_attrs["fold_errors"],
    }
