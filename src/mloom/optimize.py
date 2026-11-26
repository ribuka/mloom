"""``optimize_model``: feature selection and hyperparameter search, then the final fit.

See "処理の流れ" in ``docs/spec.md``.
"""

import warnings
from datetime import datetime
from typing import Any

import numpy as np
import optuna
import polars as pl
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

from mloom.clip import describe_clip
from mloom.columns import check_columns, check_role_values, plan_columns
from mloom.config import Config
from mloom.cv import CVPlan, order_rows, plan_cv
from mloom.data import Data, load_data
from mloom.errors import InputError, OptimizationError
from mloom.explain import shap_background
from mloom.filters import apply_filters
from mloom.metrics import BINARY_ONLY_METRICS
from mloom.pipeline import (
    Preprocessing,
    as_float,
    build_pipeline,
    fit_preprocess,
    fitted_clip_bounds,
    plan_preprocessing,
    to_matrix,
)
from mloom.result import DataSummary, Result
from mloom.row_hash import check_hashable, hash_columns, hash_rows
from mloom.rows import select_rows
from mloom.selection import (
    correlation_clusters,
    describe_selection,
    rank_columns,
    top_columns,
)
from mloom.sensors import load_unavailable_sensors
from mloom.settings import load_settings
from mloom.training_period import last_time

K_PARAM = "k"


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
    frame, path = load_data(data)
    sensors = load_unavailable_sensors(config.unavailable_sensors)
    settings = load_settings(config.settings)
    preprocessing = plan_preprocessing(config, settings)
    check_columns(frame, config, sensors)
    filtered = apply_filters(frame, config.filters, "filters")
    trainable = apply_filters(filtered, config.train_filters, "train_filters")
    check_role_values(trainable, config)
    check_hashable(trainable.schema, [config.target, *config.role_columns])
    training = select_rows(trainable, config, sensors, use_unlabeled=settings.preprocessing.use_unlabeled_rows)
    rows = order_rows(training.labeled, config)
    y = _target(rows, config)
    columns = plan_columns(rows, training.unlabeled, config, sensors, settings.columns.max_missing_rate)
    x = to_matrix(rows, columns.candidates)
    x_unlabeled = to_matrix(training.unlabeled, columns.candidates)
    x_statistics = np.vstack([x, x_unlabeled])
    # The clusters do not use the target, so they come from the statistic rows of the whole period.
    clusters = correlation_clusters(x_statistics, settings.selection.max_correlation)
    cv_plan = plan_cv(rows, y, training.unlabeled, config)

    study = _search(config, preprocessing, (x, y, x_unlabeled), clusters, cv_plan)
    best = study.best_trial
    params = {name: value for name, value in best.params.items() if name != K_PARAM}
    preprocess = fit_preprocess(preprocessing, x_statistics)
    ranking = rank_columns(config.task, preprocess, x, y, x_statistics, clusters)
    keep = top_columns(ranking, best.params[K_PARAM])
    pipeline = build_pipeline(config, params, preprocess, keep, columns.candidates)
    pipeline.fit(rows, y)
    fold_scores = best.user_attrs["fold_scores"]

    return Result(
        pipeline=pipeline,
        config=config,
        data=DataSummary(
            path=path,
            n_rows=frame.height,
            n_columns=frame.width,
            n_rows_filtered_out=frame.height - filtered.height,
            n_rows_train_filtered_out=filtered.height - trainable.height,
            n_rows_used=rows.height,
            n_unlabeled_rows_used=training.unlabeled.height,
            n_rows_without_target=training.n_without_target,
            n_rows_with_infinite_target=training.n_with_infinite_target,
            infinite_by_column=training.infinite_by_column,
        ),
        sensors=sensors,
        settings=settings,
        clip=describe_clip(settings.clip.method, fitted_clip_bounds(preprocess), columns.candidates),
        impute={"method": preprocessing.impute},
        columns=columns,
        selection=describe_selection(clusters, ranking, keep, columns.candidates, settings.selection.max_correlation),
        selected_columns=[columns.candidates[i] for i in keep],
        best_params=best.params,
        cv_scores={"folds": fold_scores, "mean": float(np.mean(fold_scores)), "std": float(np.std(fold_scores))},
        cv=cv_plan.describe(config),
        hpo=_describe_search(study, config),
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        training_hashes=np.unique(hash_rows(rows, hash_columns(columns.candidates, config))),
        last_time=last_time(rows, config.time_column),
        shap_background=shap_background(pipeline, rows),
    )


def _target(rows: pl.DataFrame, config: Config) -> np.ndarray:
    dtype = rows.schema[config.target]
    if config.task == "regression":
        if not dtype.is_numeric():
            raise InputError(f"regression needs a numeric target; {config.target!r} is {dtype}")
        return rows.select(as_float(config.target)).to_series().to_numpy()
    n_classes = rows[config.target].n_unique()
    if n_classes < 2:
        raise InputError(f"classification needs 2 or more classes; {config.target!r} has {n_classes}")
    if config.metric_name in BINARY_ONLY_METRICS and n_classes != 2:
        raise InputError(f"metric {config.metric_name!r} needs 2 classes; {config.target!r} has {n_classes}")
    return rows[config.target].to_numpy()


def _search(
    config: Config,
    preprocessing: Preprocessing,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    clusters: list[list[int]],
    cv_plan: CVPlan,
) -> optuna.Study:
    x, y, x_unlabeled = data
    # Only one representative of each cluster can be selected.
    n_clusters = len(clusters)
    metric = config.metric_spec
    scorer = get_scorer(metric.scorer)
    # Each fold fits the preprocessing and ranks the columns once for every trial.
    prepared = [
        _prepare_fold(config, preprocessing, (x[train], y[train]), x_unlabeled[unlabeled], clusters)
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
    return study


def _prepare_fold(
    config: Config,
    preprocessing: Preprocessing,
    train: tuple[np.ndarray, np.ndarray],
    x_unlabeled: np.ndarray,
    clusters: list[list[int]],
) -> tuple[Pipeline, np.ndarray]:
    # The statistics come from the training side; the representatives and their ranking, from its labeled rows only.
    x_statistics = np.vstack([train[0], x_unlabeled])
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
