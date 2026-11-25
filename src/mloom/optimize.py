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

from mloom.columns import check_columns, plan_columns
from mloom.config import Config
from mloom.cv import CVPlan, order_rows, plan_cv
from mloom.data import Data, load_data
from mloom.errors import InputError, OptimizationError
from mloom.metrics import BINARY_ONLY_METRICS
from mloom.pipeline import as_float, build_pipeline, fit_preprocess, to_matrix
from mloom.result import DataSummary, Result
from mloom.rows import select_rows
from mloom.sensors import load_unavailable_sensors
from mloom.settings import load_settings

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
        ``model.pkl`` and ``run.json``.

    Raises
    ------
    InputError
        When the data, ``unavailable-sensors.json`` or ``mloom.toml`` is
        invalid or does not fit the config.
        Raised before any model is fitted.
    OptimizationError
        When no trial of the search finished with a score.
    """
    frame, path = load_data(data)
    sensors = load_unavailable_sensors(config.unavailable_sensors)
    settings = load_settings(config.settings)
    check_columns(frame, config, sensors)
    training = select_rows(frame, config, sensors, use_unlabeled=settings.use_unlabeled_rows)
    rows = order_rows(training.labeled, config)
    y = _target(rows, config)
    columns = plan_columns(rows, training.unlabeled, config, sensors, settings.max_missing_rate)
    cv_plan = plan_cv(rows, y, training.unlabeled, config)
    x = to_matrix(rows, columns.candidates)
    x_unlabeled = to_matrix(training.unlabeled, columns.candidates)

    study = _search(config, x, y, x_unlabeled, cv_plan)
    best = study.best_trial
    params = {name: value for name, value in best.params.items() if name != K_PARAM}
    preprocess = fit_preprocess(config, np.vstack([x, x_unlabeled]))
    pipeline = build_pipeline(config, params, best.params[K_PARAM], preprocess, columns.candidates)
    pipeline.fit(rows, y)
    keep = pipeline.named_steps["select"].get_support()
    fold_scores = best.user_attrs["fold_scores"]

    return Result(
        pipeline=pipeline,
        config=config,
        data=DataSummary(
            path=path,
            n_rows=frame.height,
            n_columns=frame.width,
            n_rows_used=rows.height,
            n_unlabeled_rows_used=training.unlabeled.height,
            n_rows_without_target=training.n_without_target,
            n_rows_with_infinite=training.n_with_infinite,
            infinite_by_column=training.infinite_by_column,
        ),
        sensors=sensors,
        settings=settings,
        columns=columns,
        selected_columns=[name for name, kept in zip(columns.candidates, keep, strict=True) if kept],
        best_params=best.params,
        cv_scores={"folds": fold_scores, "mean": float(np.mean(fold_scores)), "std": float(np.std(fold_scores))},
        cv=cv_plan.describe(config),
        hpo=_describe_search(study, config),
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
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
    config: Config, x: np.ndarray, y: np.ndarray, x_unlabeled: np.ndarray, cv_plan: CVPlan
) -> optuna.Study:
    n_candidates = x.shape[1]
    metric = config.metric_spec
    scorer = get_scorer(metric.scorer)
    # The preprocessing uses no target, so each fold fits it once for every trial.
    preprocesses = [
        fit_preprocess(config, np.vstack([x[train], x_unlabeled[unlabeled]]))
        for (train, _), unlabeled in zip(cv_plan.folds, cv_plan.unlabeled_folds, strict=True)
    ]

    def objective(trial: optuna.Trial) -> float:
        params = config.model_spec.suggest(trial, n_candidates)
        k = trial.suggest_int(K_PARAM, 1, n_candidates, log=True)
        fold_scores = []
        for (train, test), preprocess in zip(cv_plan.folds, preprocesses, strict=True):
            pipeline = build_pipeline(config, params, k, preprocess)
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
