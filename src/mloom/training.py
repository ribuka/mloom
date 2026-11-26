"""The steps a run shares between ``optimize_model`` and ``refit_model``.

Both prepare the data they train on the same way (the row filters, the rows,
the feature candidates and the CV folds; see "処理の流れ" in
``docs/spec.md``), and both end with the same final fit and the same record.
Only what lies between differs: the search of ``optimize_model``, and the
fixed columns and parameters of ``refit_model``.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from mloom.clip import describe_clip
from mloom.columns import ColumnPlan, check_columns, check_role_values, plan_columns
from mloom.config import Config
from mloom.cv import CVPlan, order_rows, plan_cv
from mloom.data import Data, load_data
from mloom.errors import InputError
from mloom.explain import shap_background
from mloom.filters import apply_filters
from mloom.interpretation import (
    FoldModels,
    linear_coefficients,
    permutation_importance,
    pls_vip,
)
from mloom.metrics import BINARY_ONLY_METRICS
from mloom.pipeline import (
    Preprocessing,
    as_float,
    build_pipeline,
    fitted_clip_bounds,
    plan_preprocessing,
    to_matrix,
)
from mloom.result import DataSummary, Result
from mloom.row_hash import check_hashable, hash_columns, hash_rows
from mloom.rows import select_rows
from mloom.sensors import UnavailableSensors, load_unavailable_sensors
from mloom.settings import Settings
from mloom.training_period import last_time


@dataclass(frozen=True)
class TrainingData:
    """The data a run trains on, before any model is fitted.

    Parameters
    ----------
    summary : DataSummary
        The input data, as recorded.
    sensors : UnavailableSensors
        The unavailable sensors that were read.
    settings : Settings
        The settings of the run.
    preprocessing : Preprocessing
        The preprocessing of the run.
    rows : pl.DataFrame
        The labeled rows, in the order of the CV split.
    y : np.ndarray
        The target of ``rows``.
    columns : ColumnPlan
        The candidate and excluded columns.
    x : np.ndarray
        The candidate columns of ``rows`` (``to_matrix``).
    x_unlabeled : np.ndarray
        The candidate columns of the unlabeled rows used for the statistics.
    x_statistics : np.ndarray
        ``x`` and ``x_unlabeled`` together: the rows the statistics of the
        final model come from.
    cv_plan : CVPlan
        The CV folds.
    """

    summary: DataSummary
    sensors: UnavailableSensors
    settings: Settings
    preprocessing: Preprocessing
    rows: pl.DataFrame
    y: np.ndarray
    columns: ColumnPlan
    x: np.ndarray
    x_unlabeled: np.ndarray
    x_statistics: np.ndarray
    cv_plan: CVPlan

    def fold_statistics(self, train: np.ndarray, unlabeled: np.ndarray) -> np.ndarray:
        """Gather the rows the statistics of a CV fold come from.

        Parameters
        ----------
        train : np.ndarray
            Indices of the labeled rows on the training side of the fold.
        unlabeled : np.ndarray
            Indices of the unlabeled rows on the training side of the fold.

        Returns
        -------
        np.ndarray
            The candidate columns of those rows, labeled rows first.
        """
        return np.vstack([self.x[train], self.x_unlabeled[unlabeled]])


def prepare_training(data: Data, config: Config, settings: Settings) -> TrainingData:
    """Read the data and apply the filters, the selection of the rows, the exclusion of columns and the CV split.

    Parameters
    ----------
    data : Data
        Path of a Parquet file, or a polars DataFrame.
    config : Config
        The config of the run; its ``unavailable_sensors`` is read here.
    settings : Settings
        The settings of the run.

    Returns
    -------
    TrainingData
        The data the run trains on.

    Raises
    ------
    InputError
        When the data or ``unavailable-sensors.json`` is invalid or does not
        fit the config, including the row filters and the imputation the
        model cannot do without, or no row or no candidate column is left.
    """
    frame, path = load_data(data)
    sensors = load_unavailable_sensors(config.unavailable_sensors)
    preprocessing = plan_preprocessing(config, settings)
    check_columns(frame, config, sensors)
    filtered = apply_filters(frame, config.filters, "filters")
    trainable = apply_filters(filtered, config.train_filters, "train_filters")
    check_role_values(trainable, config)
    check_hashable(trainable.schema, [config.target, *config.role_columns])
    training = select_rows(
        trainable,
        config,
        sensors,
        use_unlabeled=settings.preprocessing.use_unlabeled_rows,
    )
    rows = order_rows(training.labeled, config)
    y = _target(rows, config)
    columns = plan_columns(rows, training.unlabeled, config, sensors, settings.columns.max_missing_rate)
    x = to_matrix(rows, columns.candidates)
    x_unlabeled = to_matrix(training.unlabeled, columns.candidates)
    return TrainingData(
        summary=DataSummary(
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
        preprocessing=preprocessing,
        rows=rows,
        y=y,
        columns=columns,
        x=x,
        x_unlabeled=x_unlabeled,
        x_statistics=np.vstack([x, x_unlabeled]),
        cv_plan=plan_cv(rows, y, training.unlabeled, config),
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


@dataclass(frozen=True)
class FinalModel:
    """What the final model is fitted with.

    Parameters
    ----------
    preprocess : Pipeline
        The preprocessing fitted on ``TrainingData.x_statistics``.
    keep : list[int]
        Positions, among the candidate columns, of the columns the model
        uses, in the order of the input of the estimator.
    fold_models : FoldModels
        The same model on each CV fold; its ``params`` are those of the
        final model.
    importance_groups : list[tuple[int, list[int]]]
        The groups of columns of the permutation importance
        (``mloom.interpretation.permutation_importance``).
    """

    preprocess: Pipeline
    keep: list[int]
    fold_models: FoldModels
    importance_groups: list[tuple[int, list[int]]]


def fit_final(
    config: Config,
    training: TrainingData,
    final: FinalModel,
    fold_scores: list[float],
    *,
    selection: dict[str, Any],
    best_params: dict[str, Any],
    hpo: dict[str, Any],
) -> Result:
    """Fit the final model on all training rows and build the result of the run.

    Parameters
    ----------
    config : Config
        The config of the run.
    training : TrainingData
        The data the run trains on.
    final : FinalModel
        What the final model is fitted with.
    fold_scores : list[float]
        The metric of each CV fold (``Metric.report_value``).
    selection : dict[str, Any]
        ``Result.selection``.
    best_params : dict[str, Any]
        ``Result.best_params``.
    hpo : dict[str, Any]
        ``Result.hpo``.

    Returns
    -------
    Result
        The fitted pipeline and the record of the run.
    """
    candidates = training.columns.candidates
    pipeline = build_pipeline(config, final.fold_models.params, final.preprocess, final.keep, candidates)
    pipeline.fit(training.rows, training.y)
    background = shap_background(pipeline, training.rows)
    importance = permutation_importance(
        final.fold_models,
        (training.x, training.y),
        final.importance_groups,
        candidates,
        training.settings.importance.n_repeats,
    )
    return Result(
        pipeline=pipeline,
        config=config,
        data=training.summary,
        sensors=training.sensors,
        settings=training.settings,
        clip=describe_clip(
            training.settings.clip.method,
            fitted_clip_bounds(final.preprocess),
            candidates,
        ),
        impute={"method": training.preprocessing.impute},
        columns=training.columns,
        selected_columns=[candidates[i] for i in final.keep],
        cv_scores={
            "folds": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
        },
        interpretation={
            "permutation_importance": importance.record,
            "coefficients": linear_coefficients(pipeline, background),
            "vip": pls_vip(pipeline, training.rows),
        },
        cv=training.cv_plan.describe(config),
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        training_hashes=np.unique(hash_rows(training.rows, hash_columns(candidates, config))),
        last_time=last_time(training.rows, config.time_column),
        shap_background=background,
        selection=selection,
        best_params=best_params,
        hpo=hpo,
    )
