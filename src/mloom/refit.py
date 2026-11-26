"""``refit_model``: fitting the model of a base again on new data, with its columns and parameters fixed.

See "ベースのモデル" in ``docs/spec.md``. The rows, the candidate columns and
the CV folds are prepared as ``optimize_model`` prepares them
(``mloom.training``), with the config of the base; nothing is searched.
"""

import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

from sklearn.metrics import get_scorer

from mloom.base import Base
from mloom.columns import require_candidates
from mloom.config import DEFAULT_UNAVAILABLE_SENSORS
from mloom.data import Data
from mloom.errors import InputError
from mloom.filters import Condition
from mloom.interpretation import FoldModels, fit_fold
from mloom.optimize import K_PARAM
from mloom.pipeline import fit_preprocess
from mloom.result import Result
from mloom.settings import load_settings
from mloom.training import FinalModel, TrainingData, fit_final, prepare_training

MODE = "refit"
NOT_IN_DATA = "not_in_data"


def refit_model(
    data: Data,
    base: Base,
    *,
    filters: tuple[Condition, ...] | None = None,
    train_filters: tuple[Condition, ...] | None = None,
    unavailable_sensors: Path | None = DEFAULT_UNAVAILABLE_SENSORS,
    settings: Path | None = None,
) -> Result:
    """Fit the columns and the parameters of a base on new data, without a search.

    The columns of the base that are no longer candidates are dropped, each
    with a warning (``warnings.warn``) and its reason in ``base.dropped_columns``
    of the record.

    Parameters
    ----------
    data : Data
        Path of a Parquet file, or a polars DataFrame.
    base : Base
        The base (``mloom.load_base``).
    filters : tuple[Condition, ...] | None
        Conditions on the rows of training and prediction; ``None`` keeps
        those of the base.
    train_filters : tuple[Condition, ...] | None
        Conditions on the rows of training only; ``None`` keeps those of the
        base.
    unavailable_sensors : Path | None
        JSON file of the sensors that must not be used now; ``None`` reads no
        file. That of the base is not used, since a sensor may have broken
        since.
    settings : Path | None
        Project settings file, read as ``optimize_model`` reads it; ``None``
        uses the settings recorded in the base.

    Returns
    -------
    Result
        The fitted pipeline and the record of the run, with ``hpo`` and
        ``selection`` ``None`` and ``base`` set; ``Result.save`` refuses to
        overwrite the base.

    Raises
    ------
    InputError
        When the data, the filters, ``unavailable-sensors.json`` or
        ``mloom.toml`` is invalid or does not fit the config of the base, or
        no column of the base is left as a candidate. Raised before any
        model is fitted.
    """
    config = replace(
        base.config,
        filters=base.config.filters if filters is None else filters,
        train_filters=base.config.train_filters if train_filters is None else train_filters,
        unavailable_sensors=unavailable_sensors,
        settings=settings,
    )
    run_settings = base.settings if settings is None else load_settings(settings)
    training = prepare_training(data, config, run_settings, require_candidates)
    keep, dropped = _keep(base, training)
    params = {name: value for name, value in base.best_params.items() if name != K_PARAM}
    cv_plan = training.cv_plan
    fold_models = FoldModels(
        config,
        params,
        cv_plan.folds,
        [
            (fit_preprocess(training.preprocessing, training.fold_statistics(train, unlabeled)), keep)
            for (train, _), unlabeled in zip(cv_plan.folds, cv_plan.unlabeled_folds, strict=True)
        ],
    )
    return fit_final(
        config,
        training,
        FinalModel(
            preprocess=fit_preprocess(training.preprocessing, training.x_statistics),
            keep=keep,
            fold_models=fold_models,
            # No cluster is formed, so each column is permuted alone.
            importance_groups=[(i, [i]) for i in keep],
        ),
        _fold_scores(fold_models, training),
        selection=None,
        best_params={**base.best_params, K_PARAM: len(keep)},
        hpo=None,
        base={
            "path": str(base.path),
            "mode": MODE,
            "created_at": base.record["created_at"],
            "mloom_version": base.record["mloom_version"],
            "cv_scores": base.record["cv_scores"],
            "dropped_columns": dropped,
        },
    )


def _keep(base: Base, training: TrainingData) -> tuple[list[int], list[dict[str, Any]]]:
    # The columns of the base that are still candidates, in the order of the base, and why the others are not.
    position = {name: i for i, name in enumerate(training.columns.candidates)}
    reasons = {column.name: column.reason for column in training.columns.excluded}
    keep: list[int] = []
    dropped: list[dict[str, Any]] = []
    for name in base.selected_columns:
        if name in position:
            keep.append(position[name])
        else:
            dropped.append({"name": name, "reason": reasons.get(name, NOT_IN_DATA)})
    if not keep:
        described = ", ".join(f"{column['name']} ({column['reason']})" for column in dropped)
        raise InputError(
            f"no column the base selected is left as a candidate: {described}; "
            "make the model again with mloom optimize"
        )
    for column in dropped:
        warnings.warn(
            f"column {column['name']!r} of the base is dropped: {column['reason']}", UserWarning, stacklevel=3
        )
    return keep, dropped


def _fold_scores(models: FoldModels, training: TrainingData) -> list[float]:
    # Each fold fits the fixed model on its training side; a failure is not expected, so it is raised.
    metric = models.config.metric_spec
    scorer = get_scorer(metric.scorer)
    scores = []
    for fold, (_, test) in enumerate(models.folds):
        pipeline = fit_fold(models, fold, training.x, training.y)
        scores.append(metric.report_value(float(scorer(pipeline, training.x[test], training.y[test]))))
    return scores
