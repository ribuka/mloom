"""``refit_model``: fitting the model of a base again on new data, with its columns and parameters fixed.

See "ベースのモデル" in ``docs/spec.md``. The rows, the candidate columns and
the CV folds are prepared as ``optimize_model`` prepares them
(``mloom.training``), with the config of the base; nothing is searched.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

from mloom.base import Base, base_columns, describe_base, warn_dropped
from mloom.columns import ColumnPlan
from mloom.config import DEFAULT_UNAVAILABLE_SENSORS
from mloom.data import Data
from mloom.errors import InputError
from mloom.filters import Condition
from mloom.fixed_cv import fixed_fold_models, fixed_fold_scores
from mloom.models import K_PARAM
from mloom.pipeline import fit_preprocess
from mloom.result import Result
from mloom.settings import load_settings
from mloom.training import FinalModel, fit_final, prepare_training

MODE = "refit"


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
        settings=base.config.settings if settings is None else settings,
    )
    # Checked as soon as the candidates are decided, so that no column left names every dropped column.
    training = prepare_training(data, config, load_settings(config.settings), lambda plan: _keep(base, plan))
    keep, dropped = _keep(base, training.columns)
    warn_dropped(dropped)
    params = {name: value for name, value in base.best_params.items() if name != K_PARAM}
    fold_models = fixed_fold_models(config, params, training.preprocessing, keep, training)
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
        fixed_fold_scores(fold_models, training),
        selection=None,
        best_params={**base.best_params, K_PARAM: len(keep)},
        hpo=None,
        base=describe_base(base, MODE, dropped),
    )


def _keep(base: Base, plan: ColumnPlan) -> tuple[list[int], list[dict[str, Any]]]:
    keep, dropped = base_columns(base, plan)
    if not keep:
        described = ", ".join(f"{column['name']} ({column['reason']})" for column in dropped)
        raise InputError(
            f"no column the base selected is left as a candidate: {described}; make the model again with mloom optimize"
        )
    return keep, dropped
