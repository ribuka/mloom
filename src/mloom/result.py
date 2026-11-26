"""The result of a run: its record and its model, saved together as ``model.pkl``.

The record holds plain Python values only, and the model a dict of plain
Python values, numpy arrays and scikit-learn parts only, so that both can be
read without mloom (see "``model.pkl``" in ``docs/spec.md``; the format of the
file is ``mloom.model_file``).
"""

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from mloom.base import check_output
from mloom.columns import ColumnPlan
from mloom.config import Config
from mloom.model_file import write_model_file
from mloom.row_hash import hash_columns
from mloom.sensors import UnavailableSensors
from mloom.settings import Settings
from mloom.training_period import TimeValue

_PACKAGES = ("mloom", "scikit-learn", "lightgbm", "optuna", "polars", "numpy", "scipy")


@dataclass(frozen=True)
class DataSummary:
    """The input data, as recorded in the record of the run.

    Parameters
    ----------
    path : str | None
        Path of the Parquet file, or ``None`` for a DataFrame.
    n_rows : int
        Rows of the input.
    n_columns : int
        Columns of the input.
    n_rows_filtered_out : int
        Rows removed by ``filters``.
    n_rows_train_filtered_out : int
        Rows removed by ``train_filters``, among those left by ``filters``.
    n_rows_used : int
        Rows used for training.
    n_unlabeled_rows_used : int
        Rows without a target used for the statistics of the features.
    n_rows_without_target : int
        Rows whose target is missing (null or NaN), among those left by the
        filters. They are not trained on; some may join the statistics.
    n_rows_with_infinite_target : int
        Rows dropped because their target is +inf or -inf.
    infinite_by_column : dict[str, int]
        Rows with +inf or -inf in each column (``mloom.rows.TrainingRows``).
    """

    path: str | None
    n_rows: int
    n_columns: int
    n_rows_filtered_out: int
    n_rows_train_filtered_out: int
    n_rows_used: int
    n_unlabeled_rows_used: int
    n_rows_without_target: int
    n_rows_with_infinite_target: int
    infinite_by_column: dict[str, int]


@dataclass(frozen=True)
class Result:
    """A fitted pipeline and the record of how it was found.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline fitted on all training rows with the best parameters. It
        takes a table with the candidate columns and ignores other columns.
    config : Config
        The config of the run.
    data : DataSummary
        The input data.
    sensors : UnavailableSensors
        The unavailable sensors that were read.
    settings : Settings
        The settings that were used.
    clip : dict[str, Any]
        The clipping of the fitted pipeline (``mloom.clip.describe_clip``).
    impute : dict[str, Any]
        The imputation of the fitted pipeline: ``{"method": <method>}``, with
        ``"auto"`` resolved for the model (``Preprocessing.impute``).
    columns : ColumnPlan
        The candidate and excluded columns.
    selection : dict[str, Any] | None
        The clusters of similar candidate columns and their representatives
        in the fitted pipeline (``mloom.selection.describe_selection``);
        ``None`` for a refit, which selects no representative.
    selected_columns : list[str]
        The candidate columns the fitted pipeline uses.
    best_params : dict[str, Any]
        Best hyperparameters, including ``k`` of the feature selection.
    cv_scores : dict[str, Any]
        Metric of each fold of the best trial, with its mean and std.
    interpretation : dict[str, Any]
        ``permutation_importance`` of each group of similar columns on the
        test side of the CV folds (``mloom.interpretation.permutation_importance``),
        ``coefficients`` of a linear model (``linear_coefficients``) and
        ``vip`` of PLS (``pls_vip``); each is ``None`` when it does not
        apply.
    cv : dict[str, Any]
        Description of the CV split.
    hpo : dict[str, Any] | None
        Description of the search; ``None`` for a refit, which searches
        nothing.
    created_at : str
        ISO 8601 time the run finished.
    training_hashes : np.ndarray
        Sorted unique uint64 hashes (``mloom.row_hash``) of the rows the
        pipeline was fitted on, by the columns of
        ``mloom.row_hash.hash_columns``.
    last_time : TimeValue | None
        The latest ``time_column`` of the rows the pipeline was fitted on
        (``mloom.training_period.last_time``); ``None`` without
        ``time_column``.
    shap_background : np.ndarray
        Mean of the input of the estimator on the rows the pipeline was
        fitted on (``mloom.explain.shap_background``).
    base : dict[str, Any] | None
        The base of a refit (``mloom.refit.refit_model``) or of
        ``optimize_model``: its ``path``, the ``mode``, its ``created_at``,
        ``mloom_version`` and ``cv_scores``, and the ``dropped_columns``
        (``mloom.base.describe_base``); ``optimize_model`` adds
        ``enqueued``, ``cv_scores_on_this_data``, ``comparison`` and
        ``not_compared``. ``None`` without a base.
    """

    pipeline: Pipeline
    config: Config
    data: DataSummary
    sensors: UnavailableSensors
    settings: Settings
    clip: dict[str, Any]
    impute: dict[str, Any]
    columns: ColumnPlan
    selection: dict[str, Any] | None
    selected_columns: list[str]
    best_params: dict[str, Any]
    cv_scores: dict[str, Any]
    interpretation: dict[str, Any]
    cv: dict[str, Any]
    hpo: dict[str, Any] | None
    created_at: str
    training_hashes: np.ndarray
    last_time: TimeValue | None
    shap_background: np.ndarray
    base: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Build the record of the run, the first pickle of ``model.pkl``.

        Returns
        -------
        dict[str, Any]
            JSON-serializable record of the run (see "記録" in ``docs/spec.md``).
        """
        config = self.config
        return {
            "mloom_version": version("mloom"),
            "base": self.base,
            "created_at": self.created_at,
            "data": vars(self.data),
            "target": config.target,
            "task": config.task,
            "model": config.model,
            "metric": config.metric_name,
            "metric_direction": config.metric_spec.direction,
            "seed": config.seed,
            "exclude_columns": list(config.exclude_columns),
            "filters": [dict(condition) for condition in config.filters],
            "train_filters": [dict(condition) for condition in config.train_filters],
            "unavailable_sensors": {
                "path": self.sensors.path,
                "found": self.sensors.found,
                "sensors": self.sensors.reasons,
                "not_in_data": self.columns.sensors_not_in_data,
            },
            "settings": self.settings.to_dict(),
            "preprocessing": {"clip": self.clip, "impute": self.impute},
            "excluded_columns": [vars(column) for column in self.columns.excluded],
            "candidate_columns": self.columns.candidates,
            "selection": self.selection,
            "selected_columns": self.selected_columns,
            "cv": self.cv,
            "hpo": self.hpo,
            "best_params": self.best_params,
            "cv_scores": self.cv_scores,
            "interpretation": self.interpretation,
            "versions": _versions(),
        }

    def model_bundle(self) -> dict[str, Any]:
        """Build the model, the second pickle of ``model.pkl``.

        Returns
        -------
        dict[str, Any]
            ``pipeline``, ``filters``, ``train_filters``, ``training_rows`` and
            ``shap_background`` (see "``model.pkl``" in ``docs/spec.md``).
        """
        config = self.config
        return {
            "pipeline": self.pipeline,
            "filters": [dict(condition) for condition in config.filters],
            "train_filters": [dict(condition) for condition in config.train_filters],
            "training_rows": {
                "columns": hash_columns(self.columns.candidates, config),
                "target": config.target,
                "time_column": config.time_column,
                "group_column": config.group_column,
                "hashes": self.training_hashes,
                "last_time": self.last_time,
            },
            "shap_background": self.shap_background,
        }

    def save(self, output_dir: str | Path) -> Path:
        """Save the record and the model as ``model.pkl``.

        Parameters
        ----------
        output_dir : str | Path
            Directory to write to; created when missing. An existing
            ``model.pkl`` is replaced only once the new one is fully written.

        Returns
        -------
        Path
            Path of ``model.pkl``.

        Raises
        ------
        InputError
            When the result has a base and ``model.pkl`` of ``output_dir``
            is the base (``mloom.base.check_output``); nothing is written.
        """
        if self.base is not None:
            check_output(output_dir, self.base["path"])
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return write_model_file(directory, self.to_dict(), self.model_bundle())


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions
