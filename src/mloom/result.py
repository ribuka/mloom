"""The result of a run and how it is saved (``model.pkl`` and ``run.json``)."""

import json
import pickle
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from mloom.columns import ColumnPlan
from mloom.config import Config
from mloom.sensors import UnavailableSensors
from mloom.settings import Settings

MODEL_FILE = "model.pkl"
RUN_FILE = "run.json"
_PACKAGES = ("mloom", "scikit-learn", "optuna", "polars", "numpy", "scipy", "lightgbm")


@dataclass(frozen=True)
class DataSummary:
    """The input data, as recorded in ``run.json``.

    Parameters
    ----------
    path : str | None
        Path of the Parquet file, or ``None`` for a DataFrame.
    n_rows : int
        Rows of the input.
    n_columns : int
        Columns of the input.
    n_rows_used : int
        Rows used for training.
    n_rows_without_target : int
        Rows dropped because the target is missing.
    n_rows_with_infinite : int
        Rows dropped because of +inf or -inf.
    infinite_by_column : dict[str, int]
        Rows with +inf or -inf in each column (``mloom.rows.TrainingRows``).
    """

    path: str | None
    n_rows: int
    n_columns: int
    n_rows_used: int
    n_rows_without_target: int
    n_rows_with_infinite: int
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
    columns : ColumnPlan
        The candidate and excluded columns.
    selected_columns : list[str]
        The candidate columns the fitted pipeline uses.
    best_params : dict[str, Any]
        Best hyperparameters, including ``k`` of the feature selection.
    cv_scores : dict[str, Any]
        Metric of each fold of the best trial, with its mean and std.
    cv : dict[str, Any]
        Description of the CV split.
    hpo : dict[str, Any]
        Description of the search.
    created_at : str
        ISO 8601 time the run finished.
    """

    pipeline: Pipeline
    config: Config
    data: DataSummary
    sensors: UnavailableSensors
    settings: Settings
    columns: ColumnPlan
    selected_columns: list[str]
    best_params: dict[str, Any]
    cv_scores: dict[str, Any]
    cv: dict[str, Any]
    hpo: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Build the content of ``run.json``.

        Returns
        -------
        dict[str, Any]
            JSON-serializable record of the run.
        """
        config = self.config
        return {
            "mloom_version": version("mloom"),
            "created_at": self.created_at,
            "data": vars(self.data),
            "target": config.target,
            "task": config.task,
            "model": config.model,
            "metric": config.metric_name,
            "metric_direction": config.metric_spec.direction,
            "seed": config.seed,
            "unavailable_sensors": {
                "path": self.sensors.path,
                "found": self.sensors.found,
                "sensors": self.sensors.reasons,
                "not_in_data": self.columns.sensors_not_in_data,
            },
            "settings": vars(self.settings),
            "excluded_columns": [vars(column) for column in self.columns.excluded],
            "candidate_columns": self.columns.candidates,
            "selected_columns": self.selected_columns,
            "cv": self.cv,
            "hpo": self.hpo,
            "best_params": self.best_params,
            "cv_scores": self.cv_scores,
            "versions": _versions(),
        }

    def save(self, output_dir: str | Path) -> Path:
        """Save ``model.pkl`` and ``run.json``.

        Parameters
        ----------
        output_dir : str | Path
            Directory to write to; created when missing. Existing files are
            overwritten.

        Returns
        -------
        Path
            Path of ``run.json``.
        """
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / MODEL_FILE).open("wb") as file:
            pickle.dump(self.pipeline, file)
        run_path = directory / RUN_FILE
        run_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return run_path


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions
