"""The config of a run (``Config``) and its checks that need no data."""

from dataclasses import dataclass, field
from pathlib import Path

from mloom.errors import InputError
from mloom.metrics import DEFAULT_METRICS, METRICS, Metric
from mloom.models import MODELS, ModelSpec

TASKS: tuple[str, ...] = ("regression", "classification")
CV_METHODS: tuple[str, ...] = ("timeseries", "group", "kfold")
DEFAULT_UNAVAILABLE_SENSORS = Path("config/unavailable-sensors.json")
DEFAULT_SETTINGS = Path("config/mloom.toml")


@dataclass(frozen=True)
class Config:
    """What to optimize and how (see "設定" in ``docs/spec.md``).

    Parameters
    ----------
    target : str
        Name of the target column.
    task : str
        ``"regression"`` or ``"classification"``.
    model : str
        Name of a model of the registry (``mloom.models.MODELS``).
    cv : str
        ``"timeseries"``, ``"group"`` or ``"kfold"``.
    n_splits : int
        Number of CV folds.
    time_column : str | None
        Column that orders the rows; required when ``cv`` is ``"timeseries"``.
    group_column : str | None
        Column of the groups; required when ``cv`` is ``"group"``.
    metric : str | None
        Evaluation metric. ``None`` uses the default of the task.
    exclude_columns : tuple[str, ...]
        Columns the user excludes from the features.
    unavailable_sensors : Path | None
        JSON file of the sensors that must not be used, relative to the
        current directory. ``None`` reads no file.
    settings : Path | None
        Project settings file (``mloom.settings``), relative to the
        current directory. ``None`` reads only the defaults of mloom.
    n_trials : int
        Number of trials of the hyperparameter search.
    timeout : float | None
        Seconds after which the search stops starting new trials.
    seed : int
        Seed of the CV split, the sampler, and the models.
    """

    target: str
    task: str
    model: str
    cv: str
    n_splits: int = 5
    time_column: str | None = None
    group_column: str | None = None
    metric: str | None = None
    exclude_columns: tuple[str, ...] = ()
    unavailable_sensors: Path | None = field(default=DEFAULT_UNAVAILABLE_SENSORS)
    settings: Path | None = field(default=DEFAULT_SETTINGS)
    n_trials: int = 100
    timeout: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """Check the values that can be checked without the data.

        Raises
        ------
        InputError
            When a value is invalid.
        """
        _check_choice("task", self.task, TASKS)
        _check_choice("cv", self.cv, CV_METHODS)
        _check_choice("model", self.model, tuple(MODELS))
        if self.task not in self.model_spec.builders:
            raise InputError(f"model {self.model!r} does not support task {self.task!r}")
        _check_choice("metric", self.metric_name, tuple(METRICS[self.task]))
        if self.cv == "timeseries" and self.time_column is None:
            raise InputError("cv 'timeseries' needs time_column")
        if self.cv == "group" and self.group_column is None:
            raise InputError("cv 'group' needs group_column")
        if self.n_splits < 2:
            raise InputError(f"n_splits must be 2 or more, got {self.n_splits}")
        if self.n_trials < 1:
            raise InputError(f"n_trials must be 1 or more, got {self.n_trials}")

    @property
    def metric_name(self) -> str:
        """The metric, with the default of the task filled in.

        Returns
        -------
        str
            Name of the metric.
        """
        return self.metric if self.metric is not None else DEFAULT_METRICS[self.task]

    @property
    def metric_spec(self) -> Metric:
        """The metric as an object.

        Returns
        -------
        Metric
            The metric of ``metric_name``.
        """
        return METRICS[self.task][self.metric_name]

    @property
    def model_spec(self) -> ModelSpec:
        """The model of the registry.

        Returns
        -------
        ModelSpec
            The model of ``model``.
        """
        return MODELS[self.model]

    @property
    def role_columns(self) -> tuple[str, ...]:
        """Columns that drive the CV split and are never features.

        Returns
        -------
        tuple[str, ...]
            ``time_column`` and ``group_column`` when given.
        """
        return tuple(c for c in (self.time_column, self.group_column) if c is not None)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise InputError(f"{name} must be one of {', '.join(choices)}; got {value!r}")
