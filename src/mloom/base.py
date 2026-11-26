"""The base model of ``mloom refit``: reading how it was made, and keeping it from being overwritten.

See "ベースのモデル" in ``docs/spec.md``. Only the record of the run (the first
pickle of ``model.pkl``) is read, so that a base whose model can no longer be
read (with another version of scikit-learn, for example) can still be
refitted. A record made by another version of mloom is read when it has every
key the base needs, with valid values; an older form is never converted
(``AGENTS.md``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mloom.config import Config
from mloom.errors import InputError
from mloom.model_file import MODEL_FILE, read_run
from mloom.settings import Settings, settings_from_record

_NONE = type(None)


@dataclass(frozen=True)
class Base:
    """A base ``model.pkl``: how its model was made.

    Parameters
    ----------
    path : str | Path
        The path of ``model.pkl``, as it was given.
    record : dict[str, Any]
        The record of the run read from it.
    config : Config
        The config of the run, from the record. ``unavailable_sensors`` and
        ``settings`` are the defaults of ``Config``: the record does not hold
        them, and ``refit_model`` takes them as its own arguments.
    settings : Settings
        The settings recorded (``mloom.settings.settings_from_record``).
    selected_columns : list[str]
        The columns the model of the base uses.
    best_params : dict[str, Any]
        The hyperparameters of the model of the base, with ``k``.
    """

    path: str | Path
    record: dict[str, Any]
    config: Config
    settings: Settings
    selected_columns: list[str]
    best_params: dict[str, Any]


def load_base(path: str | Path) -> Base:
    """Read how the model of a ``model.pkl`` was made, from its record only.

    A pickle is only to be read from a trusted source.

    Parameters
    ----------
    path : str | Path
        Path of a ``model.pkl`` saved by ``mloom optimize`` or ``mloom refit``.

    Returns
    -------
    Base
        The base.

    Raises
    ------
    InputError
        When the file does not exist or does not start with a record
        (``mloom.read_run``), or a key the base needs is missing or invalid;
        then the message names the key and the version of mloom that made
        the record.
    """
    record = read_run(path)
    try:
        return _base(path, record)
    except InputError as error:
        raise InputError(
            f"{path} cannot be a base: {error} (in the record made by mloom {record['mloom_version']}); "
            "make the base again with mloom optimize"
        ) from error


def _base(path: str | Path, record: dict[str, Any]) -> Base:
    # A refit records no search (hpo is null); its n_trials and timeout are then the defaults, never used.
    search = (
        {}
        if _value(record, "hpo", (dict, _NONE)) is None
        else {
            "n_trials": _value(record, "hpo.n_trials", (int,)),
            "timeout": _value(record, "hpo.timeout", (int, float, _NONE)),
        }
    )
    config = Config(
        target=_value(record, "target", (str,)),
        task=_value(record, "task", (str,)),
        model=_value(record, "model", (str,)),
        cv=_value(record, "cv.method", (str,)),
        n_splits=_value(record, "cv.n_splits", (int,)),
        time_column=_value(record, "cv.time_column", (str, _NONE)),
        group_column=_value(record, "cv.group_column", (str, _NONE)),
        metric=_value(record, "metric", (str,)),
        exclude_columns=tuple(_names(record, "exclude_columns")),
        filters=tuple(_value(record, "filters", (list,))),
        train_filters=tuple(_value(record, "train_filters", (list,))),
        seed=_value(record, "seed", (int,)),
        **search,
    )
    # Copied to the record of the refit.
    _value(record, "created_at", (str,))
    _value(record, "cv_scores", (dict,))
    return Base(
        path=path,
        record=record,
        config=config,
        settings=settings_from_record(_value(record, "settings", (dict,)), str(path)),
        selected_columns=_names(record, "selected_columns"),
        best_params=_value(record, "best_params", (dict,)),
    )


def _value(record: dict[str, Any], key: str, kinds: tuple[type, ...]) -> Any:  # noqa: ANN401 - one of kinds
    value: Any = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise InputError(f"{key} is missing")
        value = value[part]
    # bool is an int in Python, but never a valid number of the record.
    if not isinstance(value, kinds) or (isinstance(value, bool) and bool not in kinds):
        names = " or ".join("null" if kind is _NONE else kind.__name__ for kind in kinds)
        raise InputError(f"{key} must be {names}; got {value!r}")
    return value


def _names(record: dict[str, Any], key: str) -> list[str]:
    names = _value(record, key, (list,))
    if not all(isinstance(name, str) for name in names):
        raise InputError(f"{key} must be a list of column names; got {names!r}")
    return names


def check_output(output_dir: str | Path, base: str | Path) -> None:
    """Check that saving to a directory does not overwrite the base ``model.pkl``.

    Parameters
    ----------
    output_dir : str | Path
        The directory ``model.pkl`` is to be saved in.
    base : str | Path
        Path of the base ``model.pkl``.

    Raises
    ------
    InputError
        When ``model.pkl`` of ``output_dir`` is the base: the same path once
        resolved, or the same existing file.
    """
    output = Path(output_dir) / MODEL_FILE
    base = Path(base)
    same_path = output.resolve() == base.resolve()
    if same_path or (output.is_file() and base.is_file() and output.samefile(base)):
        raise InputError(f"saving to {output_dir} would overwrite the base {base}; save to another directory")
