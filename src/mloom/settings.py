"""Reading the settings: ``config/mloom.toml`` and its defaults.

A key is read from the first of these files that sets it (see "設定ファイル"
in ``docs/spec.md``):

1. the project file: ``config/mloom.toml`` of the repository that uses mloom
   (the current directory of the run), or the file the config names;
2. ``config/mloom.local.toml`` of the mloom repository (not tracked by git),
   read only when mloom runs from a clone of its repository;
3. ``config/mloom.toml`` of the mloom repository: the defaults, which set every
   key. Without a clone, the copy shipped in the wheel is read.

An unknown section or key, or an invalid value, is an error in every file, so
that a typo does not silently keep the default.
"""

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from mloom.errors import InputError, MloomError

SETTINGS_FILE = "mloom.toml"
LOCAL_SETTINGS_FILE = "mloom.local.toml"
# Where the wheel ships config/mloom.toml (``force-include`` in pyproject.toml).
PACKAGED_DEFAULTS = "defaults.toml"


def _rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"must be a number from 0 to 1; got {value!r}")
    if not 0 <= value <= 1:
        raise ValueError(f"must be a number from 0 to 1; got {value!r}")
    return float(value)


def _flag(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"must be true or false; got {value!r}")
    return value


# Table → key → check that returns the value. Keys are unique across tables.
_KEYS: dict[str, dict[str, Callable[[object], Any]]] = {
    "columns": {"max_missing_rate": _rate},
    "preprocessing": {"use_unlabeled_rows": _flag},
}


@dataclass(frozen=True)
class SettingsFile:
    """A file the settings were looked up in.

    Parameters
    ----------
    role : str
        ``"project"``, ``"local_defaults"`` or ``"defaults"``.
    path : str
        Path of the file.
    found : bool
        Whether the file existed.
    """

    role: str
    path: str
    found: bool


@dataclass(frozen=True)
class Settings:
    """The settings of a run.

    Parameters
    ----------
    max_missing_rate : float
        A feature column whose missing rate is above this is excluded
        (``[columns] max_missing_rate``).
    use_unlabeled_rows : bool
        Whether rows without a target are used for the missing rates and the
        imputation and scaling statistics (``[preprocessing]
        use_unlabeled_rows``).
    sources : dict[str, str]
        Path of the file each key was read from.
    files : list[SettingsFile]
        The files looked up, from the highest priority.
    """

    max_missing_rate: float
    use_unlabeled_rows: bool
    sources: dict[str, str]
    files: list[SettingsFile]

    def to_dict(self) -> dict[str, Any]:
        """Describe the settings for ``run.json``.

        Returns
        -------
        dict[str, Any]
            The values, where each came from, and the files looked up.
        """
        return {
            "max_missing_rate": self.max_missing_rate,
            "use_unlabeled_rows": self.use_unlabeled_rows,
            "sources": self.sources,
            "files": [vars(file) for file in self.files],
        }


def mloom_config_dir() -> Path | None:
    """Find ``config/`` of the mloom repository when mloom runs from a clone.

    Returns
    -------
    Path | None
        The directory, or ``None`` when mloom is installed from a wheel.
    """
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    config = root / "config"
    if package == root / "src" / "mloom" and (config / SETTINGS_FILE).is_file():
        return config
    return None


def load_settings(path: Path | None) -> Settings:
    """Read the settings from the project file and the defaults of mloom.

    Parameters
    ----------
    path : Path | None
        The project file. ``None`` reads only the defaults of mloom.

    Returns
    -------
    Settings
        The settings.

    Raises
    ------
    InputError
        When a file is not valid TOML, or has an unknown section or key, or an
        invalid value.
    MloomError
        When the defaults of mloom are missing or do not set every key.
    """
    config_dir = mloom_config_dir()
    layers: list[tuple[str, Path | Traversable | None]] = [
        ("project", path),
        ("local_defaults", config_dir / LOCAL_SETTINGS_FILE if config_dir else None),
        ("defaults", config_dir / SETTINGS_FILE if config_dir else files("mloom") / PACKAGED_DEFAULTS),
    ]
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    looked_up: list[SettingsFile] = []
    read: dict[str, Any] = {}
    for role, file in layers:
        if file is None:
            continue
        found = file.is_file()
        looked_up.append(SettingsFile(role=role, path=str(file), found=found))
        read = _read(file) if found else {}
        for key, value in read.items():
            if key not in values:
                values[key] = value
                sources[key] = str(file)
    # The last file read is the defaults; it alone must set every key.
    _check_defaults(looked_up[-1], set(read))
    return Settings(**values, sources=sources, files=looked_up)


def _read(file: Path | Traversable) -> dict[str, Any]:
    try:
        content = tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise InputError(f"{file} is not valid TOML: {error}") from error
    _check_unknown(content, set(_KEYS), str(file))
    values: dict[str, Any] = {}
    for table_name, checks in _KEYS.items():
        table = content.get(table_name, {})
        if not isinstance(table, dict):
            raise InputError(f"[{table_name}] in {file} must be a table")
        _check_unknown(table, set(checks), f"[{table_name}] of {file}")
        for key, check in checks.items():
            if key in table:
                try:
                    values[key] = check(table[key])
                except (TypeError, ValueError) as error:
                    raise InputError(f"{key} in {file} {error}") from error
    return values


def _check_unknown(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise InputError(f"unknown key in {where}: {', '.join(unknown)}")


def _check_defaults(defaults: SettingsFile, keys: set[str]) -> None:
    if not defaults.found:
        raise MloomError(f"the defaults of mloom are missing: {defaults.path}")
    missing = sorted({key for checks in _KEYS.values() for key in checks} - keys)
    if missing:
        raise MloomError(f"the defaults of mloom do not set {', '.join(missing)}: {defaults.path}")
