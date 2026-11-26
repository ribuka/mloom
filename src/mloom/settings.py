"""Reading the settings: ``config/mloom.toml`` and its defaults.

A key is read from the first of these files that sets it (see "設定ファイル"
in ``docs/spec.md``):

1. the project file: ``config/mloom.toml`` of the repository that uses mloom
   (the current directory of the run), or the file the config names;
2. ``config/mloom.local.toml`` of the mloom repository (not tracked by git),
   read only when mloom runs from a clone of its repository;
3. ``config/mloom.toml`` of the mloom repository: the defaults, which set every
   key. Without a clone, the copy shipped in the wheel is read.

A key is named by its table and its name (``[clip] method`` and ``[impute]
method`` are different keys). An unknown table or key, or an invalid value, is
an error in every file, so that a typo does not silently keep the default.
"""

import math
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from mloom.errors import InputError, MloomError

SETTINGS_FILE = "mloom.toml"
LOCAL_SETTINGS_FILE = "mloom.local.toml"
# Where the wheel ships config/mloom.toml (``force-include`` in pyproject.toml).
PACKAGED_DEFAULTS = "defaults.toml"
CLIP_METHODS: tuple[str, ...] = ("none", "quantile", "iqr", "mad", "sigma")
IMPUTE_METHODS: tuple[str, ...] = ("auto", "none", "median", "mean", "most_frequent", "constant", "knn", "iterative")

Check = Callable[[object], Any]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise TypeError(f"must be a finite number; got {value!r}")
    return float(value)


def _rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        raise ValueError(f"must be a number from 0 to 1; got {value!r}")
    return float(value)


def _positive(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < math.inf:
        raise ValueError(f"must be a positive number; got {value!r}")
    return float(value)


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"must be an integer of 1 or more; got {value!r}")
    return value


def _flag(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"must be true or false; got {value!r}")
    return value


def _one_of(choices: tuple[str, ...]) -> Check:
    def check(value: object) -> str:
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"must be one of {', '.join(choices)}; got {value!r}")
        return value

    return check


@dataclass(frozen=True)
class ColumnSettings:
    """``[columns]``: the exclusion of columns.

    Parameters
    ----------
    max_missing_rate : float
        A feature column whose missing rate is above this is excluded.
    """

    max_missing_rate: float


@dataclass(frozen=True)
class PreprocessingSettings:
    """``[preprocessing]``: the rows the statistics of the features come from.

    Parameters
    ----------
    use_unlabeled_rows : bool
        Whether rows without a target are used for the missing rates and the
        statistics of the clipping, the imputation and the scaling.
    """

    use_unlabeled_rows: bool


@dataclass(frozen=True)
class ClipSettings:
    """``[clip]``: how the features are clipped (see "クリップ" in ``docs/spec.md``).

    Parameters
    ----------
    method : str
        One of ``CLIP_METHODS``.
    lower_quantile : float
        Lower quantile of ``quantile``.
    upper_quantile : float
        Upper quantile of ``quantile``; above ``lower_quantile``.
    iqr_k : float
        k of ``iqr``.
    mad_k : float
        k of ``mad``.
    sigma_k : float
        k of ``sigma``.
    """

    method: str
    lower_quantile: float
    upper_quantile: float
    iqr_k: float
    mad_k: float
    sigma_k: float


@dataclass(frozen=True)
class ImputeSettings:
    """``[impute]``: how missing features are filled (see "欠損の補完" in ``docs/spec.md``).

    Parameters
    ----------
    method : str
        One of ``IMPUTE_METHODS``.
    fill_value : float
        Value of ``constant``.
    n_neighbors : int
        Neighbors of ``knn``.
    """

    method: str
    fill_value: float
    n_neighbors: int


# Table → (its settings, key → check that returns the value).
_TABLES: dict[str, tuple[Callable[..., Any], dict[str, Check]]] = {
    "columns": (ColumnSettings, {"max_missing_rate": _rate}),
    "preprocessing": (PreprocessingSettings, {"use_unlabeled_rows": _flag}),
    "clip": (
        ClipSettings,
        {
            "method": _one_of(CLIP_METHODS),
            "lower_quantile": _rate,
            "upper_quantile": _rate,
            "iqr_k": _positive,
            "mad_k": _positive,
            "sigma_k": _positive,
        },
    ),
    "impute": (
        ImputeSettings,
        {"method": _one_of(IMPUTE_METHODS), "fill_value": _number, "n_neighbors": _count},
    ),
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
    """The settings of a run, one field for each table of ``mloom.toml``.

    Parameters
    ----------
    columns : ColumnSettings
        ``[columns]``.
    preprocessing : PreprocessingSettings
        ``[preprocessing]``.
    clip : ClipSettings
        ``[clip]``.
    impute : ImputeSettings
        ``[impute]``.
    sources : dict[str, dict[str, str]]
        Path of the file each key was read from, by table and key.
    files : list[SettingsFile]
        The files looked up, from the highest priority.
    """

    columns: ColumnSettings
    preprocessing: PreprocessingSettings
    clip: ClipSettings
    impute: ImputeSettings
    sources: dict[str, dict[str, str]]
    files: list[SettingsFile]

    def to_dict(self) -> dict[str, Any]:
        """Describe the settings for ``run.json``.

        Returns
        -------
        dict[str, Any]
            The values by table, where each came from, and the files looked up.
        """
        return asdict(self)


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
        When a file is not valid TOML, or has an unknown table or key, or an
        invalid value, or when the values read from several files do not fit
        together (``lower_quantile`` not below ``upper_quantile``).
    MloomError
        When the defaults of mloom are missing or do not set every key.
    """
    config_dir = mloom_config_dir()
    layers: list[tuple[str, Path | Traversable | None]] = [
        ("project", path),
        ("local_defaults", config_dir / LOCAL_SETTINGS_FILE if config_dir else None),
        ("defaults", config_dir / SETTINGS_FILE if config_dir else files("mloom") / PACKAGED_DEFAULTS),
    ]
    values: dict[str, dict[str, Any]] = {table: {} for table in _TABLES}
    sources: dict[str, dict[str, str]] = {table: {} for table in _TABLES}
    looked_up: list[SettingsFile] = []
    read: dict[str, dict[str, Any]] = {}
    for role, file in layers:
        if file is None:
            continue
        found = file.is_file()
        looked_up.append(SettingsFile(role=role, path=str(file), found=found))
        read = _read(file) if found else {}
        for table, table_values in read.items():
            for key, value in table_values.items():
                if key not in values[table]:
                    values[table][key] = value
                    sources[table][key] = str(file)
    # The last file read is the defaults; it alone must set every key.
    _check_defaults(looked_up[-1], read)
    # Each key may come from another file, so the pair is checked once more as merged.
    clip_sources = sources["clip"]
    _check_quantiles(
        values["clip"], f"{clip_sources['lower_quantile']} and {clip_sources['upper_quantile']} (as merged)"
    )
    return Settings(
        **{table: settings(**values[table]) for table, (settings, _) in _TABLES.items()},
        sources=sources,
        files=looked_up,
    )


def _read(file: Path | Traversable) -> dict[str, dict[str, Any]]:
    try:
        content = tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise InputError(f"{file} is not valid TOML: {error}") from error
    _check_unknown(content, set(_TABLES), str(file))
    values: dict[str, dict[str, Any]] = {}
    for table_name, (_, checks) in _TABLES.items():
        table = content.get(table_name, {})
        if not isinstance(table, dict):
            raise InputError(f"[{table_name}] in {file} must be a table")
        _check_unknown(table, set(checks), f"[{table_name}] of {file}")
        values[table_name] = {}
        for key, check in checks.items():
            if key in table:
                try:
                    values[table_name][key] = check(table[key])
                except (TypeError, ValueError) as error:
                    raise InputError(f"[{table_name}] {key} in {file} {error}") from error
    _check_quantiles(values["clip"], str(file))
    return values


def _check_quantiles(clip: dict[str, Any], where: str) -> None:
    if {"lower_quantile", "upper_quantile"} <= set(clip) and not clip["lower_quantile"] < clip["upper_quantile"]:
        raise InputError(
            f"[clip] lower_quantile ({clip['lower_quantile']}) must be below upper_quantile "
            f"({clip['upper_quantile']}) in {where}"
        )


def _check_unknown(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise InputError(f"unknown key in {where}: {', '.join(unknown)}")


def _check_defaults(defaults: SettingsFile, read: dict[str, dict[str, Any]]) -> None:
    if not defaults.found:
        raise MloomError(f"the defaults of mloom are missing: {defaults.path}")
    missing = [
        f"[{table}] {key}" for table, (_, checks) in _TABLES.items() for key in checks if key not in read.get(table, {})
    ]
    if missing:
        raise MloomError(f"the defaults of mloom do not set {', '.join(missing)}: {defaults.path}")
