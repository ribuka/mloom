"""Reading ``config/settings.toml``: values of the processing a repository can tune.

See "設定ファイル" in ``docs/spec.md`` for the keys and their defaults.
A missing file uses the defaults; an unknown section or key is an error, so
that a typo does not silently keep the default.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mloom.errors import InputError

DEFAULT_MAX_MISSING_RATE = 0.5


@dataclass(frozen=True)
class Settings:
    """The settings of a run.

    Parameters
    ----------
    path : str | None
        Path of the file, or ``None`` when no file was configured.
    found : bool
        Whether the file existed.
    max_missing_rate : float
        A feature column whose missing rate is above this is excluded
        (``[columns] max_missing_rate``).
    """

    path: str | None
    found: bool
    max_missing_rate: float = DEFAULT_MAX_MISSING_RATE


def load_settings(path: Path | None) -> Settings:
    """Read the settings.

    Parameters
    ----------
    path : Path | None
        Path of ``settings.toml``. ``None`` reads nothing.

    Returns
    -------
    Settings
        The settings; the defaults when ``path`` is ``None`` or the file is
        missing.

    Raises
    ------
    InputError
        When the file is not valid TOML, or has an unknown section or key, or
        an invalid value.
    """
    if path is None:
        return Settings(path=None, found=False)
    if not path.is_file():
        return Settings(path=str(path), found=False)
    try:
        content = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise InputError(f"{path} is not valid TOML: {error}") from error
    _check_keys(content, {"columns"}, str(path))
    columns = content.get("columns", {})
    if not isinstance(columns, dict):
        raise InputError(f"[columns] in {path} must be a table")
    _check_keys(columns, {"max_missing_rate"}, f"[columns] of {path}")
    rate = columns.get("max_missing_rate", DEFAULT_MAX_MISSING_RATE)
    if isinstance(rate, bool) or not isinstance(rate, int | float) or not 0 <= rate <= 1:
        raise InputError(f"max_missing_rate in {path} must be a number from 0 to 1; got {rate!r}")
    return Settings(path=str(path), found=True, max_missing_rate=float(rate))


def _check_keys(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise InputError(f"unknown key in {where}: {', '.join(unknown)}")
