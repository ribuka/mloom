"""Reading ``unavailable-sensors.json``: the sensors that must not be used.

See "使えないセンサの定義" in ``docs/spec.md`` for the format and how a
missing or invalid file is handled.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from mloom.errors import InputError


@dataclass(frozen=True)
class UnavailableSensors:
    """The sensors read from ``unavailable-sensors.json``.

    Parameters
    ----------
    path : str | None
        Path of the file, or ``None`` when no file was configured.
    found : bool
        Whether the file existed.
    reasons : dict[str, str | None]
        Reason of each sensor name (``None`` when the file gives none).
    """

    path: str | None
    found: bool
    reasons: dict[str, str | None]


def load_unavailable_sensors(path: Path | None) -> UnavailableSensors:
    """Read the unavailable sensors.

    Parameters
    ----------
    path : Path | None
        Path of ``unavailable-sensors.json``. ``None`` reads nothing.

    Returns
    -------
    UnavailableSensors
        The sensors; empty when ``path`` is ``None`` or the file is missing.

    Raises
    ------
    InputError
        When the file is not valid JSON or not in the expected format.
    """
    if path is None:
        return UnavailableSensors(path=None, found=False, reasons={})
    if not path.is_file():
        return UnavailableSensors(path=str(path), found=False, reasons={})
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InputError(f"{path} is not valid JSON: {error}") from error
    return UnavailableSensors(path=str(path), found=True, reasons=_parse(content, path))


def _parse(content: object, path: Path) -> dict[str, str | None]:
    if not isinstance(content, dict) or not isinstance(content.get("sensors"), list):
        raise InputError(f'{path} must be an object with a "sensors" list')
    reasons: dict[str, str | None] = {}
    for entry in content["sensors"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise InputError(f'each entry of "sensors" in {path} needs a string "name"; got {entry!r}')
        reason = entry.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise InputError(f'"reason" of {entry["name"]!r} in {path} must be a string')
        reasons[entry["name"]] = reason
    return reasons
