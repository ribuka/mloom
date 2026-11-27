"""Reading ``unavailable-sensors.json``: the sensors that must not be used, and which columns they match.

See "使えないセンサの定義" in ``docs/spec.md`` for the format, how an entry
matches a column, and how a missing or invalid file is handled.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from mloom.errors import InputError

EXACT = "exact"
# How an entry matches a column name: the characters as they are, case-sensitive.
MATCHES = {
    EXACT: str.__eq__,
    "starts_with": str.startswith,
    "ends_with": str.endswith,
    "contains": str.__contains__,
}
_KEYS = {"name", "match", "reason"}


@dataclass(frozen=True)
class SensorEntry:
    """One entry of ``unavailable-sensors.json``.

    Parameters
    ----------
    name : str
        The name, or the part of the column names to match.
    match : str
        How ``name`` matches a column: one of ``MATCHES``.
    reason : str | None
        Why the sensor must not be used (``None`` when the file gives none).
    """

    name: str
    match: str
    reason: str | None

    def matches(self, column: str) -> bool:
        """Tell whether a column is of this sensor.

        Parameters
        ----------
        column : str
            Column name.

        Returns
        -------
        bool
            Whether the column matches ``name`` the way ``match`` says.
        """
        return MATCHES[self.match](column, self.name)


@dataclass(frozen=True)
class UnavailableSensors:
    """The sensors read from ``unavailable-sensors.json``.

    Parameters
    ----------
    path : str | None
        Path of the file, or ``None`` when no file was configured.
    found : bool
        Whether the file existed.
    entries : list[SensorEntry]
        The entries, in the order of the file.
    """

    path: str | None
    found: bool
    entries: list[SensorEntry]

    def first_match(self, column: str) -> SensorEntry | None:
        """Find the first entry, in the order of the file, that a column matches.

        Parameters
        ----------
        column : str
            Column name.

        Returns
        -------
        SensorEntry | None
            The entry, or ``None`` when the column is of no unavailable sensor.
        """
        return next((entry for entry in self.entries if entry.matches(column)), None)


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
        return UnavailableSensors(path=None, found=False, entries=[])
    if not path.is_file():
        return UnavailableSensors(path=str(path), found=False, entries=[])
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InputError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(content, dict) or not isinstance(content.get("sensors"), list):
        raise InputError(f'{path} must be an object with a "sensors" list')
    return UnavailableSensors(path=str(path), found=True, entries=[_entry(entry, path) for entry in content["sensors"]])


def _entry(entry: object, path: Path) -> SensorEntry:
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
        raise InputError(f'each entry of "sensors" in {path} needs a non-empty string "name"; got {entry!r}')
    name = entry["name"]
    unknown = sorted(str(key) for key in entry.keys() - _KEYS)
    if unknown:
        raise InputError(f"entry {name!r} in {path} has unknown keys {unknown}; the keys are {sorted(_KEYS)}")
    match = entry.get("match", EXACT)
    if not isinstance(match, str) or match not in MATCHES:
        raise InputError(f'"match" of {name!r} in {path} must be one of {list(MATCHES)}; got {match!r}')
    reason = entry.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise InputError(f'"reason" of {name!r} in {path} must be a string')
    return SensorEntry(name=name, match=match, reason=reason)
