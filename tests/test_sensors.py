"""Tests of reading ``unavailable-sensors.json`` and matching its entries to columns."""

from pathlib import Path

import pytest

from mloom.errors import InputError
from mloom.sensors import EXACT, SensorEntry, load_unavailable_sensors


def _load(tmp_path: Path, content: str) -> list[SensorEntry]:
    path = tmp_path / "unavailable-sensors.json"
    path.write_text(content, encoding="utf-8")
    sensors = load_unavailable_sensors(path)
    assert sensors.found
    return sensors.entries


def test_reads_names_matches_and_reasons(tmp_path: Path) -> None:
    """Each entry keeps its name, match and reason; a missing match is exact and a missing reason ``None``."""
    entries = _load(
        tmp_path, '{"sensors": [{"name": "s0", "reason": "故障"}, {"name": "s9"}, {"name": "TI-", "match": "starts_with"}]}'
    )
    assert entries == [SensorEntry("s0", EXACT, "故障"), SensorEntry("s9", EXACT, None), SensorEntry("TI-", "starts_with", None)]


def test_missing_file_excludes_nothing(tmp_path: Path) -> None:
    """A missing file is recorded as not found and excludes nothing."""
    sensors = load_unavailable_sensors(tmp_path / "none.json")
    assert not sensors.found
    assert sensors.entries == []


def test_none_reads_nothing() -> None:
    """``None`` reads no file."""
    sensors = load_unavailable_sensors(None)
    assert sensors.path is None
    assert sensors.entries == []


@pytest.mark.parametrize(
    ("match", "matched"),
    [
        (EXACT, ["TI-101"]),
        ("starts_with", ["TI-101", "TI-101_mean", "TI-1010"]),
        ("ends_with", ["TI-101", "xTI-101"]),
        ("contains", ["TI-101", "TI-101_mean", "TI-1010", "xTI-101"]),
    ],
)
def test_each_match_compares_the_characters_as_they_are(match: str, matched: list[str]) -> None:
    """Each way of matching compares the characters as they are, case-sensitive and with no pattern."""
    columns = ["TI-101", "TI-101_mean", "TI-1010", "xTI-101", "ti-101", "TI-10"]
    entry = SensorEntry("TI-101", match, None)
    assert [column for column in columns if entry.matches(column)] == matched
    assert not SensorEntry("AI-[1]", "contains", None).matches("AI-1")
    assert SensorEntry("AI-[1]", EXACT, None).matches("AI-[1]")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "not valid JSON"),
        ('["s0"]', '"sensors" list'),
        ('{"sensors": "s0"}', '"sensors" list'),
        ('{"sensors": [{"reason": "x"}]}', 'non-empty string "name"'),
        ('{"sensors": [{"name": "", "match": "contains"}]}', 'non-empty string "name"'),
        ('{"sensors": [{"name": "s0", "reason": 1}]}', '"reason" of \'s0\''),
        ('{"sensors": [{"name": "s0", "match": "regex"}]}', '"match" of \'s0\''),
        ('{"sensors": [{"name": "s0", "match": ["exact"]}]}', '"match" of \'s0\''),
        ('{"sensors": [{"name": "s0", "mach": "starts_with"}]}', "unknown keys \\['mach'\\]"),
    ],
)
def test_invalid_file_raises(tmp_path: Path, content: str, message: str) -> None:
    """A file that is not valid JSON or not in the format raises ``InputError`` naming the entry."""
    path = tmp_path / "unavailable-sensors.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputError, match=message):
        load_unavailable_sensors(path)
