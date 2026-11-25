"""Tests of reading ``unavailable-sensors.json``."""

from pathlib import Path

import pytest

from mloom.errors import InputError
from mloom.sensors import load_unavailable_sensors


def test_reads_names_and_reasons(tmp_path: Path) -> None:
    """Each sensor keeps its reason; a missing reason is ``None``."""
    path = tmp_path / "unavailable-sensors.json"
    path.write_text('{"sensors": [{"name": "s0", "reason": "故障"}, {"name": "s9"}]}', encoding="utf-8")
    sensors = load_unavailable_sensors(path)
    assert sensors.found
    assert sensors.reasons == {"s0": "故障", "s9": None}


def test_missing_file_excludes_nothing(tmp_path: Path) -> None:
    """A missing file is recorded as not found and excludes nothing."""
    sensors = load_unavailable_sensors(tmp_path / "none.json")
    assert not sensors.found
    assert sensors.reasons == {}


def test_none_reads_nothing() -> None:
    """``None`` reads no file."""
    sensors = load_unavailable_sensors(None)
    assert sensors.path is None
    assert sensors.reasons == {}


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '["s0"]',
        '{"sensors": "s0"}',
        '{"sensors": [{"reason": "x"}]}',
        '{"sensors": [{"name": "s0", "reason": 1}]}',
    ],
)
def test_invalid_file_raises(tmp_path: Path, content: str) -> None:
    """A file that is not valid JSON or not in the format raises ``InputError``."""
    path = tmp_path / "unavailable-sensors.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputError):
        load_unavailable_sensors(path)
