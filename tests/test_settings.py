"""Tests of reading ``settings.toml``."""

from pathlib import Path

import pytest

from mloom.errors import InputError
from mloom.settings import DEFAULT_MAX_MISSING_RATE, load_settings


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_reads_max_missing_rate(tmp_path: Path) -> None:
    """``[columns] max_missing_rate`` is read; an integer is accepted."""
    assert load_settings(_write(tmp_path, "[columns]\nmax_missing_rate = 0.2\n")).max_missing_rate == 0.2
    assert load_settings(_write(tmp_path, "[columns]\nmax_missing_rate = 1\n")).max_missing_rate == 1.0


@pytest.mark.parametrize("content", ["", "[columns]\n"])
def test_missing_key_uses_default(tmp_path: Path, content: str) -> None:
    """A file without the key keeps the default."""
    settings = load_settings(_write(tmp_path, content))
    assert settings.found
    assert settings.max_missing_rate == DEFAULT_MAX_MISSING_RATE


def test_missing_file_uses_default(tmp_path: Path) -> None:
    """A missing file is recorded as not found and uses the default."""
    settings = load_settings(tmp_path / "none.toml")
    assert not settings.found
    assert settings.max_missing_rate == DEFAULT_MAX_MISSING_RATE


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[columns\n", "not valid TOML"),
        ("[column]\nmax_missing_rate = 0.2\n", "unknown key"),
        ("[columns]\nmax_missing = 0.2\n", "unknown key"),
        ("columns = 1\n", "must be a table"),
        ("[columns]\nmax_missing_rate = 1.5\n", "from 0 to 1"),
        ("[columns]\nmax_missing_rate = true\n", "from 0 to 1"),
        ('[columns]\nmax_missing_rate = "0.2"\n', "from 0 to 1"),
    ],
)
def test_invalid_file_raises(tmp_path: Path, content: str, message: str) -> None:
    """Invalid TOML, unknown sections or keys, and invalid values raise ``InputError``."""
    with pytest.raises(InputError, match=message):
        load_settings(_write(tmp_path, content))
