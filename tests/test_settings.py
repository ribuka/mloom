"""Tests of reading ``settings.toml``."""

from pathlib import Path

import pytest

from mloom.errors import InputError
from mloom.settings import (
    DEFAULT_MAX_MISSING_RATE,
    DEFAULT_USE_UNLABELED_ROWS,
    load_settings,
)


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_reads_max_missing_rate(tmp_path: Path) -> None:
    """``[columns] max_missing_rate`` is read; an integer is accepted."""
    assert load_settings(_write(tmp_path, "[columns]\nmax_missing_rate = 0.2\n")).max_missing_rate == 0.2
    assert load_settings(_write(tmp_path, "[columns]\nmax_missing_rate = 1\n")).max_missing_rate == 1.0


def test_reads_use_unlabeled_rows(tmp_path: Path) -> None:
    """``[preprocessing] use_unlabeled_rows`` is read."""
    settings = load_settings(_write(tmp_path, "[preprocessing]\nuse_unlabeled_rows = false\n"))
    assert settings.use_unlabeled_rows is False


@pytest.mark.parametrize("content", ["", "[columns]\n", "[preprocessing]\n"])
def test_missing_key_uses_default(tmp_path: Path, content: str) -> None:
    """A file without the keys keeps the defaults."""
    settings = load_settings(_write(tmp_path, content))
    assert settings.found
    assert settings.max_missing_rate == DEFAULT_MAX_MISSING_RATE
    assert settings.use_unlabeled_rows is DEFAULT_USE_UNLABELED_ROWS


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
        ("[preprocessing]\nuse_unlabeled_rows = 1\n", "true or false"),
        ("[preprocessing]\nuse_unlabeled = true\n", "unknown key"),
    ],
)
def test_invalid_file_raises(tmp_path: Path, content: str, message: str) -> None:
    """Invalid TOML, unknown sections or keys, and invalid values raise ``InputError``."""
    with pytest.raises(InputError, match=message):
        load_settings(_write(tmp_path, content))
