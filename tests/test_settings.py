"""Tests of reading the settings and their defaults."""

import tomllib
from pathlib import Path

import pytest

import mloom.settings
from mloom.errors import InputError, MloomError
from mloom.settings import LOCAL_SETTINGS_FILE, SETTINGS_FILE, load_settings

REPOSITORY_DEFAULTS = Path(__file__).parents[1] / "config" / SETTINGS_FILE


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_repository_defaults_set_every_key() -> None:
    """config/mloom.toml of this repository is complete: every setting has a default there."""
    defaults = tomllib.loads(REPOSITORY_DEFAULTS.read_text(encoding="utf-8"))
    assert defaults == {"columns": {"max_missing_rate": 0.5}, "preprocessing": {"use_unlabeled_rows": True}}


def test_defaults_alone(mloom_config: Path) -> None:
    """Without a project file, every key comes from the defaults."""
    settings = load_settings(None)
    assert (settings.max_missing_rate, settings.use_unlabeled_rows) == (0.5, True)
    assert set(settings.sources.values()) == {str(mloom_config / SETTINGS_FILE)}
    assert [file.role for file in settings.files] == ["local_defaults", "defaults"]


def test_each_key_comes_from_the_first_file_that_sets_it(mloom_config: Path, tmp_path: Path) -> None:
    """The project file wins over the local defaults, which win over the defaults, key by key."""
    local = _write(mloom_config / LOCAL_SETTINGS_FILE, "[columns]\nmax_missing_rate = 0.3\n")
    project = _write(tmp_path / "project.toml", "[preprocessing]\nuse_unlabeled_rows = false\n")
    settings = load_settings(project)
    assert (settings.max_missing_rate, settings.use_unlabeled_rows) == (0.3, False)
    assert settings.sources == {"max_missing_rate": str(local), "use_unlabeled_rows": str(project)}
    assert [(file.role, file.found) for file in settings.files] == [
        ("project", True),
        ("local_defaults", True),
        ("defaults", True),
    ]

    _write(project, "[columns]\nmax_missing_rate = 0.2\n")
    assert load_settings(project).max_missing_rate == 0.2


def test_missing_project_file_uses_defaults(mloom_config: Path, tmp_path: Path) -> None:
    """A missing project file is recorded as not found."""
    settings = load_settings(tmp_path / "none.toml")
    assert settings.files[0].found is False
    assert settings.max_missing_rate == 0.5


def test_wheel_reads_the_packaged_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without a clone of the repository, the defaults shipped in the wheel are read and no local file is looked up."""
    package = tmp_path / "package"
    package.mkdir()
    _write(package / mloom.settings.PACKAGED_DEFAULTS, REPOSITORY_DEFAULTS.read_text(encoding="utf-8"))
    monkeypatch.setattr(mloom.settings, "mloom_config_dir", lambda: None)
    monkeypatch.setattr(mloom.settings, "files", lambda _: package)
    settings = load_settings(None)
    assert [(file.role, file.path) for file in settings.files] == [
        ("defaults", str(package / mloom.settings.PACKAGED_DEFAULTS))
    ]
    assert settings.use_unlabeled_rows is True


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
    ],
)
@pytest.mark.parametrize("layer", ["project", "local_defaults"])
def test_invalid_file_raises(mloom_config: Path, tmp_path: Path, content: str, message: str, layer: str) -> None:
    """Invalid TOML, unknown sections or keys, and invalid values raise ``InputError`` in any file."""
    if layer == "project":
        path = _write(tmp_path / "project.toml", content)
    else:
        _write(mloom_config / LOCAL_SETTINGS_FILE, content)
        path = None
    with pytest.raises(InputError, match=message):
        load_settings(path)


@pytest.mark.parametrize("content", [None, "[columns]\nmax_missing_rate = 0.5\n"])
def test_incomplete_defaults_raise(mloom_config: Path, tmp_path: Path, content: str | None) -> None:
    """Missing defaults, or defaults without every key, raise even when the project file sets the key."""
    defaults = mloom_config / SETTINGS_FILE
    if content is None:
        defaults.unlink()
    else:
        _write(defaults, content)
    project = _write(tmp_path / "project.toml", "[preprocessing]\nuse_unlabeled_rows = true\n")
    with pytest.raises(MloomError, match="defaults of mloom"):
        load_settings(project)
