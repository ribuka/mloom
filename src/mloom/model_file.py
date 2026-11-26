"""The format of ``model.pkl``: the record of the run, then the model, as two pickles in one file.

The first pickle holds only plain Python values (``dict``, ``list``, ``str``,
``int``, ``float``, ``bool`` and ``None``), so that the record is read without
scikit-learn or numpy; this module imports neither. See "``model.pkl``" in
``docs/spec.md``.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, BinaryIO

from mloom.errors import InputError

MODEL_FILE = "model.pkl"
_RECORD_KEY = "mloom_version"
_TEMP_SUFFIX = ".tmp"
# What pickle.load raises for a truncated or broken file (ValueError: an unknown protocol, for example).
_BROKEN = (EOFError, pickle.UnpicklingError, ValueError)


def write_model_file(directory: Path, record: dict[str, Any], model: dict[str, Any]) -> Path:
    """Write ``model.pkl`` in a directory, replacing an existing one only when both pickles are written.

    Parameters
    ----------
    directory : Path
        Existing directory to write to.
    record : dict[str, Any]
        The record of the run. It is passed through JSON, so that it holds
        only plain Python values.
    model : dict[str, Any]
        The model (``mloom.result.Result.model_bundle``).

    Returns
    -------
    Path
        Path of ``model.pkl``.
    """
    path = directory / MODEL_FILE
    temp = path.with_name(path.name + _TEMP_SUFFIX)
    try:
        with temp.open("wb") as file:
            pickle.dump(json.loads(json.dumps(record)), file)
            pickle.dump(model, file)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def read_run(path: str | Path) -> dict[str, Any]:
    """Read the record of the run from a ``model.pkl``, without reading the model.

    A pickle is only to be read from a trusted source.

    Parameters
    ----------
    path : str | Path
        Path of a ``model.pkl`` saved by ``mloom optimize``.

    Returns
    -------
    dict[str, Any]
        The record (see "記録" in ``docs/spec.md``).

    Raises
    ------
    InputError
        When the file does not exist, or does not start with a record.
    """
    with _open(path) as file:
        return _read_record(file)


def read_model(path: str | Path) -> Any:  # noqa: ANN401 - checked by mloom.prediction.load_model
    """Read the model from a ``model.pkl``, skipping the record of the run.

    Parameters
    ----------
    path : str | Path
        Path of a ``model.pkl`` saved by ``mloom optimize``.

    Returns
    -------
    Any
        The second pickle of the file.

    Raises
    ------
    InputError
        When the file does not exist, does not start with a record, or has no
        second pickle.
    """
    with _open(path) as file:
        _read_record(file)
        try:
            return pickle.load(file)
        except _BROKEN as error:
            raise InputError(f"{file.name} has no model after the record of the run, or it is broken") from error


def _open(path: str | Path) -> BinaryIO:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"model file not found: {path}")
    return path.open("rb")


def _read_record(file: BinaryIO) -> dict[str, Any]:
    try:
        record = pickle.load(file)
    except _BROKEN as error:
        raise InputError(f"{file.name} is not a model.pkl of mloom") from error
    if not isinstance(record, dict) or _RECORD_KEY not in record:
        raise InputError(
            f"{file.name} does not start with the record of the run: it is not a model.pkl of mloom, "
            "or one saved by an older mloom; make it again with mloom optimize"
        )
    return record
