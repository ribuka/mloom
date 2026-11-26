"""``predict``: predicting with a ``model.pkl``, row by row in the order of the input.

See "予測" in ``docs/spec.md``. The module is not named ``mloom.predict``, so
that the function ``mloom.predict`` does not hide it.
"""

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

from mloom.data import Data, load_data
from mloom.errors import InputError
from mloom.filters import filter_mask
from mloom.pipeline import candidate_columns
from mloom.row_hash import hash_rows

Model = str | Path | dict[str, Any]
PREDICTION = "prediction"
PASSED_FILTER = "passed_filter"
USED_FOR_TRAINING = "used_for_training"
PROBA_PREFIX = "proba_"
_BUNDLE_KEYS = ("pipeline", "filters", "train_filters", "training_rows")


def predict(model: Model, data: Data, *, keep_columns: Sequence[str] = ()) -> pl.DataFrame:
    """Predict every row of the data, applying the row filters of the model.

    Parameters
    ----------
    model : Model
        Path of a ``model.pkl``, or the dict read from one.
    data : Data
        Path of a Parquet file, or a polars DataFrame.
    keep_columns : Sequence[str]
        Columns of the input to copy into the output, first.

    Returns
    -------
    pl.DataFrame
        One row for each row of the input, in its order: ``keep_columns``,
        ``prediction``, ``passed_filter``, ``used_for_training``, and for
        classification ``proba_<class>`` in the order of ``classes_``. A row
        that does not pass ``filters`` is not predicted: its ``prediction``
        is NaN (null for classification) and its probabilities NaN.

    Raises
    ------
    InputError
        When the model or the data cannot be read, a column the model needs
        or a column of ``keep_columns`` is not in the data, a value of
        ``filters`` does not fit its column, or a column of ``keep_columns``
        has the name of an output column.
    """
    bundle = load_model(model)
    pipeline: Pipeline = bundle["pipeline"]
    frame, _ = load_data(data)
    _check_keep_columns(frame, keep_columns, _output_names(pipeline))
    _require_columns(frame, candidate_columns(pipeline), "candidate column")
    passed = filter_mask(frame, bundle["filters"], "filters").to_numpy()
    predicted = _predictions(pipeline, frame, passed)
    return pl.DataFrame(
        [
            *(frame[name] for name in keep_columns),
            predicted[0],
            pl.Series(PASSED_FILTER, passed),
            _used_for_training(frame, bundle, passed),
            *predicted[1:],
        ]
    )


def load_model(model: Model) -> dict[str, Any]:
    """Read a ``model.pkl``.

    Parameters
    ----------
    model : Model
        Path of a ``model.pkl``, or the dict read from one (returned as is).

    Returns
    -------
    dict[str, Any]
        ``pipeline``, ``filters``, ``train_filters`` and ``training_rows``.

    Raises
    ------
    InputError
        When the file does not exist, or does not hold that dict.
    """
    if isinstance(model, dict):
        bundle = model
    else:
        path = Path(model)
        if not path.is_file():
            raise InputError(f"model file not found: {path}")
        with path.open("rb") as file:
            bundle = pickle.load(file)
    if not isinstance(bundle, dict) or any(key not in bundle for key in _BUNDLE_KEYS):
        raise InputError(f"the model must be a dict with the keys {', '.join(_BUNDLE_KEYS)} (a model.pkl of mloom)")
    return bundle


def _output_names(pipeline: Pipeline) -> list[str]:
    return [PREDICTION, PASSED_FILTER, USED_FOR_TRAINING, *_proba_names(pipeline)]


def _proba_names(pipeline: Pipeline) -> list[str]:
    if not is_classifier(pipeline):
        return []
    return [f"{PROBA_PREFIX}{label}" for label in pipeline.classes_]


def _check_keep_columns(frame: pl.DataFrame, keep_columns: Sequence[str], outputs: list[str]) -> None:
    _require_columns(frame, keep_columns, "column of keep_columns")
    clashes = [name for name in keep_columns if name in outputs]
    if clashes:
        raise InputError(f"keep_columns {clashes} have the names of output columns")
    if len(set(keep_columns)) != len(keep_columns):
        raise InputError(f"keep_columns lists a column more than once: {list(keep_columns)}")


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise InputError(f"{what} {missing} not in the data")


def _predictions(pipeline: Pipeline, frame: pl.DataFrame, passed: np.ndarray) -> list[pl.Series]:
    # Only the rows that pass the filters are predicted; with none, predict is not called.
    rows = frame.filter(pl.Series(passed))
    if not is_classifier(pipeline):
        values = np.full(frame.height, np.nan)
        if rows.height:
            values[passed] = np.ravel(pipeline.predict(rows))
        return [pl.Series(PREDICTION, values)]
    classes = pipeline.classes_
    labels = np.full(frame.height, None, dtype=object)
    probabilities = np.full((frame.height, len(classes)), np.nan)
    if rows.height:
        labels[passed] = pipeline.predict(rows)
        probabilities[passed] = pipeline.predict_proba(rows)
    return [
        pl.Series(PREDICTION, labels.tolist(), dtype=pl.Series(classes).dtype),
        *(pl.Series(name, probabilities[:, index]) for index, name in enumerate(_proba_names(pipeline))),
    ]


def _used_for_training(frame: pl.DataFrame, bundle: dict[str, Any], passed: np.ndarray) -> pl.Series:
    training_rows = bundle["training_rows"]
    unknown = pl.Series(USED_FOR_TRAINING, [None] * frame.height, dtype=pl.Boolean)
    if training_rows["target"] not in frame.columns:
        return unknown
    _require_columns(frame, training_rows["columns"], "column the training rows are matched by")
    try:
        in_training = filter_mask(frame, bundle["train_filters"], "train_filters").to_numpy()
    except InputError:
        # train_filters does not choose the rows to predict; without it, only the answer is unknown.
        return unknown
    candidates = passed & in_training
    hashes = hash_rows(frame.filter(pl.Series(candidates)), training_rows["columns"])
    used = np.zeros(frame.height, dtype=bool)
    used[candidates] = np.isin(hashes, training_rows["hashes"])
    return pl.Series(USED_FOR_TRAINING, used)
