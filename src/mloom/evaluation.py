"""Scoring the predictions on the rows after the training period: the scores and the confusion matrix.

See "スコア" and "混同行列" in ``docs/spec.md``. The rows scored are those
after ``training_rows.last_time`` whose target is known and that were
predicted; every column is null when they cannot be told.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import r2_score, root_mean_squared_error

from mloom.errors import InputError
from mloom.training_period import after_training

SCORE_NAMES = ("error", "rmse", "r2")
CONFUSION_NAMES = ("cm", "accuracy", "precision", "recall")
POSITIVE_SIDES = ("above", "below")
DEFAULT_PROBABILITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class Confusion:
    """How a row is told positive, for the confusion matrix.

    Parameters
    ----------
    predicted_threshold : float
        Threshold of the prediction (regression) or of the probability of
        ``positive_class`` (classification).
    above : bool
        Whether the values above the thresholds are positive; otherwise those
        below. A value equal to a threshold is negative.
    actual_threshold : float | None
        Threshold of the target (regression only).
    positive_class : str | None
        ``str()`` of the positive class (classification only).
    """

    predicted_threshold: float
    above: bool = True
    actual_threshold: float | None = None
    positive_class: str | None = None

    def actual_positive(self, actual: pl.Series) -> np.ndarray:
        """Tell the rows whose target is positive.

        Parameters
        ----------
        actual : pl.Series
            The target of the data.

        Returns
        -------
        np.ndarray
            Boolean mask; meaningful only on the rows that are scored.
        """
        if self.positive_class is not None:
            return np.array([str(value) == self.positive_class for value in actual.to_list()], dtype=bool)
        return self._beyond(_numeric_target(actual), self.actual_threshold)

    def predicted_positive(self, values: np.ndarray) -> np.ndarray:
        """Tell the rows predicted positive.

        Parameters
        ----------
        values : np.ndarray
            The predictions (regression) or the probabilities of the positive
            class (classification).

        Returns
        -------
        np.ndarray
            Boolean mask; NaN is negative.
        """
        return self._beyond(values, self.predicted_threshold)

    def _beyond(self, values: np.ndarray, threshold: float | None) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return values > threshold if self.above else values < threshold


def regression_confusion(
    actual_threshold: float | None, predicted_threshold: float | None, positive: str | None, positive_class: Any  # noqa: ANN401 - any class
) -> Confusion | None:
    """Check the arguments of the confusion matrix of a regression model.

    Parameters
    ----------
    actual_threshold : float | None
        Threshold of the target.
    predicted_threshold : float | None
        Threshold of the prediction.
    positive : str | None
        ``"above"`` or ``"below"``.
    positive_class : Any
        Must be ``None``.

    Returns
    -------
    Confusion | None
        The rule, or ``None`` when no threshold is given.

    Raises
    ------
    InputError
        When only some of the three are given, ``positive`` is not
        ``"above"`` or ``"below"``, or ``positive_class`` is given.
    """
    if positive_class is not None:
        raise InputError("positive_class is for classification; a regression model takes cm_actual_threshold, "
                         "cm_predicted_threshold and cm_positive")
    given = {"cm_actual_threshold": actual_threshold, "cm_predicted_threshold": predicted_threshold,
             "cm_positive": positive}  # fmt: skip
    if all(value is None for value in given.values()):
        return None
    if actual_threshold is None or predicted_threshold is None or positive is None:
        missing = [name for name, value in given.items() if value is None]
        raise InputError(f"the confusion matrix of a regression model needs all of {', '.join(given)}; {missing} not given")
    if positive not in POSITIVE_SIDES:
        raise InputError(f"cm_positive must be one of {', '.join(POSITIVE_SIDES)}; got {positive!r}")
    return Confusion(predicted_threshold, above=positive == "above", actual_threshold=actual_threshold)


def classification_confusion(
    classes: Sequence[Any],
    actual_threshold: float | None,
    predicted_threshold: float | None,
    positive: str | None,
    positive_class: Any,  # noqa: ANN401 - any class
) -> Confusion | None:
    """Check the arguments of the confusion matrix of a classification model.

    Parameters
    ----------
    classes : Sequence[Any]
        ``classes_`` of the model.
    actual_threshold : float | None
        Must be ``None``.
    predicted_threshold : float | None
        Threshold of the probability of ``positive_class``; ``None`` for 0.5.
    positive : str | None
        Must be ``None``.
    positive_class : Any
        The positive class, matched by ``str()``; ``None`` for no matrix.

    Returns
    -------
    Confusion | None
        The rule, or ``None`` when ``positive_class`` is not given.

    Raises
    ------
    InputError
        When ``actual_threshold`` or ``positive`` is given,
        ``predicted_threshold`` is given without ``positive_class``, or
        ``positive_class`` is not a class of the model.
    """
    if actual_threshold is not None or positive is not None:
        raise InputError("cm_actual_threshold and cm_positive are for regression; a classification model takes "
                         "positive_class and cm_predicted_threshold")
    if positive_class is None:
        if predicted_threshold is not None:
            raise InputError("cm_predicted_threshold of a classification model needs positive_class")
        return None
    names = [str(label) for label in classes]
    if str(positive_class) not in names:
        raise InputError(f"positive_class {str(positive_class)!r} is not a class of the model: {names}")
    threshold = DEFAULT_PROBABILITY_THRESHOLD if predicted_threshold is None else predicted_threshold
    return Confusion(threshold, positive_class=str(positive_class))


def scored_rows(frame: pl.DataFrame, training_rows: dict[str, Any], predicted: np.ndarray) -> np.ndarray | None:
    """Tell the rows the scores and the confusion matrix are computed on.

    Parameters
    ----------
    frame : pl.DataFrame
        The data to predict.
    training_rows : dict[str, Any]
        ``training_rows`` of ``model.pkl``.
    predicted : np.ndarray
        Boolean mask of the rows that were predicted (``passed_filter``).

    Returns
    -------
    np.ndarray | None
        Boolean mask of the rows after ``last_time`` whose target is neither
        missing (null, NaN) nor infinite and that were predicted; ``None``
        when the data has no target column or the model has no
        ``last_time``.

    Raises
    ------
    InputError
        When ``mloom.training_period.after_training`` raises.
    """
    target, latest = training_rows["target"], training_rows["last_time"]
    if target not in frame.columns or latest is None:
        return None
    known = pl.col(target).is_not_null()
    if frame.schema[target].is_float():
        known = known & pl.col(target).is_finite()
    has_target = frame.select(known.fill_null(value=False)).to_series().to_numpy()
    return after_training(frame, training_rows["time_column"], latest) & has_target & predicted


def regression_scores(actual: pl.Series | None, predicted: np.ndarray, scored: np.ndarray | None) -> list[pl.Series]:
    """Compute ``error``, ``rmse`` and ``r2``.

    Parameters
    ----------
    actual : pl.Series | None
        The target of the data; ``None`` when ``scored`` is ``None``.
    predicted : np.ndarray
        The predictions.
    scored : np.ndarray | None
        The rows scored (``scored_rows``); ``None`` for all null.

    Returns
    -------
    list[pl.Series]
        ``error`` (prediction - target on the rows scored, null elsewhere),
        then ``rmse`` (null on 0 rows) and ``r2`` (null below 2 rows),
        repeated on every row.

    Raises
    ------
    InputError
        When the target is not numeric.
    """
    height = predicted.shape[0]
    if actual is None or scored is None:
        return [_nulls(name, height, pl.Float64) for name in SCORE_NAMES]
    truth, guess = _numeric_target(actual)[scored], predicted[scored]
    error = np.full(height, None, dtype=object)
    error[scored] = guess - truth
    rmse = float(root_mean_squared_error(truth, guess)) if scored.any() else None
    r2 = float(r2_score(truth, guess)) if scored.sum() >= 2 else None
    return [
        pl.Series("error", error.tolist(), dtype=pl.Float64),
        pl.Series("rmse", [rmse] * height, dtype=pl.Float64),
        pl.Series("r2", [r2] * height, dtype=pl.Float64),
    ]


def confusion_columns(
    actual_positive: np.ndarray | None, predicted_positive: np.ndarray, scored: np.ndarray | None
) -> list[pl.Series]:
    """Compute ``cm``, ``accuracy``, ``precision`` and ``recall``.

    Parameters
    ----------
    actual_positive : np.ndarray | None
        Whether the target of each row is positive; ``None`` when ``scored``
        is ``None``.
    predicted_positive : np.ndarray
        Whether each row is predicted positive.
    scored : np.ndarray | None
        The rows scored (``scored_rows``); ``None`` for all null.

    Returns
    -------
    list[pl.Series]
        ``cm`` (``tp``, ``fp``, ``fn`` or ``tn`` on the rows scored, null
        elsewhere), then ``accuracy``, ``precision`` and ``recall`` repeated
        on every row (null when the denominator is 0).
    """
    height = predicted_positive.shape[0]
    if actual_positive is None or scored is None:
        return [_nulls("cm", height, pl.String), *(_nulls(name, height, pl.Float64) for name in CONFUSION_NAMES[1:])]
    cells = np.where(actual_positive, np.where(predicted_positive, "tp", "fn"), np.where(predicted_positive, "fp", "tn"))
    cm = np.full(height, None, dtype=object)
    cm[scored] = cells[scored]
    counts = {cell: int(np.sum(scored & (cells == cell))) for cell in ("tp", "fp", "fn", "tn")}
    tp = counts["tp"]
    values = {
        "accuracy": _ratio(tp + counts["tn"], sum(counts.values())),
        "precision": _ratio(tp, tp + counts["fp"]),
        "recall": _ratio(tp, tp + counts["fn"]),
    }
    return [
        pl.Series("cm", cm.tolist(), dtype=pl.String),
        *(pl.Series(name, [value] * height, dtype=pl.Float64) for name, value in values.items()),
    ]


def _numeric_target(actual: pl.Series) -> np.ndarray:
    if not (actual.dtype.is_numeric() or actual.dtype == pl.Null):
        raise InputError(f"the scores of a regression model need a numeric target; {actual.name!r} is {actual.dtype}")
    return actual.cast(pl.Float64).fill_nan(None).fill_null(np.nan).to_numpy()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _nulls(name: str, height: int, dtype: type[pl.DataType]) -> pl.Series:
    return pl.Series(name, [None] * height, dtype=dtype)
