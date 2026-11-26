"""Bounds of the clipping of the features, by the method of ``[clip]``.

See "クリップ" in ``docs/spec.md``. The bounds come from the finite values of
the statistic rows. A column whose bounds fall on one point (no spread) or
cannot be computed (no finite value) is not clipped: its bounds are -inf to
+inf, so that clipping never turns a column into a constant.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from mloom.settings import ClipSettings

# Scale of the MAD to the standard deviation of a normal distribution.
MAD_SCALE = 1.4826

Bounds = tuple[np.ndarray, np.ndarray]


def _quantile(x: np.ndarray, settings: ClipSettings) -> Bounds:
    lower, upper = np.nanquantile(x, [settings.lower_quantile, settings.upper_quantile], axis=0)
    return lower, upper


def _iqr(x: np.ndarray, settings: ClipSettings) -> Bounds:
    q1, q3 = np.nanquantile(x, [0.25, 0.75], axis=0)
    spread = settings.iqr_k * (q3 - q1)
    return q1 - spread, q3 + spread


def _mad(x: np.ndarray, settings: ClipSettings) -> Bounds:
    median = np.nanmedian(x, axis=0)
    spread = settings.mad_k * MAD_SCALE * np.nanmedian(np.abs(x - median), axis=0)
    return median - spread, median + spread


def _sigma(x: np.ndarray, settings: ClipSettings) -> Bounds:
    mean = np.nanmean(x, axis=0)
    spread = settings.sigma_k * np.nanstd(x, axis=0)
    return mean - spread, mean + spread


_METHODS: dict[str, Callable[[np.ndarray, ClipSettings], Bounds]] = {
    "quantile": _quantile,
    "iqr": _iqr,
    "mad": _mad,
    "sigma": _sigma,
}


def clip_bounds(x: np.ndarray, settings: ClipSettings) -> Bounds:
    """Compute the lower and upper bound of each column.

    Parameters
    ----------
    x : np.ndarray
        Features of the statistic rows, with +inf and -inf already turned
        into NaN; NaN is ignored.
    settings : ClipSettings
        The method (not ``"none"``) and its parameters.

    Returns
    -------
    Bounds
        Lower and upper bounds, one value for each column; -inf and +inf for
        a column that is not clipped.
    """
    with warnings.catch_warnings():
        # A column without a finite value gives NaN bounds, and is not clipped below.
        warnings.simplefilter("ignore", RuntimeWarning)
        lower, upper = _METHODS[settings.method](x, settings)
    # Not lower < upper is also true for NaN; isclose catches a spread of rounding only.
    unclipped = ~(lower < upper) | np.isclose(lower, upper, rtol=1e-12, atol=0.0)
    return np.where(unclipped, -np.inf, lower), np.where(unclipped, np.inf, upper)


def describe_clip(method: str, bounds: Bounds | None, columns: list[str]) -> dict[str, Any]:
    """Describe the clipping for the record of the run (``preprocessing.clip``).

    Parameters
    ----------
    method : str
        ``[clip] method``.
    bounds : Bounds | None
        The bounds from ``clip_bounds``; ``None`` for ``"none"``.
    columns : list[str]
        The columns of the bounds, in their order.

    Returns
    -------
    dict[str, Any]
        ``method``, ``bounds`` (``[lower, upper]`` of each clipped column) and
        ``unclipped_columns`` (the columns whose bounds are -inf to +inf).
    """
    record: dict[str, Any] = {"method": method, "bounds": {}, "unclipped_columns": []}
    if bounds is None:
        return record
    for name, lower, upper in zip(columns, *bounds, strict=True):
        if np.isinf(lower) and np.isinf(upper):
            record["unclipped_columns"].append(name)
        else:
            record["bounds"][name] = [float(lower), float(upper)]
    return record
