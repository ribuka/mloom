"""Encoding the categorical candidate columns as integers, for the categorical features of LightGBM.

See "エンコード" in ``docs/spec.md``. The categories of a column are its
values on the statistic rows, as strings and without null, in ascending
order; a value is encoded as its position among them, and null and any other
value as NaN (missing).

The final model encodes the strings with an ``OrdinalEncoder`` fitted on the
statistic rows of the whole period (``fit_encoder``), and the feature
matrices of a run hold its codes. A CV fold encodes those codes again with
the categories of its own statistic rows (``fold_encoder``): the codes are
in the order of the strings, so this gives the codes an encoder fitted on the
strings of those rows would give.
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import OrdinalEncoder

# A category no code ever is, for a column without a value on the statistic rows of a fold.
_NO_CODE = -1.0


def _encoder(categories: list[Any]) -> OrdinalEncoder:
    return OrdinalEncoder(
        categories=categories, handle_unknown="use_encoded_value", unknown_value=np.nan, dtype=np.float64
    )


def _as_strings(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return frame.select(pl.col(name).cast(pl.String) for name in columns)


def fit_encoder(statistics: list[pl.DataFrame], columns: list[str]) -> OrdinalEncoder:
    """Fit the encoder of the final model on the statistic rows.

    Parameters
    ----------
    statistics : list[pl.DataFrame]
        The statistic rows (the labeled rows and the unlabeled rows), with
        every column of ``columns``.
    columns : list[str]
        The categorical candidate columns, at least one; each has a value on
        the statistic rows (``mloom.columns.plan_columns`` excludes
        ``all_null``).

    Returns
    -------
    OrdinalEncoder
        The fitted encoder; it takes a table with ``columns`` (String,
        Categorical or Enum).
    """
    rows = pl.concat([_as_strings(frame, columns) for frame in statistics])
    categories = [sorted(rows[name].drop_nulls().unique().to_list()) for name in columns]
    return _encoder(categories).fit(rows)


def encode(encoder: OrdinalEncoder | None, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Encode the categorical candidate columns of a table.

    Parameters
    ----------
    encoder : OrdinalEncoder | None
        The encoder from ``fit_encoder``; ``None`` when ``columns`` is empty.
    frame : pl.DataFrame
        Data with every column of ``columns``.
    columns : list[str]
        The categorical candidate columns.

    Returns
    -------
    np.ndarray
        The codes, of shape ``(frame.height, len(columns))``.
    """
    if encoder is None or frame.height == 0:  # OrdinalEncoder takes no empty table
        return np.empty((frame.height, len(columns)))
    return np.asarray(encoder.transform(frame.select(columns)), dtype=np.float64)


def fold_encoder(codes: np.ndarray) -> OrdinalEncoder:
    """Make the encoder of the codes of a CV fold, to be fitted on its statistic rows.

    Parameters
    ----------
    codes : np.ndarray
        The codes (``encode``) of the categorical candidate columns on the
        statistic rows of the fold.

    Returns
    -------
    OrdinalEncoder
        An unfitted encoder of the codes whose categories are those on
        ``codes``. A column without a code there turns every value into NaN.
    """
    categories = []
    for column in codes.T:
        found = np.unique(column[~np.isnan(column)])
        # OrdinalEncoder takes no empty list of categories.
        categories.append(found if found.size else np.array([_NO_CODE]))
    return _encoder(categories)


def describe_categorical(encoder: OrdinalEncoder | None, columns: list[str]) -> dict[str, Any]:
    """Describe the categories of the final model (``preprocessing.categorical``).

    Parameters
    ----------
    encoder : OrdinalEncoder | None
        The encoder from ``fit_encoder``; ``None`` when ``columns`` is empty.
    columns : list[str]
        The categorical candidate columns.

    Returns
    -------
    dict[str, Any]
        ``{"n_categories": <number of categories>}`` of each column.
    """
    if encoder is None:
        return {}
    return {name: {"n_categories": len(found)} for name, found in zip(columns, encoder.categories_, strict=True)}
