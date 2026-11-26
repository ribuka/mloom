"""The kind of the values of a column, as the row filters and the row hashes read it.

Both compare values across tables whose column types may differ in width or
encoding (``Int32`` and ``Float64``, ``Categorical`` and ``String``), so they
read a column by its kind instead of its exact type.
"""

import polars as pl

NUMBER = "number"
BOOLEAN = "boolean"
STRING = "string"
DATETIME = "datetime"
DATE = "date"


def value_kind(dtype: pl.DataType) -> str | None:
    """Tell the kind of the values of a column type.

    Parameters
    ----------
    dtype : pl.DataType
        Type of the column.

    Returns
    -------
    str | None
        ``NUMBER`` (integers, floats, decimals), ``BOOLEAN``, ``STRING``
        (``String``, ``Categorical``, ``Enum``), ``DATETIME`` or ``DATE``;
        ``None`` for any other type (``Duration``, ``Time``, ``List`` and so on).
    """
    if dtype == pl.Boolean:
        return BOOLEAN
    if dtype.is_numeric():
        return NUMBER
    if dtype == pl.String or isinstance(dtype, pl.Categorical | pl.Enum):
        return STRING
    if isinstance(dtype, pl.Datetime):
        return DATETIME
    if dtype == pl.Date:
        return DATE
    return None
