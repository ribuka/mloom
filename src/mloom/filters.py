"""Row filters (``filters``, ``train_filters``): checking their definition and evaluating them.

See "行のフィルタ" in ``docs/spec.md``. A filter is a list of conditions
``{"column", "op", "value"}`` that are combined with AND. The definition is
plain data (no polars expression), so that the record and the model in
``model.pkl`` hold it as it was given and a reader needs neither mloom nor
polars.
"""

import json
import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from mloom.dtypes import BOOLEAN, DATE, DATETIME, NUMBER, STRING, value_kind
from mloom.errors import InputError

Condition = dict[str, Any]
Conditions = Sequence[Condition]

COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")
NULL_OPS = ("is_null", "is_not_null")
STRING_OPS = ("str_contains_all", "str_contains_any")
OPS = (*COMPARISONS, "is_in", "is_between", *NULL_OPS, *STRING_OPS)
FILE_KEYS = ("filters", "train_filters")
_CONDITION_KEYS = {"column", "op", "value"}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NANOSECONDS = {"ns": 1, "us": 1_000, "ms": 1_000_000}
# The UTC offset at the end of an ISO 8601 datetime: Z, +HH, +HH:MM, +HH:MM:SS (and with a fraction).
_OFFSET = re.compile(r"(?:Z|[+-]\d{2}(?::?\d{2}(?::?\d{2}(?:[.,]\d+)?)?)?)$")
_VALUE_FORMS = {
    "scalar": "a number, a string or a boolean",
    "list": "a non-empty list of numbers, strings or booleans",
    "pair": "a list [lower, upper]",
    "strings": "a non-empty list of strings",
}


def load_filters(path: Path) -> tuple[tuple[Condition, ...], tuple[Condition, ...]]:
    """Read the filters file of the CLI (``--filters``).

    Parameters
    ----------
    path : Path
        JSON file ``{"filters": [...], "train_filters": [...]}``; both keys
        may be left out.

    Returns
    -------
    tuple[tuple[Condition, ...], tuple[Condition, ...]]
        ``filters`` and ``train_filters``, each checked by
        ``check_conditions``.

    Raises
    ------
    InputError
        When the file is missing, is not valid JSON, or is not in that format.
    """
    if not path.is_file():
        raise InputError(f"filters file not found: {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InputError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(content, dict):
        raise InputError(f'{path} must be an object with "filters" and "train_filters" lists')
    unknown = sorted(set(content) - set(FILE_KEYS))
    if unknown:
        raise InputError(f"{path} has unknown keys {unknown}; the keys are {', '.join(FILE_KEYS)}")
    filters, train_filters = (check_conditions(content.get(key, []), key) for key in FILE_KEYS)
    return filters, train_filters


def check_conditions(conditions: object, key: str) -> tuple[Condition, ...]:
    """Check the form of the conditions; this needs no data.

    Parameters
    ----------
    conditions : object
        A list (or tuple) of conditions.
    key : str
        ``"filters"`` or ``"train_filters"``, for the error messages.

    Returns
    -------
    tuple[Condition, ...]
        The conditions.

    Raises
    ------
    InputError
        When a condition has an unknown key or ``op``, or a ``value`` whose
        form does not fit its ``op``.
    """
    if not isinstance(conditions, list | tuple):
        raise InputError(f"{key} must be a list of conditions; got {conditions!r}")
    for index, condition in enumerate(conditions):
        _check_condition(condition, f"{key}[{index}]")
    return tuple(conditions)


def filter_mask(frame: pl.DataFrame, conditions: Conditions, key: str) -> pl.Series:
    """Tell which rows meet all the conditions.

    Parameters
    ----------
    frame : pl.DataFrame
        The data.
    conditions : Conditions
        Conditions checked by ``check_conditions``.
    key : str
        ``"filters"`` or ``"train_filters"``, for the error messages.

    Returns
    -------
    pl.Series
        Boolean of each row, without nulls: a row whose value is missing does
        not meet a comparison or a string condition. All true when there is
        no condition.

    Raises
    ------
    InputError
        When a column is not in the data, or a value does not fit the type
        of its column.
    """
    if not conditions:
        return pl.repeat(True, frame.height, eager=True)
    expressions = [
        _expression(condition, frame.schema, f"{key}[{index}]") for index, condition in enumerate(conditions)
    ]
    return frame.select(pl.all_horizontal(expressions)).to_series()


def apply_filters(frame: pl.DataFrame, conditions: Conditions, key: str) -> pl.DataFrame:
    """Keep the rows that meet all the conditions.

    Parameters
    ----------
    frame : pl.DataFrame
        The data.
    conditions : Conditions
        Conditions checked by ``check_conditions``.
    key : str
        ``"filters"`` or ``"train_filters"``, for the error messages.

    Returns
    -------
    pl.DataFrame
        The rows that meet the conditions, in their order.

    Raises
    ------
    InputError
        When ``filter_mask`` raises, or no row is left.
    """
    kept = frame.filter(filter_mask(frame, conditions, key))
    if kept.height == 0:
        raise InputError(f"no row is left after applying {key}")
    return kept


def _check_condition(condition: object, where: str) -> None:
    if not isinstance(condition, dict):
        raise InputError(f'{where} must be an object {{"column", "op", "value"}}; got {condition!r}')
    unknown = sorted(set(condition) - _CONDITION_KEYS)
    if unknown:
        raise InputError(f"{where} has unknown keys {unknown}")
    if not isinstance(condition.get("column"), str):
        raise InputError(f'{where} needs a string "column"')
    op = condition.get("op")
    if op not in OPS:
        raise InputError(f'{where} has an unknown op {op!r}; the ops are {", ".join(OPS)}')
    if op in NULL_OPS:
        if "value" in condition:
            raise InputError(f"{where}: op {op!r} takes no value")
        return
    if "value" not in condition:
        raise InputError(f"{where}: op {op!r} needs a value")
    if not _value_fits(op, condition["value"]):
        raise InputError(f"{where}: op {op!r} needs {_VALUE_FORMS[_value_form(op)]}; got {condition['value']!r}")


def _value_form(op: str) -> str:
    if op in COMPARISONS:
        return "scalar"
    if op == "is_in":
        return "list"
    if op == "is_between":
        return "pair"
    return "strings"


def _value_fits(op: str, value: object) -> bool:
    form = _value_form(op)
    if form == "scalar":
        return _is_scalar(value)
    if not isinstance(value, list | tuple) or not value:
        return False
    if form == "strings":
        return all(isinstance(item, str) for item in value)
    return all(_is_scalar(item) for item in value) and (form == "list" or len(value) == 2)


def _is_scalar(value: object) -> bool:
    return isinstance(value, bool | int | float | str)


def _expression(condition: Condition, schema: pl.Schema, where: str) -> pl.Expr:
    name, op = condition["column"], condition["op"]
    if name not in schema:
        raise InputError(f"{where}: column {name!r} is not in the data")
    dtype = schema[name]
    column = pl.col(name)
    if dtype.is_float():
        column = column.fill_nan(None)  # NaN is missing, as for the features
    elif value_kind(dtype) == STRING:
        column = column.cast(pl.String)
    if op == "is_null":
        return column.is_null()
    if op == "is_not_null":
        return column.is_not_null()
    if op in STRING_OPS:
        return _string_expression(column, op, condition["value"], dtype, where)
    literals = [_literal(item, dtype, where) for item in _as_list(op, condition["value"])]
    if op == "is_in":
        matched = pl.any_horizontal([column.eq(literal) for literal in literals])
    elif op == "is_between":
        matched = column.ge(literals[0]) & column.le(literals[1])
    else:
        matched = getattr(column, op)(literals[0])
    return matched.fill_null(value=False)


def _as_list(op: str, value: Any) -> list[Any]:  # noqa: ANN401 - a value of a condition
    return [value] if op in COMPARISONS else list(value)


def _string_expression(column: pl.Expr, op: str, values: list[str], dtype: pl.DataType, where: str) -> pl.Expr:
    if value_kind(dtype) != STRING:
        raise InputError(f"{where}: op {op!r} needs a string column; the column is {dtype}")
    contains = [column.str.contains(value, literal=True) for value in values]
    matched = pl.all_horizontal(contains) if op == "str_contains_all" else pl.any_horizontal(contains)
    return matched.fill_null(value=False)


def _literal(value: Any, dtype: pl.DataType, where: str) -> pl.Expr:  # noqa: ANN401 - a value of a condition
    kind = value_kind(dtype)
    expected = {NUMBER: (int, float), BOOLEAN: (bool,), STRING: (str,), DATETIME: (str,), DATE: (str,)}.get(kind)
    if expected is None:
        raise InputError(f"{where}: a column of type {dtype} cannot be compared with a value")
    if not isinstance(value, expected) or (kind == NUMBER and isinstance(value, bool)):
        raise InputError(f"{where}: value {value!r} does not fit the column of type {dtype}")
    if isinstance(dtype, pl.Datetime):
        return _datetime_literal(value, dtype, where)
    if kind == DATE:
        return pl.lit(_parse(date.fromisoformat, value, "an ISO 8601 date (YYYY-MM-DD)", where))
    return pl.lit(value)


def _datetime_literal(value: str, dtype: pl.Datetime, where: str) -> pl.Expr:
    moment = _parse(datetime.fromisoformat, value, "an ISO 8601 datetime", where)
    if dtype.time_zone is None and moment.tzinfo is not None:
        raise InputError(f"{where}: value {value!r} has a UTC offset, but the column has no time zone")
    if dtype.time_zone is not None and moment.tzinfo is None:
        raise InputError(f"{where}: value {value!r} needs a UTC offset; the column is in {dtype.time_zone}")
    # Count in the unit of the column from the string itself: fromisoformat drops the digits below microseconds.
    ticks, rest = divmod(_epoch_nanoseconds(moment, value, where), _NANOSECONDS[dtype.time_unit])
    if rest:
        raise InputError(f"{where}: value {value!r} is finer than the time unit {dtype.time_unit} of the column")
    if dtype.time_zone is None:
        return pl.lit(ticks, dtype=pl.Int64).cast(pl.Datetime(dtype.time_unit))
    return pl.lit(ticks, dtype=pl.Int64).cast(pl.Datetime(dtype.time_unit, "UTC")).dt.convert_time_zone(dtype.time_zone)


def _epoch_nanoseconds(moment: datetime, value: str, where: str) -> int:
    # Nanoseconds since 1970-01-01 (UTC for a value with an offset, the wall time otherwise).
    body = value
    if moment.tzinfo is not None:
        offset = _OFFSET.search(value)
        if offset is None or any(mark in offset.group() for mark in ".,"):
            raise InputError(f"{where}: value {value!r} needs a UTC offset of whole seconds (Z, +09:00 and so on)")
        body = value[: offset.start()]
    fraction = re.search(r"[.,](\d+)$", body)
    below_microseconds = fraction.group(1)[6:] if fraction else ""
    if below_microseconds.rstrip("0")[3:]:
        raise InputError(f"{where}: value {value!r} is finer than a nanosecond")
    extra = int(below_microseconds[:3].ljust(3, "0"))
    since = moment.replace(tzinfo=moment.tzinfo or UTC) - _EPOCH
    return ((since.days * 86_400 + since.seconds) * 1_000_000 + since.microseconds) * 1_000 + extra


def _parse[T](parse: Callable[[str], T], value: str, form: str, where: str) -> T:
    try:
        return parse(value)
    except ValueError as error:
        raise InputError(f"{where}: value {value!r} is not {form}") from error
