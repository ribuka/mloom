"""Tests of the row filters: their checks and what each op keeps."""

from pathlib import Path
from typing import Literal

import polars as pl
import pytest

from mloom.errors import InputError
from mloom.filters import (
    OPS,
    apply_filters,
    check_conditions,
    filter_mask,
    load_filters,
)

NAN = float("nan")


@pytest.fixture
def table() -> pl.DataFrame:
    """Five rows with a missing value in each column (row 3), and a NaN (row 4)."""
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, None, NAN],
            "n": pl.Series([1, 2, 3, None, 5], dtype=pl.Int32),
            "s": ["steady a", "a.b", "start", None, "steady b"],
            "k": pl.Series(["p", "q", "p", None, "q"], dtype=pl.Categorical),
            "b": [True, False, True, None, False],
        }
    )


def _kept(table: pl.DataFrame, *conditions: dict) -> list[bool]:
    return filter_mask(table, check_conditions(list(conditions), "filters"), "filters").to_list()


OP_CASES = [
    ({"column": "x", "op": "eq", "value": 2}, [False, True, False, False, False]),
    ({"column": "x", "op": "ne", "value": 2}, [True, False, True, False, False]),
    ({"column": "x", "op": "lt", "value": 2}, [True, False, False, False, False]),
    ({"column": "x", "op": "le", "value": 2}, [True, True, False, False, False]),
    ({"column": "x", "op": "gt", "value": 2}, [False, False, True, False, False]),
    ({"column": "x", "op": "ge", "value": 2.0}, [False, True, True, False, False]),
    ({"column": "n", "op": "is_in", "value": [1, 3, 5]}, [True, False, True, False, True]),
    ({"column": "x", "op": "is_between", "value": [1.5, 3]}, [False, True, True, False, False]),
    ({"column": "x", "op": "is_null"}, [False, False, False, True, True]),
    ({"column": "x", "op": "is_not_null"}, [True, True, True, False, False]),
    ({"column": "s", "op": "str_contains_all", "value": ["steady", " a"]}, [True, False, False, False, False]),
    ({"column": "s", "op": "str_contains_any", "value": ["a.", "b"]}, [False, True, False, False, True]),
]


@pytest.mark.parametrize(("condition", "expected"), OP_CASES)
def test_each_op_keeps_its_rows(table: pl.DataFrame, condition: dict, expected: list[bool]) -> None:
    """Each op keeps the rows of its meaning; a missing value or NaN never meets a value condition."""
    assert _kept(table, condition) == expected


def test_every_op_is_tested() -> None:
    """``OP_CASES`` covers each of the 12 ops of the spec."""
    assert len(OPS) == 12
    assert {condition["op"] for condition, _ in OP_CASES} == set(OPS)


def test_conditions_are_combined_with_and(table: pl.DataFrame) -> None:
    """Rows must meet every condition; no condition keeps every row."""
    assert _kept(table, {"column": "x", "op": "ge", "value": 2}, {"column": "s", "op": "eq", "value": "start"}) == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert _kept(table) == [True] * 5


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
    ({"column": "s", "op": "str_contains_any", "value": ["."]}, [False, True, False, False, False]),
    ({"column": "k", "op": "eq", "value": "q"}, [False, True, False, False, True]),
    ({"column": "k", "op": "str_contains_any", "value": ["p"]}, [True, False, True, False, False]),
    ({"column": "b", "op": "eq", "value": True}, [True, False, True, False, False]),
    ],
)
def test_strings_match_literally_and_categoricals_as_strings(
    table: pl.DataFrame, condition: dict, expected: list[bool]
) -> None:
    """``.`` is not a pattern, a categorical column compares as strings, and booleans compare with booleans."""
    assert _kept(table, condition) == expected


@pytest.fixture
def times() -> pl.DataFrame:
    """Three times as naive, Tokyo and nanosecond datetimes, and as dates."""
    naive = pl.Series("naive", ["2026-09-30T12:00:00", "2026-10-01T00:00:00", "2026-10-02T00:00:00"]).str.to_datetime()
    return pl.DataFrame(
        {
            "naive": naive,
            "tokyo": naive.dt.replace_time_zone("Asia/Tokyo"),
            "nanos": naive.dt.cast_time_unit("ns"),
            "day": naive.dt.date(),
        }
    )


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
    ({"column": "naive", "op": "lt", "value": "2026-10-01T00:00:00"}, [True, False, False]),
    ({"column": "naive", "op": "le", "value": "2026-10-01"}, [True, True, False]),
    ({"column": "nanos", "op": "ge", "value": "2026-10-01T00:00:00"}, [False, True, True]),
        # 2026-10-01T00:00+09:00 is 2026-09-30T15:00Z; the column holds Tokyo wall times.
    ({"column": "tokyo", "op": "lt", "value": "2026-09-30T15:00:00+00:00"}, [True, False, False]),
    ({"column": "tokyo", "op": "eq", "value": "2026-10-01T00:00:00+09:00"}, [False, True, False]),
    ({"column": "day", "op": "is_between", "value": ["2026-10-01", "2026-10-02"]}, [False, True, True]),
    ({"column": "day", "op": "is_in", "value": ["2026-09-30"]}, [True, False, False]),
    ],
)
def test_datetimes_compare_in_the_type_of_the_column(times: pl.DataFrame, condition: dict, expected: list) -> None:
    """ISO 8601 strings are converted to the type of a datetime or date column."""
    assert _kept(times, condition) == expected


@pytest.mark.parametrize(
    ("condition", "message"),
    [
    ({"column": "tokyo", "op": "lt", "value": "2026-10-01T00:00:00"}, "needs a UTC offset"),
    ({"column": "naive", "op": "lt", "value": "2026-10-01T00:00:00+09:00"}, "the column has no time zone"),
    ({"column": "naive", "op": "lt", "value": "October"}, "not an ISO 8601 datetime"),
    ({"column": "day", "op": "lt", "value": "2026-10-01T00:00:00"}, "not an ISO 8601 date"),
    ({"column": "day", "op": "lt", "value": 20261001}, "does not fit the column of type Date"),
    ],
)
def test_invalid_datetime_values_raise(times: pl.DataFrame, condition: dict, message: str) -> None:
    """A datetime value with a missing or unexpected offset, or not in ISO 8601, raises ``InputError``."""
    with pytest.raises(InputError, match=message):
        _kept(times, condition)


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ("x > 1", "must be an object"),
    ({"column": "x", "op": "gt", "value": 1, "note": "?"}, "unknown keys"),
        ({"op": "gt", "value": 1}, 'needs a string "column"'),
    ({"column": "x", "op": "greater", "value": 1}, "unknown op"),
    ({"column": "x", "op": "is_null", "value": 1}, "takes no value"),
    ({"column": "x", "op": "gt"}, "needs a value"),
    ({"column": "x", "op": "gt", "value": [1]}, "needs a number, a string or a boolean"),
    ({"column": "x", "op": "gt", "value": None}, "needs a number, a string or a boolean"),
    ({"column": "x", "op": "is_in", "value": 1}, "needs a non-empty list"),
    ({"column": "x", "op": "is_in", "value": []}, "needs a non-empty list"),
    ({"column": "x", "op": "is_between", "value": [1, 2, 3]}, r"needs a list \[lower, upper\]"),
    ({"column": "s", "op": "str_contains_any", "value": "a"}, "needs a non-empty list of strings"),
    ({"column": "s", "op": "str_contains_all", "value": ["a", 1]}, "needs a non-empty list of strings"),
    ],
)
def test_invalid_condition_raises_without_data(condition: object, message: str) -> None:
    """The form of a condition is checked without the data."""
    with pytest.raises(InputError, match=message):
        check_conditions([condition], "filters")


def test_conditions_must_be_a_list() -> None:
    """Conditions are a list, not a single condition."""
    with pytest.raises(InputError, match="must be a list of conditions"):
        check_conditions({"column": "x", "op": "is_null"}, "train_filters")


@pytest.mark.parametrize(
    ("condition", "message"),
    [
    ({"column": "missing", "op": "is_null"}, "column 'missing' is not in the data"),
    ({"column": "x", "op": "gt", "value": "1"}, "does not fit the column of type Float64"),
    ({"column": "n", "op": "eq", "value": True}, "does not fit the column of type Int32"),
    ({"column": "b", "op": "eq", "value": 1}, "does not fit the column of type Boolean"),
    ({"column": "s", "op": "is_in", "value": ["a", 1]}, "does not fit the column of type String"),
    ({"column": "x", "op": "str_contains_any", "value": ["1"]}, "needs a string column"),
    ({"column": "d", "op": "gt", "value": 1}, "cannot be compared with a value"),
    ],
)
def test_condition_that_does_not_fit_the_data_raises(table: pl.DataFrame, condition: dict, message: str) -> None:
    """A missing column, or a value of another type than its column, raises ``InputError``."""
    data = table.with_columns(d=pl.duration(days=pl.col("n")))
    with pytest.raises(InputError, match=f"filters\\[0\\]: .*{message}"):
        _kept(data, condition)


def test_apply_filters_raises_when_no_row_is_left(table: pl.DataFrame) -> None:
    """Filters that keep no row raise ``InputError``."""
    assert apply_filters(table, [{"column": "x", "op": "gt", "value": 2}], "filters")["x"].to_list() == [3.0]
    with pytest.raises(InputError, match="no row is left after applying train_filters"):
        apply_filters(table, [{"column": "x", "op": "gt", "value": 9}], "train_filters")


def test_load_filters_reads_both_lists(tmp_path: Path) -> None:
    """The file gives ``filters`` and ``train_filters``; a key left out is an empty list."""
    path = tmp_path / "filters.json"
    path.write_text('{"train_filters": [{"column": "t", "op": "lt", "value": "2026-10-01"}]}', encoding="utf-8")
    assert load_filters(path) == ((), ({"column": "t", "op": "lt", "value": "2026-10-01"},))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "filters file not found"),
        ("{", "is not valid JSON"),
        ("[]", "must be an object"),
        ('{"filter": []}', r"unknown keys \['filter'\]"),
        ('{"filters": [{"column": "x", "op": "like", "value": 1}]}', r"filters\[0\] has an unknown op"),
    ],
)
def test_invalid_filters_file_raises(tmp_path: Path, content: str | None, message: str) -> None:
    """A missing file, invalid JSON, an unknown key or an invalid condition raises ``InputError``."""
    path = tmp_path / "filters.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(InputError, match=message):
        load_filters(path)



def test_nanoseconds_of_the_value_are_kept() -> None:
    """A ``Datetime("ns")`` column compares with the nanoseconds of the value (``fromisoformat`` drops them)."""
    nanos = pl.Series("t", [1, 0, 1_500]).cast(pl.Datetime("ns"))
    data = pl.DataFrame({"t": nanos, "z": nanos.dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Tokyo")})
    assert _kept(data, {"column": "t", "op": "eq", "value": "1970-01-01T00:00:00.000000001"}) == [True, False, False]
    assert _kept(data, {"column": "t", "op": "ge", "value": "1970-01-01T00:00:00,0000015"}) == [False, False, True]
    assert _kept(data, {"column": "z", "op": "eq", "value": "1970-01-01T09:00:00.000000001+09:00"}) == [True, False, False]


@pytest.mark.parametrize(
    ("unit", "value"),
    [("ms", "2026-10-01T00:00:00.0015"), ("us", "2026-10-01T00:00:00.0000001"), ("ns", "2026-10-01T00:00:00.0000000001")],
)
def test_value_finer_than_the_time_unit_raises(unit: Literal["ns", "us", "ms"], value: str) -> None:
    """A value that the time unit of the column cannot hold raises ``InputError`` instead of being truncated."""
    data = pl.DataFrame({"t": pl.Series([0]).cast(pl.Datetime(unit))})
    with pytest.raises(InputError, match="is finer than"):
        _kept(data, {"column": "t", "op": "ge", "value": value})


def test_fraction_of_the_offset_is_not_read_as_the_fraction_of_the_time() -> None:
    """The fraction of seconds is read from the time, not the offset; an offset with a fraction raises."""
    utc = pl.DataFrame({"t": pl.Series([-3_600_000_000_500, -3_599_999_999_500]).cast(pl.Datetime("ns", "UTC"))})
    assert _kept(utc, {"column": "t", "op": "eq", "value": "1969-12-31T22:59:59.999999500Z"}) == [True, False]
    assert _kept(utc, {"column": "t", "op": "eq", "value": "1969-12-31T23:59:59.9999995+01:00:00"}) == [True, False]
    with pytest.raises(InputError, match="needs a UTC offset of whole seconds"):
        _kept(utc, {"column": "t", "op": "eq", "value": "1970-01-01T00:00:00+01:00:00.0000005"})
