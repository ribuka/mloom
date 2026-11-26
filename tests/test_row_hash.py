"""Tests of hashing rows by their values (``training_rows`` of ``model.pkl``)."""

import hashlib
import struct
from datetime import date

import polars as pl
import pytest

from mloom.errors import InputError
from mloom.row_hash import hash_rows

COLUMNS = ["x", "s", "t", "d", "b"]
# Changing these values makes every model.pkl saved before fail to match its training rows.
EXPECTED = [4054743693015717358, 17558426049854471538, 12384254604247643303]


@pytest.fixture
def table() -> pl.DataFrame:
    """Three rows of a number, a string, a datetime, a date and a boolean, with nulls."""
    return pl.DataFrame(
        {
            "x": [1.5, None, -0.0],
            "s": ["ab", None, "é"],
            "t": pl.Series(["2026-10-01T00:00:00", None, "1970-01-01T00:00:01"]).str.to_datetime(),
            "d": pl.Series(["2026-10-01", "1970-01-02", None]).str.to_date(),
            "b": [True, False, None],
        }
    )


def test_hashes_of_a_fixed_table_are_fixed(table: pl.DataFrame) -> None:
    """The hashes depend on the values only, by the normal form of ``mloom.row_hash``."""
    assert hash_rows(table, COLUMNS).tolist() == EXPECTED


def test_hash_follows_the_documented_encoding(table: pl.DataFrame) -> None:
    """The first row, encoded by hand as the module docstring says, gives the same hash."""
    days = date(2026, 10, 1).toordinal() - date(1970, 1, 1).toordinal()
    encoded = b"".join(
        [
            b"F" + struct.pack("<d", 1.5) + b"\x1f",
            b"S" + struct.pack("<Q", 2) + b"ab\x1f",
            b"T" + struct.pack("<q", days * 86_400 * 10**6) + b"\x1f",
            b"D" + struct.pack("<q", days) + b"\x1f",
            b"F" + struct.pack("<d", 1.0) + b"\x1f",
        ]
    )
    expected = int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")
    assert hash_rows(table, COLUMNS)[0] == expected


def test_types_of_the_same_kind_hash_alike(table: pl.DataFrame) -> None:
    """Integer and float, categorical and string, any time unit, and a Null column give the same hashes."""
    other = table.with_columns(
        pl.Series("x", [1.5, None, 0.0], dtype=pl.Float32),
        pl.col("s").cast(pl.Categorical),
        pl.col("t").dt.cast_time_unit("ns"),
    )
    assert hash_rows(other, COLUMNS).tolist() == EXPECTED
    integers = pl.DataFrame({"x": pl.Series([1, None], dtype=pl.Int32)})
    floats = pl.DataFrame({"x": [1.0, float("nan")]})
    assert hash_rows(integers, ["x"]).tolist() == hash_rows(floats, ["x"]).tolist()
    only_nulls = pl.DataFrame({"x": [None, None]})
    assert only_nulls.schema["x"] == pl.Null
    assert hash_rows(only_nulls, ["x"])[0] == hash_rows(floats, ["x"])[1]


def test_time_zone_columns_hash_by_utc() -> None:
    """A column with a time zone hashes the same instant alike, in any time zone."""
    utc = pl.DataFrame({"t": pl.Series(["2026-10-01T00:00:00"]).str.to_datetime().dt.replace_time_zone("UTC")})
    tokyo = utc.with_columns(pl.col("t").dt.convert_time_zone("Asia/Tokyo"))
    assert hash_rows(utc, ["t"]).tolist() == hash_rows(tokyo, ["t"]).tolist()


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"a": ["ab"], "b": [""]}, {"a": ["a"], "b": ["b"]}),
        ({"a": [1.0], "b": [2.0]}, {"a": [2.0], "b": [1.0]}),
        ({"a": ["1"], "b": [2.0]}, {"a": [1.0], "b": [2.0]}),
        ({"a": pl.Series([None], dtype=pl.Float64), "b": [1.0]}, {"a": [0.0], "b": [1.0]}),
    ],
)
def test_rows_that_differ_hash_differently(left: dict, right: dict) -> None:
    """Column boundaries, order, types and nulls are part of the hash."""
    assert hash_rows(pl.DataFrame(left), ["a", "b"]) != hash_rows(pl.DataFrame(right), ["a", "b"])


def test_unsupported_type_raises() -> None:
    """A column that is not a number, boolean, string, datetime or date raises ``InputError``."""
    data = pl.DataFrame({"d": pl.Series([1]).cast(pl.Duration("us"))})
    with pytest.raises(InputError, match="column 'd' is of type Duration"):
        hash_rows(data, ["d"])
