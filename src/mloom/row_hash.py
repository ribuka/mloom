"""Hashing rows by their values, to tell the rows a model was trained on (``training_rows``).

See "予測" in ``docs/spec.md``. The hash is made from a normal form of the
values, not from polars' ``hash_rows``, so that it stays the same across
versions of polars and across column types of the same kind (``Int32`` and
``Float64``, ``Categorical`` and ``String``, ``Datetime`` of any time unit).

Each value is encoded as a tag byte, a body and a separator byte ``0x1f``:

- null (and NaN, and every value of a column of type ``Null``): ``N`` and
  8 zero bytes;
- number or boolean: ``F`` and the IEEE 754 bits of the value as a
  little-endian Float64 (``-0.0`` as ``0.0``; +inf and -inf as they are);
- string: ``S``, the length of its UTF-8 bytes as a little-endian uint64, and
  the bytes;
- datetime: ``T`` and the microseconds since 1970-01-01 as a little-endian
  int64 (UTC for a column with a time zone; the wall time as it is otherwise);
- date: ``D`` and the days since 1970-01-01 as a little-endian int64.

A row is the encoded values of its columns in order, hashed with
``hashlib.blake2b(digest_size=8)`` into a uint64.
"""

import hashlib
from collections.abc import Sequence

import numpy as np
import polars as pl

from mloom.config import Config
from mloom.dtypes import DATE, DATETIME, STRING, value_kind
from mloom.errors import InputError

_SEPARATOR = 0x1F
_NULL = b"N" + bytes(8) + bytes([_SEPARATOR])
_TAGS = {DATETIME: b"T", DATE: b"D"}
_WIDTH = 10  # tag, 8 bytes of body, separator


def hash_columns(candidates: Sequence[str], config: Config) -> list[str]:
    """List the columns a row is hashed by, in their fixed order.

    Parameters
    ----------
    candidates : Sequence[str]
        Feature candidates, in the order of ``candidate_columns``.
    config : Config
        The config of the run.

    Returns
    -------
    list[str]
        Candidates, the target, then ``time_column`` and ``group_column``
        when given.
    """
    return [*candidates, config.target, *config.role_columns]


def check_hashable(schema: pl.Schema, columns: Sequence[str]) -> None:
    """Check that the columns have a type the hash takes.

    Parameters
    ----------
    schema : pl.Schema
        Schema of the data.
    columns : Sequence[str]
        Columns to hash; all in ``schema``.

    Raises
    ------
    InputError
        When a column is not a number, a boolean, a string (including
        ``Categorical`` and ``Enum``), a datetime, a date, or of type ``Null``.
    """
    for name in columns:
        # A column of only nulls (type Null), such as an empty target of new data, hashes as nulls.
        if value_kind(schema[name]) is None and schema[name] != pl.Null:
            raise InputError(
                f"column {name!r} is of type {schema[name]}; the rows can be matched only by numbers, "
                "booleans, strings, datetimes and dates"
            )


def hash_rows(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Hash each row by the values of the columns.

    Parameters
    ----------
    frame : pl.DataFrame
        Data that has every column of ``columns``.
    columns : Sequence[str]
        Columns to hash, in order.

    Returns
    -------
    np.ndarray
        uint64 hash of each row.

    Raises
    ------
    InputError
        When ``check_hashable`` raises.
    """
    check_hashable(frame.schema, columns)
    segments: list[list[bytes]] = []
    fixed: list[np.ndarray] = []
    for name in columns:
        if value_kind(frame.schema[name]) == STRING:
            if fixed:
                segments.append(_rows_of(np.hstack(fixed)))
                fixed = []
            segments.append(_encode_strings(frame[name]))
        else:
            fixed.append(_encode_fixed(frame[name]))
    if fixed:
        segments.append(_rows_of(np.hstack(fixed)))
    digests = b"".join(hashlib.blake2b(b"".join(parts), digest_size=8).digest() for parts in zip(*segments, strict=True))
    return np.frombuffer(digests, dtype="<u8").astype(np.uint64)


def _rows_of(block: np.ndarray) -> list[bytes]:
    return [row.tobytes() for row in np.ascontiguousarray(block)]


def _encode_fixed(series: pl.Series) -> np.ndarray:
    kind = value_kind(series.dtype)
    if kind in _TAGS:
        if kind == DATETIME:
            series = series.dt.cast_time_unit("us")
        null = series.is_null().to_numpy()
        body = series.to_physical().cast(pl.Int64).fill_null(0).to_numpy().astype("<i8")
        tag = _TAGS[kind]
    else:
        values = series.cast(pl.Float64).fill_nan(None).to_numpy()
        null = np.isnan(values)
        body = (np.where(null, 0.0, values) + 0.0).astype("<f8")  # + 0.0 turns -0.0 into 0.0
        tag = b"F"
    block = np.empty((series.len(), _WIDTH), dtype=np.uint8)
    block[:, 0] = tag[0]
    block[:, 1:9] = body.view(np.uint8).reshape(-1, 8)
    block[:, 9] = _SEPARATOR
    block[null] = np.frombuffer(_NULL, dtype=np.uint8)
    return block


def _encode_strings(series: pl.Series) -> list[bytes]:
    encoded = []
    for value in series.cast(pl.String).to_list():
        if value is None:
            encoded.append(_NULL)
        else:
            body = value.encode("utf-8")
            encoded.append(b"S" + len(body).to_bytes(8, "little") + body + bytes([_SEPARATOR]))
    return encoded
