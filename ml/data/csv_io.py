"""Version-1 streaming CSV parsing with formatting-stable identity.

Only UTF-8 comma CSV is supported. Values stay strings: empty cells represent
missing input; NA, numeric spellings and whitespace are preserved. Canonical
identity includes ordered headers/records, ignores BOM, quoting and line-ending
style, and normalizes embedded CR/CRLF to LF. No statistical inference occurs.
Paths are internal service inputs, never names supplied directly by HTTP clients.
"""

import csv
import hashlib
import io
import json
import os
import stat
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ml.contracts import ResourceLimits
from ml.data.validation import CsvValidationError


@dataclass(frozen=True, slots=True)
class ValidatedDataset:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...] = field(repr=False)
    byte_count: int
    raw_sha256: str
    canonical_sha256: str
    parse_policy_version: int = 1

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _signature(info: os.stat_result) -> tuple[int, int, int, int]:
    # Python 3.12 Windows stat/fstat disagree on ctime after file rewrites.
    # Device, file identity, size and modification time share the same semantics.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class _BoundedSource(io.RawIOBase):
    """Hash exactly the bytes decoded, with at most one byte beyond the cap."""

    def __init__(self, source, limit: int) -> None:
        self.source = source
        self.limit = limit
        self.byte_count = 0
        self.digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        chunk = self.source.read(min(len(buffer), self.limit - self.byte_count + 1))
        self.byte_count += len(chunk)
        if self.byte_count > self.limit:
            raise CsvValidationError("upload_too_large", "CSV exceeds the byte limit.")
        self.digest.update(chunk)
        buffer[:len(chunk)] = chunk
        return len(chunk)


def _checked_lines(text: io.TextIOWrapper, limits: ResourceLimits) -> Iterator[str]:
    # csv.reader(strict=True) alone accepts quotes embedded in unquoted fields.
    state = "start"
    columns, cell_bytes, record_bytes = 1, 0, 0
    parts: list[str] = []
    for chunk in iter(lambda: text.read(4096), ""):
        start = 0
        for index, char in enumerate(chunk):
            if ord(char) < 32 and char not in "\t\n":
                raise CsvValidationError("binary_input", "CSV contains unsupported control characters.")
            if ord(char) == 127:
                raise CsvValidationError("binary_input", "CSV contains unsupported control characters.")
            if state == "quoted":
                if char == '"':
                    state = "closed"
                else:
                    cell_bytes += len(char.encode("utf-8"))
            elif state == "closed":
                if char == '"':
                    state = "quoted"
                    cell_bytes += 1
                elif char in ",\n":
                    state = "start"
                else:
                    raise CsvValidationError("invalid_csv", "CSV quoting is invalid.")
            elif char == '"':
                if state != "start":
                    raise CsvValidationError("invalid_csv", "CSV quoting is invalid.")
                state = "quoted"
            elif char in ",\n":
                state = "start"
            else:
                state = "unquoted"
                cell_bytes += len(char.encode("utf-8"))
            record_bytes += len(char.encode("utf-8"))
            if cell_bytes > limits.max_cell_bytes:
                raise CsvValidationError("cell_too_large", "CSV exceeds the cell byte limit.")
            if record_bytes * 3 > limits.preprocessing_bytes:
                raise CsvValidationError("parsed_memory_limit", "Parsed CSV exceeds the memory budget.")
            if state == "start":
                cell_bytes = 0
                if char == ",":
                    columns += 1
                    if columns > limits.max_columns:
                        raise CsvValidationError("too_many_columns", "CSV exceeds the column limit.")
                elif char == "\n":
                    columns, record_bytes = 1, 0
            if char == "\n":
                parts.append(chunk[start:index + 1])
                yield "".join(parts)
                parts.clear()
                start = index + 1
        if start < len(chunk):
            parts.append(chunk[start:])
    if parts:
        yield "".join(parts)


def validate_csv(
    path: Path, limits: ResourceLimits, *, expected_root: Path | None = None,
) -> ValidatedDataset:
    """Parse and hash a private file in one bounded pass; reject substitution.

    A single stream avoids rereading a mutable source for hashing versus parsing.
    Device/inode/size/timestamp checks detect ordinary concurrent replacement;
    this is not a defense against a malicious owner of the local machine.
    """
    try:
        if ".." in path.parts or path.is_symlink():
            raise CsvValidationError("unsafe_path", "CSV path is not permitted.")
        resolved = path.resolve(strict=True)
        if resolved != path.absolute():
            raise CsvValidationError("unsafe_path", "CSV path cannot use redirected components.")
        if expected_root is not None and not resolved.is_relative_to(expected_root.resolve(strict=True)):
            raise CsvValidationError("unsafe_path", "CSV path is outside private storage.")
        initial = resolved.stat()
        if not stat.S_ISREG(initial.st_mode):
            raise CsvValidationError("unsafe_path", "CSV source must be a regular file.")
        if initial.st_size > limits.upload_bytes:
            raise CsvValidationError("upload_too_large", "CSV exceeds the byte limit.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(resolved, flags), "rb") as source:
            if _signature(os.fstat(source.fileno())) != _signature(initial):
                raise CsvValidationError("source_changed", "CSV source changed during validation.")
            bounded = _BoundedSource(source, limits.upload_bytes)
            with io.TextIOWrapper(io.BufferedReader(bounded), encoding="utf-8-sig", newline=None) as text:
                reader = csv.reader(_checked_lines(text, limits), strict=True)
                columns = tuple(next(reader, ()))
                if not columns or any(not name.strip() for name in columns):
                    raise CsvValidationError("invalid_header", "CSV requires nonempty column names.")
                if len(set(columns)) != len(columns):
                    raise CsvValidationError("duplicate_header", "CSV column names must be unique.")
                if len(columns) > limits.max_columns:
                    raise CsvValidationError("too_many_columns", "CSV exceeds the column limit.")
                canonical = hashlib.sha256(b"modelarena-csv-v1\n")
                rows: list[tuple[str, ...]] = []
                memory_bytes = 0

                def record(values: tuple[str, ...]) -> None:
                    nonlocal memory_bytes
                    if any(len(cell.encode("utf-8")) > limits.max_cell_bytes for cell in values):
                        raise CsvValidationError("cell_too_large", "CSV exceeds the cell byte limit.")
                    # Conservative allowance for tuple/list growth and canonical encoding.
                    memory_bytes += 3 * (sys.getsizeof(values) + 16
                                         + sum(sys.getsizeof(cell) for cell in values))
                    if memory_bytes > limits.preprocessing_bytes:
                        raise CsvValidationError("parsed_memory_limit", "Parsed CSV exceeds the memory budget.")
                    canonical.update(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                    canonical.update(b"\n")

                record(columns)
                for row in reader:
                    if len(rows) >= limits.max_rows:
                        raise CsvValidationError("too_many_rows", "CSV exceeds the row limit.")
                    if len(row) != len(columns):
                        raise CsvValidationError("irregular_rows", "CSV rows must match the header width.")
                    values = tuple(row)
                    record(values)
                    rows.append(values)
                if not rows:
                    raise CsvValidationError("empty_data", "CSV requires at least one data row.")
                if (_signature(os.fstat(source.fileno())) != _signature(initial)
                        or _signature(resolved.stat()) != _signature(initial)):
                    raise CsvValidationError("source_changed", "CSV source changed during validation.")
                return ValidatedDataset(columns, tuple(rows), bounded.byte_count,
                                        bounded.digest.hexdigest(), canonical.hexdigest())
    except CsvValidationError:
        raise
    except (OSError, RuntimeError):
        raise CsvValidationError("unreadable_source", "CSV source is unavailable or unsafe.") from None
    except (UnicodeError, csv.Error):
        raise CsvValidationError("invalid_csv", "CSV must be valid UTF-8 with well-formed records.") from None
