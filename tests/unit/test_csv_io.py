"""Structural parsing must reject unsafe input before ML configuration."""

import hashlib
import tracemalloc
from dataclasses import replace
from pathlib import Path

import pytest

from ml.contracts import ResourceLimits
from ml.data.csv_io import validate_csv
from ml.data.validation import CsvValidationError


def parse(tmp_path, content: bytes, **limits):
    path = tmp_path / "input.csv"
    path.write_bytes(content)
    return validate_csv(path, replace(ResourceLimits(), **limits))


def test_preserves_values_missing_cells_and_raw_identity(tmp_path):
    payload = b'name,value,target\n"a,b",,1\nNA,001,0\n'
    data = parse(tmp_path, payload)
    assert data.columns == ("name", "value", "target")
    assert data.rows == (("a,b", "", "1"), ("NA", "001", "0"))
    assert data.row_count == 2
    assert data.byte_count == len(payload)
    assert data.raw_sha256 == hashlib.sha256(payload).hexdigest()
    assert "a,b" not in repr(data)


def test_formatting_does_not_reset_canonical_identity(tmp_path):
    first = parse(tmp_path, b'a,b\nx,1\ny,2\n')
    second = parse(tmp_path, b'\xef\xbb\xbf"a","b"\r\n"x","1"\r\n"y","2"')
    assert first.canonical_sha256 == second.canonical_sha256
    assert first.raw_sha256 != second.raw_sha256
    assert first.parse_policy_version == 1
    assert first.canonical_sha256 != parse(tmp_path, b'a,b\ny,2\nx,1\n').canonical_sha256
    assert first.canonical_sha256 != parse(tmp_path, b'a,c\nx,1\ny,2\n').canonical_sha256


def test_quoted_multiline_and_escaped_quotes(tmp_path):
    data = parse(tmp_path, b'a,b\r\n"two\r\nlines","say ""hi"""\r\n')
    assert data.rows == (("two\nlines", 'say "hi"'),)


@pytest.mark.parametrize("payload", [
    b'', b'\xef\xbb\xbf', b'a,b\n', b'\n', b'a,a\n1,2\n',
    b'a,\n1,2\n', b'a, \n1,2\n', b'a,b\n1\n', b'a,b\n1,2,3\n',
    b'a,b\n\n', b'a,b\n"unterminated,2', b'a,b\n"x"junk,2\n',
    b'a,b\nx"y,2\n', b'a,b\n\x00,2\n', b'a,b\n\xff,2\n',
    b'PK\x03\x04binary', b'\x1f\x8bcompressed', b'a,b\n\x01,2\n',
])
def test_rejects_invalid_csv_without_exposing_contents(tmp_path, payload):
    with pytest.raises(CsvValidationError) as caught:
        parse(tmp_path, payload)
    assert str(tmp_path) not in str(caught.value)
    assert caught.value.code


@pytest.mark.parametrize("payload,limits", [
    (b'a,b\n1,2\n', {"upload_bytes": 7}),
    (b'a,b\n1,2\n3,4\n', {"max_rows": 1}),
    (b'a,b\n1,2\n', {"max_columns": 1}),
    ('a\n\u00e9\u00e9\n'.encode(), {"max_cell_bytes": 3}),
    (b'long\nx\n', {"max_cell_bytes": 3}),
    (b'a,b\n1,2\n', {"preprocessing_bytes": 20}),
])
def test_each_resource_bound_is_enforced(tmp_path, payload, limits):
    with pytest.raises(CsvValidationError):
        parse(tmp_path, payload, **limits)


def test_exact_limits_are_accepted(tmp_path):
    assert parse(tmp_path, b'a,b\n1,2\n', upload_bytes=8, max_rows=1,
                 max_columns=2, max_cell_bytes=1).row_count == 1


def test_rejects_paths_outside_private_root_and_parent_traversal(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n")
    with pytest.raises(CsvValidationError):
        validate_csv(outside, ResourceLimits(), expected_root=root)
    with pytest.raises(CsvValidationError):
        validate_csv(root / ".." / "outside.csv", ResourceLimits())
    with pytest.raises(CsvValidationError):
        validate_csv(root, ResourceLimits())
    with pytest.raises(CsvValidationError):
        validate_csv(tmp_path / "absent.csv", ResourceLimits())


def test_rejects_file_substitution_before_open(tmp_path, monkeypatch):
    import ml.data.csv_io as module

    path = tmp_path / "input.csv"
    path.write_bytes(b'a\n1\n')
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(b'a\n2\n')
    original_open = module.os.open

    def substitute(name, flags, *args, **kwargs):
        replacement.replace(path)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", substitute)
    with pytest.raises(CsvValidationError, match="changed"):
        validate_csv(path, ResourceLimits())


def test_delimiter_flood_is_rejected_before_large_cell_list_allocation(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_bytes(b"," * 1_000_000 + b"\n1\n")
    tracemalloc.start()
    try:
        with pytest.raises(CsvValidationError):
            validate_csv(path, ResourceLimits())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 6_000_000
