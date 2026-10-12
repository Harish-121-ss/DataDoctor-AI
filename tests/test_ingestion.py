"""
Unit tests for core.ingestion module.
Verifies clean CSVs, encodings, delimiters, size limits, error handling, and immutability.
"""

import hashlib
import io
from pathlib import Path

import pandas as pd
import pytest

from core.ingestion import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    EmptyFileError,
    FileTooLargeError,
    MalformedFileError,
    detect_delimiter,
    detect_encoding,
    load_csv,
)


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


class TestCSVIngestion:
    """Test suite for CSV ingestion engine."""

    def test_clean_csv_loading(self, tmp_path: Path):
        csv_file = tmp_path / "clean.csv"
        csv_file.write_text("id,name,score\n1,Alice,95.5\n2,Bob,88.0\n3,Charlie,72.3\n", encoding="utf-8")

        df, meta = load_csv(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["id", "name", "score"]
        assert meta["row_count"] == 3
        assert meta["col_count"] == 3
        assert meta["delimiter"] == ","
        assert meta["encoding"] in ("utf-8", "utf-8-sig")

    def test_utf8_with_bom_loading(self, tmp_path: Path):
        csv_file = tmp_path / "bom.csv"
        # Prepend UTF-8 BOM bytes: \xef\xbb\xbf
        bom_bytes = b"\xef\xbb\xbfid,city,temp\n101,New York,22.5\n102,Tokyo,19.0\n"
        csv_file.write_bytes(bom_bytes)

        df, meta = load_csv(csv_file)
        assert len(df) == 2
        # Ensure BOM was not prepended to the first column name
        assert df.columns[0] == "id"
        assert meta["encoding"] == "utf-8-sig"

    @pytest.mark.parametrize("delim,name", [
        (";", "semicolon.csv"),
        ("\t", "tab.tsv"),
        ("|", "pipe.csv"),
    ])
    def test_alternate_delimiters(self, tmp_path: Path, delim: str, name: str):
        csv_file = tmp_path / name
        content = f"col_a{delim}col_b{delim}col_c\n10{delim}20{delim}30\n40{delim}50{delim}60\n"
        csv_file.write_text(content, encoding="utf-8")

        df, meta = load_csv(csv_file)
        assert len(df) == 2
        assert list(df.columns) == ["col_a", "col_b", "col_c"]
        assert meta["delimiter"] == delim

    def test_empty_file_zero_bytes(self, tmp_path: Path):
        empty_file = tmp_path / "empty.csv"
        empty_file.write_bytes(b"")

        with pytest.raises(EmptyFileError, match="empty"):
            load_csv(empty_file)

    def test_empty_file_whitespace_only(self, tmp_path: Path):
        whitespace_file = tmp_path / "whitespace.csv"
        whitespace_file.write_text("   \n\n\t  \n", encoding="utf-8")

        with pytest.raises(EmptyFileError):
            load_csv(whitespace_file)

    def test_file_not_found(self, tmp_path: Path):
        missing = tmp_path / "non_existent.csv"
        with pytest.raises(FileNotFoundError):
            load_csv(missing)

    def test_file_size_limit_exceeded(self, tmp_path: Path):
        large_file = tmp_path / "oversize.csv"
        # Write 200 bytes and set limit to 100 bytes
        large_file.write_bytes(b"x" * 200)

        with pytest.raises(FileTooLargeError, match="exceeds maximum allowed limit"):
            load_csv(large_file, max_file_size_bytes=100)

    def test_malformed_csv_input(self, tmp_path: Path):
        # Binary non-text file
        binary_file = tmp_path / "corrupt.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\x04\xff\xfe\x00\x00\x12\x34\x56")

        # Ingestion should raise MalformedFileError or fail gracefully
        with pytest.raises((MalformedFileError, EmptyFileError)):
            load_csv(binary_file)

    def test_source_file_immutability(self, tmp_path: Path):
        csv_file = tmp_path / "immutable.csv"
        original_content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
        csv_file.write_text(original_content, encoding="utf-8")

        hash_before = compute_file_hash(csv_file)
        df, _ = load_csv(csv_file)

        # Mutate the returned DataFrame
        df["new_col"] = [10, 20, 30]
        df.drop(columns=["a"], inplace=True)

        hash_after = compute_file_hash(csv_file)
        assert hash_before == hash_after, "The source file was modified on disk!"

    def test_in_memory_buffer_loading(self):
        csv_text = "item,qty,price\nWidget,10,4.99\nGadget,5,19.99\n"
        buf = io.StringIO(csv_text)

        df, meta = load_csv(buf)
        assert len(df) == 2
        assert list(df.columns) == ["item", "qty", "price"]
        assert meta["delimiter"] == ","
