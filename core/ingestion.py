"""
DataDoctor AI - Ingestion Engine
Handles resilient, non-destructive CSV loading with delimiter detection,
encoding handling, size limits, and robust error classification.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import pandas as pd

# Default maximum file size: 50 Megabytes
DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
SAMPLE_SNIFF_BYTES = 64 * 1024  # 64 KB


class DataDoctorIngestionError(Exception):
    """Base exception for all ingestion failures."""
    pass


class FileTooLargeError(DataDoctorIngestionError):
    """Raised when the input file exceeds the configured maximum allowed size."""
    pass


class EmptyFileError(DataDoctorIngestionError):
    """Raised when the input file has 0 bytes or contains no rows."""
    pass


class MalformedFileError(DataDoctorIngestionError):
    """Raised when the input file cannot be parsed as a valid tabular CSV."""
    pass


def detect_encoding(raw_bytes: bytes) -> str:
    """
    Detect text encoding from raw bytes, properly checking for UTF-8 with BOM,
    standard UTF-8, and falling back to Latin-1.
    """
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    try:
        raw_bytes.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        raw_bytes.decode("latin-1")
        return "latin-1"
    except UnicodeDecodeError:
        pass

    return "utf-8"


def detect_delimiter(sample_text: str) -> str:
    """
    Infer the column delimiter using csv.Sniffer with candidate delimiter fallbacks.
    Default fallback is comma (',').
    """
    if not sample_text or not sample_text.strip():
        return ","

    # Try csv.Sniffer first
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=CANDIDATE_DELIMITERS)
        if dialect.delimiter in CANDIDATE_DELIMITERS:
            return dialect.delimiter
    except Exception:
        pass

    # Fallback heuristic: count delimiter frequency in non-empty lines
    lines = [line.strip() for line in sample_text.splitlines() if line.strip()][:10]
    if not lines:
        return ","

    best_delimiter = ","
    max_score = -1

    for delim in CANDIDATE_DELIMITERS:
        counts = [line.count(delim) for line in lines]
        # Must appear at least once per line to be a reliable delimiter
        if all(c > 0 for c in counts) and len(set(counts)) == 1:
            # Consistent delimiter count across all sample lines
            score = counts[0] * 100
        else:
            score = sum(counts)

        if score > max_score:
            max_score = score
            best_delimiter = delim

    return best_delimiter if max_score > 0 else ","


def load_csv(
    source: Union[str, Path, io.BytesIO, io.StringIO, BinaryIO, TextIO],
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    delimiter: str | None = None,
    encoding: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a CSV file into a pandas DataFrame safely and immutably.

    Parameters
    ----------
    source : str, Path, or file-like buffer
        Path to the CSV file or an in-memory stream.
    max_file_size_bytes : int, default 50MB
        Maximum permitted file size in bytes to prevent memory exhaustion.
    delimiter : str, optional
        Explicit delimiter to override automatic detection.
    encoding : str, optional
        Explicit character encoding to override automatic detection.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A tuple of (DataFrame, metadata_dict) where metadata describes the ingestion.

    Raises
    ------
    FileNotFoundError
        If a file path was given but does not exist.
    FileTooLargeError
        If file size exceeds max_file_size_bytes.
    EmptyFileError
        If file is 0 bytes or has no headers/rows.
    MalformedFileError
        If file parsing fails or content is unparseable.
    """
    file_path_str: str | None = None
    raw_bytes: bytes

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found at path: {path}")
        if not path.is_file():
            raise MalformedFileError(f"Provided path is not a regular file: {path}")

        file_size = path.stat().st_size
        if file_size > max_file_size_bytes:
            raise FileTooLargeError(
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit "
                f"of {max_file_size_bytes / (1024 * 1024):.2f} MB."
            )
        if file_size == 0:
            raise EmptyFileError(f"File is completely empty (0 bytes): {path.name}")

        file_path_str = str(path.resolve())

        # Read sample bytes for encoding & delimiter sniff
        with open(path, "rb") as f:
            raw_sample = f.read(SAMPLE_SNIFF_BYTES)
            f.seek(0)
            raw_bytes = f.read()

    elif hasattr(source, "read"):
        # File-like object (BytesIO, StringIO, or Streamlit UploadedFile)
        content = source.read()

        # Handle buffer position reset if possible
        if hasattr(source, "seek"):
            source.seek(0)

        if isinstance(content, str):
            raw_bytes = content.encode("utf-8")
        elif isinstance(content, bytes):
            raw_bytes = content
        else:
            raise MalformedFileError("Unsupported buffer content type; expected str or bytes.")

        file_size = len(raw_bytes)
        if file_size > max_file_size_bytes:
            raise FileTooLargeError(
                f"Buffer size ({file_size / (1024 * 1024):.2f} MB) exceeds limit of "
                f"{max_file_size_bytes / (1024 * 1024):.2f} MB."
            )
        if file_size == 0:
            raise EmptyFileError("Buffer stream is empty (0 bytes).")

        raw_sample = raw_bytes[:SAMPLE_SNIFF_BYTES]
        file_path_str = getattr(source, "name", "<in-memory-stream>")
    else:
        raise MalformedFileError(
            f"Unsupported input type '{type(source).__name__}'. Expected str, Path, or file-like buffer."
        )

    # Check for binary files containing null bytes
    if b"\x00" in raw_sample:
        raise MalformedFileError("Binary or null bytes detected; file is not a valid text CSV.")

    # 1. Determine encoding
    resolved_encoding = encoding or detect_encoding(raw_sample)

    # 2. Decode sample to detect delimiter
    try:
        sample_text = raw_sample.decode(resolved_encoding, errors="replace")
    except Exception as e:
        raise MalformedFileError(f"Failed to decode file with encoding '{resolved_encoding}': {e}") from e

    # Check for empty / whitespace-only content
    if not sample_text.strip():
        raise EmptyFileError("The file contains only whitespace and no valid tabular data.")

    # 3. Determine delimiter
    resolved_delimiter = delimiter or detect_delimiter(sample_text)

    # 4. Parse with pandas
    try:
        buffer_stream = io.BytesIO(raw_bytes)
        df = pd.read_csv(
            buffer_stream,
            sep=resolved_delimiter,
            encoding=resolved_encoding,
            engine="python",  # Python engine handles varied separators and ragged delimiters gracefully
            on_bad_lines="error",
        )
    except pd.errors.EmptyDataError as e:
        raise EmptyFileError("No columns or rows could be parsed from the file.") from e
    except (pd.errors.ParserError, ValueError) as e:
        raise MalformedFileError(f"Unable to parse CSV structure: {e}") from e
    except Exception as e:
        raise MalformedFileError(f"Unexpected error parsing file: {e}") from e

    # Verify parsed dataframe has structure
    if df.empty and len(df.columns) == 0:
        raise EmptyFileError("Parsed dataset contains zero rows and zero columns.")

    metadata = {
        "source": file_path_str,
        "file_size_bytes": file_size,
        "encoding": resolved_encoding,
        "delimiter": resolved_delimiter,
        "row_count": int(len(df)),
        "col_count": int(len(df.columns)),
    }

    return df, metadata
