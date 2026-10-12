"""
DataDoctor AI - Core Engine
Stage 1: Ingestion, Profiling, and Diagnostics.
"""

from core.ingestion import (
    load_csv,
    DataDoctorIngestionError,
    FileTooLargeError,
    EmptyFileError,
    MalformedFileError,
    DEFAULT_MAX_FILE_SIZE_BYTES,
)
from core.profiler import profile_dataset, DatasetProfile, ColumnProfile
from core.diagnostics import diagnose_dataset, generate_quality_report, DiagnosticIssue

__all__ = [
    "load_csv",
    "DataDoctorIngestionError",
    "FileTooLargeError",
    "EmptyFileError",
    "MalformedFileError",
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "profile_dataset",
    "DatasetProfile",
    "ColumnProfile",
    "diagnose_dataset",
    "generate_quality_report",
    "DiagnosticIssue",
]
