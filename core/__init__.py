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
from core.baselines import (
    detect_zscore_anomalies,
    detect_iqr_anomalies,
    select_usable_numerical_features,
    AnomalyDetectionResult,
)
from core.ml_anomaly import detect_isolation_forest_anomalies
from core.cleaner import (
    CleaningError,
    ProposalNotApprovedError,
    InvalidProposalError,
    ProposalAlreadyAppliedError,
    CleaningProposal,
    ProposalPreview,
    AuditRecord,
    QualityComparison,
    generate_cleaning_proposals,
    preview_cleaning_proposal,
    apply_cleaning_proposal,
    compare_datasets,
    DataCleaner,
)

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
    "detect_zscore_anomalies",
    "detect_iqr_anomalies",
    "select_usable_numerical_features",
    "detect_isolation_forest_anomalies",
    "AnomalyDetectionResult",
    "CleaningError",
    "ProposalNotApprovedError",
    "InvalidProposalError",
    "ProposalAlreadyAppliedError",
    "CleaningProposal",
    "ProposalPreview",
    "AuditRecord",
    "QualityComparison",
    "generate_cleaning_proposals",
    "preview_cleaning_proposal",
    "apply_cleaning_proposal",
    "compare_datasets",
    "DataCleaner",
]
