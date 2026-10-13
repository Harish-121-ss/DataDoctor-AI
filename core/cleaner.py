"""
DataDoctor AI - Reviewable Data Cleaning and Repair Engine
Provides safe, auditable data transformations with mandatory explicit approval,
preview diffs, audit trail logging, and honest before/after quality comparisons.
Original datasets are guaranteed to remain completely immutable.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from core.baselines import AnomalyDetectionResult
from core.diagnostics import DiagnosticIssue, diagnose_dataset, generate_quality_report
from core.profiler import DatasetProfile, profile_dataset


class CleaningError(Exception):
    """Base exception for data cleaning operations."""
    pass


class ProposalNotApprovedError(CleaningError):
    """Raised when an operation is executed without explicit user approval."""
    pass


class InvalidProposalError(CleaningError):
    """Raised when a proposal is malformed, unknown, or incompatible with the dataset."""
    pass


class ProposalAlreadyAppliedError(CleaningError):
    """Raised when a proposal has already been applied to prevent double application."""
    pass


@dataclass
class CleaningProposal:
    """
    A documented, reviewable repair suggestion generated from detected issues.
    """
    proposal_id: str
    issue_type: str
    column: Optional[str]
    operation: str
    params: Dict[str, Any]
    rationale: str
    affected_count: int
    affected_percentage: float
    risks: str
    status: str = "pending"  # "pending", "applied", "rejected"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProposalPreview:
    """
    Dry-run preview of a cleaning proposal on an isolated copy of the data.
    """
    proposal_id: str
    operation: str
    column: Optional[str]
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    affected_rows: int
    affected_cells: int
    sample_diff: List[Dict[str, Any]]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditRecord:
    """
    Immutable audit log entry recording every approved and applied transformation.
    Sensitive cell contents are excluded by default to preserve privacy.
    """
    record_id: str
    proposal_id: str
    operation: str
    column: Optional[str]
    params: Dict[str, Any]
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    affected_rows: int
    affected_cells: int
    status: str  # "SUCCESS" or "FAILED"
    explanation: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityComparison:
    """
    Comprehensive before-and-after quality comparison.
    Does not assume fewer issues equals an objectively superior dataset.
    """
    rows_before: int
    rows_after: int
    rows_diff: int
    cols_before: int
    cols_after: int
    cols_diff: int
    missing_cells_before: int
    missing_cells_after: int
    missing_cells_diff: int
    duplicate_rows_before: int
    duplicate_rows_after: int
    duplicate_rows_diff: int
    issues_count_before: int
    issues_count_after: int
    issues_count_diff: int
    health_score_before: float
    health_score_after: float
    health_score_diff: float
    applied_operations_count: int
    evaluation_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Proposal Generation Logic
# ---------------------------------------------------------------------------

def check_whitespace_issues(series: pd.Series) -> int:
    """Count non-null text entries containing leading/trailing or redundant internal spaces."""
    if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
        return 0
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return 0
    pattern = re.compile(r"^\s+|\s+$|\s{2,}")
    return int(non_null.apply(lambda x: bool(pattern.search(x))).sum())


def generate_cleaning_proposals(
    df: pd.DataFrame,
    issues: Optional[List[DiagnosticIssue]] = None,
    anomaly_results: Optional[Sequence[AnomalyDetectionResult]] = None,
) -> List[CleaningProposal]:
    """
    Inspect dataset defects and generate concrete, reviewable cleaning proposals.
    Never alters data. All proposals are created in 'pending' status.
    """
    if issues is None:
        issues = diagnose_dataset(df)

    proposals: List[CleaningProposal] = []
    prop_counter = 1
    total_rows = len(df)

    # 1. Generate proposals from diagnostic issues
    for issue in issues:
        col = issue.column
        itype = issue.issue_type

        # A. Duplicate rows
        if itype == "DUPLICATE_ROWS":
            proposals.append(
                CleaningProposal(
                    proposal_id=f"PROP-DEDUP-{prop_counter:03d}",
                    issue_type=itype,
                    column=None,
                    operation="deduplicate_rows",
                    params={"keep": "first"},
                    rationale="Remove exact duplicate rows to prevent artificially inflated statistical weights.",
                    affected_count=issue.affected_count,
                    affected_percentage=issue.affected_percentage,
                    risks="Dropping rows eliminates identical records which might represent legitimate repeated events.",
                    details={"keep": "first"},
                )
            )
            prop_counter += 1

        # B. All missing column
        elif itype == "ALL_MISSING_COLUMN" and col in df.columns:
            proposals.append(
                CleaningProposal(
                    proposal_id=f"PROP-DROPCOL-{prop_counter:03d}",
                    issue_type=itype,
                    column=col,
                    operation="drop_column",
                    params={"column": col},
                    rationale=f"Column '{col}' is 100% missing and contains no analytical information.",
                    affected_count=issue.affected_count,
                    affected_percentage=100.0,
                    risks="Permanently removes column schema from working copy.",
                )
            )
            prop_counter += 1

        # C. Missing values (Partial)
        elif itype in ("MISSING_VALUES", "HIGH_MISSING_VALUES") and col in df.columns:
            s = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(s)

            if is_numeric:
                # Primary recommendation: median imputation (robust to outliers)
                median_val = float(s.median()) if s.dropna().shape[0] > 0 else 0.0
                proposals.append(
                    CleaningProposal(
                        proposal_id=f"PROP-IMPUTE-MEDIAN-{prop_counter:03d}",
                        issue_type=itype,
                        column=col,
                        operation="impute_median",
                        params={"column": col, "value": median_val},
                        rationale=f"Impute {issue.affected_count} missing numerical entries in '{col}' with column median ({median_val:.2f}).",
                        affected_count=issue.affected_count,
                        affected_percentage=issue.affected_percentage,
                        risks="Imputation preserves row count but compresses variance and can underestimate standard errors.",
                        details={"imputed_value": median_val},
                    )
                )
                prop_counter += 1
            else:
                # Categorical / text imputation: mode
                non_null = s.dropna()
                mode_val = str(non_null.mode().iloc[0]) if len(non_null) > 0 else "Unknown"
                proposals.append(
                    CleaningProposal(
                        proposal_id=f"PROP-IMPUTE-MODE-{prop_counter:03d}",
                        issue_type=itype,
                        column=col,
                        operation="impute_mode",
                        params={"column": col, "value": mode_val},
                        rationale=f"Impute {issue.affected_count} missing entries in '{col}' with most frequent value ('{mode_val}').",
                        affected_count=issue.affected_count,
                        affected_percentage=issue.affected_percentage,
                        risks="Mode imputation artificially inflates frequency of the dominant class.",
                        details={"imputed_value": mode_val},
                    )
                )
                prop_counter += 1

            # Alternative recommendation: drop affected rows
            proposals.append(
                CleaningProposal(
                    proposal_id=f"PROP-DROPROWS-NULL-{prop_counter:03d}",
                    issue_type=itype,
                    column=col,
                    operation="drop_missing_rows",
                    params={"column": col},
                    rationale=f"Drop {issue.affected_count} rows containing missing entries in '{col}' for strict complete-case analysis.",
                    affected_count=issue.affected_count,
                    affected_percentage=issue.affected_percentage,
                    risks="Dropping rows discards valid observations in other columns and may introduce selection bias.",
                )
            )
            prop_counter += 1

        # D. Type inconsistency (text containing predominantly numeric data)
        elif itype == "TYPE_INCONSISTENCY" and col in df.columns:
            proposals.append(
                CleaningProposal(
                    proposal_id=f"PROP-CAST-NUM-{prop_counter:03d}",
                    issue_type=itype,
                    column=col,
                    operation="convert_numeric",
                    params={"column": col, "strip_characters": "$€£¥, "},
                    rationale=f"Sanitize stray currency and punctuation tokens and cast '{col}' to numeric float/int.",
                    affected_count=issue.affected_count,
                    affected_percentage=issue.affected_percentage,
                    risks="Non-parseable text placeholders will be converted to NaN for downstream handling.",
                )
            )
            prop_counter += 1

        # E. Constant column (zero variance)
        elif itype == "CONSTANT_COLUMN" and col in df.columns:
            proposals.append(
                CleaningProposal(
                    proposal_id=f"PROP-DROP-CONST-{prop_counter:03d}",
                    issue_type=itype,
                    column=col,
                    operation="drop_column",
                    params={"column": col},
                    rationale=f"Column '{col}' has zero variance (1 constant value across all rows) providing no predictive information.",
                    affected_count=issue.affected_count,
                    affected_percentage=100.0,
                    risks="Removes column completely from the dataset.",
                )
            )
            prop_counter += 1

        # F. Suspicious negative values in naturally positive domains
        elif itype == "SUSPICIOUS_NEGATIVE_VALUE" and col in df.columns:
            proposals.append(
                CleaningProposal(
                    proposal_id=f"PROP-CLIP-ZERO-{prop_counter:03d}",
                    issue_type=itype,
                    column=col,
                    operation="clip_bounds",
                    params={"column": col, "lower": 0.0, "upper": None},
                    rationale=f"Clip negative entries in '{col}' to 0.0 (review domain bounds).",
                    affected_count=issue.affected_count,
                    affected_percentage=issue.affected_percentage,
                    risks="Modifies raw numerical measurements; negative values may reflect data recording conventions or valid credits.",
                )
            )
            prop_counter += 1

        # G. Suspicious infinite values
        elif itype == "SUSPICIOUS_INFINITE_VALUE" and col in df.columns:
            proposals.append(
                CleaningProposal(
                    proposal_id=f"PROP-REP-INF-{prop_counter:03d}",
                    issue_type=itype,
                    column=col,
                    operation="replace_with_nan",
                    params={"column": col, "replace_targets": ["inf", "-inf"]},
                    rationale=f"Replace {issue.affected_count} infinite values in '{col}' with NaN so they can be handled safely.",
                    affected_count=issue.affected_count,
                    affected_percentage=issue.affected_percentage,
                    risks="Introduces missing values requiring subsequent imputation or filtering.",
                )
            )
            prop_counter += 1

    # 2. Check for inconsistent text formatting / whitespace issues across string columns
    for col in df.columns:
        ws_count = check_whitespace_issues(df[col])
        if ws_count > 0:
            ws_pct = round(ws_count / max(total_rows, 1) * 100.0, 2)
            proposals.append(
                CleaningProposal(
                    proposal_id=f"PROP-NORM-TEXT-{prop_counter:03d}",
                    issue_type="INCONSISTENT_TEXT",
                    column=str(col),
                    operation="normalize_text",
                    params={"column": str(col), "strip": True, "collapse_spaces": True},
                    rationale=f"Normalize whitespace in {ws_count} text cells in '{col}' (strip leading/trailing and collapse internal spaces).",
                    affected_count=ws_count,
                    affected_percentage=ws_pct,
                    risks="Whitespace changes may alter intentional formatting (e.g. multi-line text or code strings).",
                )
            )
            prop_counter += 1

    # 3. Incorporate Anomaly Detection results from Stage 2 (if provided)
    if anomaly_results:
        for anom_res in anomaly_results:
            if anom_res.anomaly_count > 0:
                flag_col_name = f"is_anomaly_{anom_res.method}"
                # Non-destructive option: flag records with indicator column
                proposals.append(
                    CleaningProposal(
                        proposal_id=f"PROP-ANOM-FLAG-{prop_counter:03d}",
                        issue_type="ANOMALY_REVIEW",
                        column=None,
                        operation="flag_anomalies",
                        params={"indicator_column": flag_col_name, "indices": anom_res.anomaly_indices},
                        rationale=(
                            f"Add boolean indicator column '{flag_col_name}' tagging {anom_res.anomaly_count} "
                            f"records flagged as statistically unusual by {anom_res.method}."
                        ),
                        affected_count=anom_res.anomaly_count,
                        affected_percentage=anom_res.anomaly_percentage,
                        risks="Adds a new metadata column to the working dataset without altering original features.",
                        details={"method": anom_res.method, "anomaly_indices": anom_res.anomaly_indices[:10]},
                    )
                )
                prop_counter += 1

                # Destructive option: filter/remove anomalous rows
                proposals.append(
                    CleaningProposal(
                        proposal_id=f"PROP-ANOM-FILTER-{prop_counter:03d}",
                        issue_type="ANOMALY_REVIEW",
                        column=None,
                        operation="filter_anomalies",
                        params={"indices": anom_res.anomaly_indices},
                        rationale=(
                            f"Remove {anom_res.anomaly_count} anomalous rows flagged by {anom_res.method} "
                            f"for models highly sensitive to outliers."
                        ),
                        affected_count=anom_res.anomaly_count,
                        affected_percentage=anom_res.anomaly_percentage,
                        risks=(
                            "CRITICAL: Statistical anomalies are not automatically erroneous or fraudulent! "
                            "Removing them discards rare but valid events and distorts real-world population distributions."
                        ),
                        details={"method": anom_res.method},
                    )
                )
                prop_counter += 1

    return proposals


# ---------------------------------------------------------------------------
# Transformation Execution Helpers
# ---------------------------------------------------------------------------

def execute_operation(
    df: pd.DataFrame,
    proposal: CleaningProposal,
) -> Tuple[pd.DataFrame, int, int, List[Dict[str, Any]], str]:
    """
    Execute a specific approved cleaning proposal on a copy of the DataFrame.
    Returns: (transformed_df, affected_rows, affected_cells, sample_diff, explanation)
    """
    op = proposal.operation
    params = proposal.params
    col = proposal.column
    target_df = df.copy(deep=True)

    sample_diff: List[Dict[str, Any]] = []

    # 1. Deduplicate rows
    if op == "deduplicate_rows":
        keep = params.get("keep", "first")
        dup_mask = target_df.duplicated(keep=keep)
        affected_rows = int(dup_mask.sum())
        affected_cells = affected_rows * len(target_df.columns)
        dup_indices = target_df.index[dup_mask].tolist()[:5]

        for idx in dup_indices:
            sample_diff.append({
                "row_index": idx,
                "column": "<all_columns>",
                "before": "Duplicate row",
                "after": "<removed>",
            })

        transformed_df = target_df.drop_duplicates(keep=keep).reset_index(drop=True)
        explanation = f"Removed {affected_rows} duplicate rows (keeping '{keep}' instance)."
        return transformed_df, affected_rows, affected_cells, sample_diff, explanation

    # 2. Drop column
    elif op == "drop_column":
        if col not in target_df.columns:
            raise InvalidProposalError(f"Column '{col}' does not exist in DataFrame.")
        affected_rows = len(target_df)
        affected_cells = affected_rows
        sample_diff.append({
            "row_index": "<all_rows>",
            "column": col,
            "before": f"Column present with {target_df[col].dropna().count()} non-null entries",
            "after": "<column_dropped>",
        })
        transformed_df = target_df.drop(columns=[col])
        explanation = f"Dropped column '{col}' ({affected_rows} cells removed)."
        return transformed_df, affected_rows, affected_cells, sample_diff, explanation

    # 3. Drop missing rows
    elif op == "drop_missing_rows":
        if col not in target_df.columns:
            raise InvalidProposalError(f"Column '{col}' does not exist in DataFrame.")
        null_mask = target_df[col].isna()
        affected_rows = int(null_mask.sum())
        affected_cells = affected_rows
        null_indices = target_df.index[null_mask].tolist()[:5]

        for idx in null_indices:
            sample_diff.append({
                "row_index": idx,
                "column": col,
                "before": None,
                "after": "<row_dropped>",
            })

        transformed_df = target_df[~null_mask].reset_index(drop=True)
        explanation = f"Dropped {affected_rows} rows containing missing entries in '{col}'."
        return transformed_df, affected_rows, affected_cells, sample_diff, explanation

    # 4. Impute median / mean / mode / constant
    elif op in ("impute_median", "impute_mean", "impute_mode", "impute_constant"):
        if col not in target_df.columns:
            raise InvalidProposalError(f"Column '{col}' does not exist in DataFrame.")

        fill_val = params.get("value")
        if fill_val is None:
            if op == "impute_median":
                fill_val = float(target_df[col].median())
            elif op == "impute_mean":
                fill_val = float(target_df[col].mean())
            elif op == "impute_mode":
                mode_s = target_df[col].mode()
                fill_val = mode_s.iloc[0] if len(mode_s) > 0 else "Unknown"
            else:
                fill_val = params.get("fill_value", 0)

        null_mask = target_df[col].isna()
        affected_cells = int(null_mask.sum())
        affected_rows = affected_cells
        null_indices = target_df.index[null_mask].tolist()[:5]

        for idx in null_indices:
            sample_diff.append({
                "row_index": idx,
                "column": col,
                "before": None,
                "after": fill_val,
            })

        target_df[col] = target_df[col].fillna(fill_val)
        explanation = f"Imputed {affected_cells} missing entries in '{col}' with {op.replace('impute_', '')} value: {fill_val}."
        return target_df, affected_rows, affected_cells, sample_diff, explanation

    # 5. Convert numeric
    elif op == "convert_numeric":
        if col not in target_df.columns:
            raise InvalidProposalError(f"Column '{col}' does not exist in DataFrame.")

        series = target_df[col]
        # Clean string tokens
        strip_chars = params.get("strip_characters", "$€£¥, ")
        cleaned_series = series.astype(str)
        for char in strip_chars:
            cleaned_series = cleaned_series.str.replace(char, "", regex=False)

        coerced = pd.to_numeric(cleaned_series.str.strip(), errors="coerce")
        # Find cells that actually changed
        changed_mask = series.isna() != coerced.isna()
        # Also check values that had tokens removed
        changed_mask |= (series.astype(str) != coerced.astype(str)) & series.notna()

        affected_cells = int(changed_mask.sum())
        affected_rows = affected_cells

        diff_indices = target_df.index[changed_mask].tolist()[:5]
        for idx in diff_indices:
            sample_diff.append({
                "row_index": idx,
                "column": col,
                "before": series.iloc[idx],
                "after": coerced.iloc[idx] if not pd.isna(coerced.iloc[idx]) else None,
            })

        target_df[col] = coerced
        explanation = f"Sanitized and converted '{col}' to numeric ({affected_cells} cells modified)."
        return target_df, affected_rows, affected_cells, sample_diff, explanation

    # 6. Normalize text (whitespace stripping & space collapsing)
    elif op == "normalize_text":
        if col not in target_df.columns:
            raise InvalidProposalError(f"Column '{col}' does not exist in DataFrame.")

        series = target_df[col].dropna().astype(str)
        collapsed = series.str.strip().str.replace(r"\s+", " ", regex=True)

        changed_mask = series != collapsed
        affected_cells = int(changed_mask.sum())
        affected_rows = affected_cells

        diff_indices = series.index[changed_mask].tolist()[:5]
        for idx in diff_indices:
            sample_diff.append({
                "row_index": idx,
                "column": col,
                "before": target_df.loc[idx, col],
                "after": collapsed.loc[idx],
            })

        target_df.loc[series.index, col] = collapsed
        explanation = f"Normalized whitespace across {affected_cells} cells in '{col}'."
        return target_df, affected_rows, affected_cells, sample_diff, explanation

    # 7. Clip bounds (e.g. non-negative clipping)
    elif op == "clip_bounds":
        if col not in target_df.columns:
            raise InvalidProposalError(f"Column '{col}' does not exist in DataFrame.")

        lower = params.get("lower")
        upper = params.get("upper")
        series = pd.to_numeric(target_df[col], errors="coerce")

        clipped = series.clip(lower=lower, upper=upper)
        changed_mask = (series != clipped) & series.notna()
        affected_cells = int(changed_mask.sum())
        affected_rows = affected_cells

        diff_indices = target_df.index[changed_mask].tolist()[:5]
        for idx in diff_indices:
            sample_diff.append({
                "row_index": idx,
                "column": col,
                "before": target_df.loc[idx, col],
                "after": clipped.iloc[idx],
            })

        target_df[col] = clipped
        explanation = f"Clipped {affected_cells} entries in '{col}' to bounds [lower={lower}, upper={upper}]."
        return target_df, affected_rows, affected_cells, sample_diff, explanation

    # 8. Replace with NaN
    elif op == "replace_with_nan":
        if col not in target_df.columns:
            raise InvalidProposalError(f"Column '{col}' does not exist in DataFrame.")

        targets = params.get("replace_targets", ["inf", "-inf"])
        series = target_df[col]
        num_s = pd.to_numeric(series, errors="coerce")

        if "inf" in targets or "-inf" in targets:
            inf_mask = np.isinf(num_s)
            affected_cells = int(inf_mask.sum())
            affected_rows = affected_cells
            diff_indices = target_df.index[inf_mask].tolist()[:5]
            for idx in diff_indices:
                sample_diff.append({
                    "row_index": idx,
                    "column": col,
                    "before": str(target_df.loc[idx, col]),
                    "after": None,
                })
            target_df.loc[inf_mask, col] = np.nan
        else:
            affected_cells = 0
            affected_rows = 0

        explanation = f"Replaced {affected_cells} invalid/infinite targets with NaN in '{col}'."
        return target_df, affected_rows, affected_cells, sample_diff, explanation

    # 9. Flag anomalies
    elif op == "flag_anomalies":
        indicator_col = params.get("indicator_column", "is_anomaly")
        indices = params.get("indices", [])
        valid_indices = [i for i in indices if i in target_df.index]
        affected_rows = len(valid_indices)
        affected_cells = len(target_df)

        target_df[indicator_col] = False
        target_df.loc[valid_indices, indicator_col] = True

        for idx in valid_indices[:5]:
            sample_diff.append({
                "row_index": idx,
                "column": indicator_col,
                "before": "<column_added>",
                "after": True,
            })

        explanation = f"Added indicator column '{indicator_col}' tagging {affected_rows} anomalous records."
        return target_df, affected_rows, affected_cells, sample_diff, explanation

    # 10. Filter anomalies
    elif op == "filter_anomalies":
        indices = params.get("indices", [])
        valid_indices = [i for i in indices if i in target_df.index]
        affected_rows = len(valid_indices)
        affected_cells = affected_rows * len(target_df.columns)

        for idx in valid_indices[:5]:
            sample_diff.append({
                "row_index": idx,
                "column": "<all_columns>",
                "before": "Anomalous row",
                "after": "<row_filtered>",
            })

        transformed_df = target_df.drop(index=valid_indices).reset_index(drop=True)
        explanation = f"Filtered out {affected_rows} records flagged as anomalies."
        return transformed_df, affected_rows, affected_cells, sample_diff, explanation

    else:
        raise InvalidProposalError(f"Unsupported or unknown cleaning operation: '{op}'")


# ---------------------------------------------------------------------------
# Public Functional API
# ---------------------------------------------------------------------------

def preview_cleaning_proposal(
    df: pd.DataFrame,
    proposal: CleaningProposal,
) -> ProposalPreview:
    """
    Generate a dry-run preview of the proposal on an isolated copy of df.
    Guarantees zero mutation of input DataFrame.
    """
    rows_before = len(df)
    cols_before = len(df.columns)

    transformed_df, affected_rows, affected_cells, sample_diff, explanation = execute_operation(
        df, proposal
    )

    rows_after = len(transformed_df)
    cols_after = len(transformed_df.columns)

    return ProposalPreview(
        proposal_id=proposal.proposal_id,
        operation=proposal.operation,
        column=proposal.column,
        rows_before=rows_before,
        rows_after=rows_after,
        cols_before=cols_before,
        cols_after=cols_after,
        affected_rows=affected_rows,
        affected_cells=affected_cells,
        sample_diff=sample_diff,
        explanation=explanation,
    )


def apply_cleaning_proposal(
    df: pd.DataFrame,
    proposal: CleaningProposal,
    approved: bool = False,
) -> Tuple[pd.DataFrame, AuditRecord]:
    """
    Apply an approved cleaning proposal to df.
    Requires explicit approval (`approved=True`), returning a new cleaned DataFrame
    and an AuditRecord. Does not mutate the input DataFrame.
    """
    if not approved:
        raise ProposalNotApprovedError(
            f"Cannot execute proposal '{proposal.proposal_id}' without explicit approval. "
            "Please review the preview and set `approved=True`."
        )

    rows_before = len(df)
    cols_before = len(df.columns)

    transformed_df, affected_rows, affected_cells, _, explanation = execute_operation(
        df, proposal
    )

    rows_after = len(transformed_df)
    cols_after = len(transformed_df.columns)

    audit_entry = AuditRecord(
        record_id=f"AUDIT-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        proposal_id=proposal.proposal_id,
        operation=proposal.operation,
        column=proposal.column,
        params=proposal.params,
        rows_before=rows_before,
        rows_after=rows_after,
        cols_before=cols_before,
        cols_after=cols_after,
        affected_rows=affected_rows,
        affected_cells=affected_cells,
        status="SUCCESS",
        explanation=explanation,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    proposal.status = "applied"
    return transformed_df, audit_entry


def compare_datasets(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    applied_count: int = 0,
) -> QualityComparison:
    """
    Produce a rigorous before-and-after quality profile comparison.
    Articulates trade-offs honestly without equating fewer issues to automatic correctness.
    """
    report_before = generate_quality_report(original_df)
    report_after = generate_quality_report(cleaned_df)

    p_before = report_before["profile"]
    p_after = report_after["profile"]

    rows_b = p_before["total_rows"]
    rows_a = p_after["total_rows"]
    cols_b = p_before["total_columns"]
    cols_a = p_after["total_columns"]

    missing_b = p_before["total_missing_cells"]
    missing_a = p_after["total_missing_cells"]

    dup_b = p_before["duplicate_rows"]
    dup_a = p_after["duplicate_rows"]

    issues_b = report_before["total_issues"]
    issues_a = report_after["total_issues"]

    score_b = report_before["health_score"]
    score_a = report_after["health_score"]

    notes: List[str] = []

    if rows_a < rows_b:
        dropped = rows_b - rows_a
        notes.append(
            f"Dataset size decreased by {dropped} rows ({dropped/max(rows_b, 1)*100:.1f}%). "
            "Caution: Dropping observations reduces sample power and may discard legitimate rare instances."
        )

    if missing_a < missing_b:
        resolved_missing = missing_b - missing_a
        notes.append(
            f"Resolved {resolved_missing} missing values. "
            "Note: Where imputation was used, sample variance may be reduced and relationships between features altered."
        )

    if dup_a < dup_b:
        resolved_dup = dup_b - dup_a
        notes.append(f"Removed {resolved_dup} duplicate rows.")

    notes.append(
        "Important: A higher dataset health score reflects conformance to standard tabular rules, "
        "not guaranteed domain truth or statistical validity for all downstream ML models."
    )

    return QualityComparison(
        rows_before=rows_b,
        rows_after=rows_a,
        rows_diff=rows_a - rows_b,
        cols_before=cols_b,
        cols_after=cols_a,
        cols_diff=cols_a - cols_b,
        missing_cells_before=missing_b,
        missing_cells_after=missing_a,
        missing_cells_diff=missing_a - missing_b,
        duplicate_rows_before=dup_b,
        duplicate_rows_after=dup_a,
        duplicate_rows_diff=dup_a - dup_b,
        issues_count_before=issues_b,
        issues_count_after=issues_a,
        issues_count_diff=issues_a - issues_b,
        health_score_before=score_b,
        health_score_after=score_a,
        health_score_diff=round(score_a - score_b, 1),
        applied_operations_count=applied_count,
        evaluation_notes=notes,
    )


# ---------------------------------------------------------------------------
# Stateful Cleaner Session Wrapper
# ---------------------------------------------------------------------------

class DataCleaner:
    """
    Session manager providing an interactive, review-driven workflow:
    propose -> preview -> approve -> audit -> compare.
    """

    def __init__(self, df: pd.DataFrame):
        # Guaranteed deep copy to preserve original dataset
        self.original_df = df.copy(deep=True)
        self.current_df = df.copy(deep=True)
        self.proposals: Dict[str, CleaningProposal] = {}
        self.audit_log: List[AuditRecord] = []
        self._applied_ids: set[str] = set()

    def generate_proposals(
        self,
        issues: Optional[List[DiagnosticIssue]] = None,
        anomaly_results: Optional[Sequence[AnomalyDetectionResult]] = None,
    ) -> List[CleaningProposal]:
        """Generate and index proposals for current dataset state."""
        prop_list = generate_cleaning_proposals(
            self.current_df, issues=issues, anomaly_results=anomaly_results
        )
        self.proposals = {p.proposal_id: p for p in prop_list}
        return prop_list

    def preview_proposal(self, proposal_or_id: Union[str, CleaningProposal]) -> ProposalPreview:
        """Preview proposal changes on current dataset without applying."""
        prop = self._resolve_proposal(proposal_or_id)
        return preview_cleaning_proposal(self.current_df, prop)

    def apply_proposal(
        self,
        proposal_or_id: Union[str, CleaningProposal],
        approved: bool = False,
    ) -> Tuple[pd.DataFrame, AuditRecord]:
        """Apply an approved proposal with double-application safeguards."""
        prop = self._resolve_proposal(proposal_or_id)

        if prop.proposal_id in self._applied_ids:
            raise ProposalAlreadyAppliedError(
                f"Proposal '{prop.proposal_id}' has already been applied in this session."
            )

        updated_df, audit_record = apply_cleaning_proposal(
            self.current_df, prop, approved=approved
        )

        self.current_df = updated_df
        self._applied_ids.add(prop.proposal_id)
        self.audit_log.append(audit_record)

        return self.current_df, audit_record

    def apply_approved(self, proposal_ids: Sequence[str]) -> pd.DataFrame:
        """Batch apply a sequence of explicitly approved proposal IDs."""
        for pid in proposal_ids:
            self.apply_proposal(pid, approved=True)
        return self.current_df

    def compare_quality(self) -> QualityComparison:
        """Compute before-and-after quality comparison."""
        return compare_datasets(
            self.original_df, self.current_df, applied_count=len(self.audit_log)
        )

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return full audit log as a list of dictionaries."""
        return [r.to_dict() for r in self.audit_log]

    def _resolve_proposal(self, proposal_or_id: Union[str, CleaningProposal]) -> CleaningProposal:
        if isinstance(proposal_or_id, CleaningProposal):
            return proposal_or_id
        if proposal_or_id in self.proposals:
            return self.proposals[proposal_or_id]
        raise InvalidProposalError(f"Proposal ID '{proposal_or_id}' was not found in active proposals.")
