"""
DataDoctor AI - Data Quality & Remediation Report Generator
Produces structured Markdown quality reports suitable for auditing, compliance,
and downstream stakeholders.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.cleaner import AuditRecord, QualityComparison
from core.diagnostics import DiagnosticIssue


def generate_markdown_audit_report(
    metadata: Dict[str, Any],
    original_report: Dict[str, Any],
    cleaned_df: Optional[pd.DataFrame] = None,
    comparison: Optional[QualityComparison] = None,
    audit_trail: Optional[List[AuditRecord]] = None,
    anomaly_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format a complete dataset quality and remediation audit report as Markdown.
    """
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    profile = original_report.get("profile", {})
    issues = original_report.get("issues", [])
    health_score = original_report.get("health_score", 0.0)

    lines = [
        "# DataDoctor AI — Dataset Quality & Remediation Audit Report",
        "",
        f"**Generated:** {timestamp_str}  ",
        f"**Source File:** `{metadata.get('source', 'Uploaded Dataset')}`  ",
        f"**File Size:** {metadata.get('file_size_bytes', 0):,} bytes | **Encoding:** `{metadata.get('encoding', 'utf-8')}` | **Delimiter:** `{metadata.get('delimiter', ',')}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Quality Baseline",
        "",
        f"- **Initial Health Score:** **{health_score:.1f} / 100.0**",
        f"- **Initial Dimensions:** {profile.get('total_rows', 0):,} rows × {profile.get('total_columns', 0):,} columns",
        f"- **Missing Value Burden:** {profile.get('total_missing_cells', 0):,} cells ({profile.get('total_missing_percentage', 0.0):.2f}% missingness)",
        f"- **Duplicate Rows:** {profile.get('duplicate_rows', 0):,} rows ({profile.get('duplicate_percentage', 0.0):.2f}%)",
        f"- **Total Diagnostic Defects Detected:** {len(issues)}",
        "",
    ]

    # Breakdown by severity
    severity_counts = original_report.get("issues_by_severity", {})
    lines.extend([
        "### Diagnostic Defects by Severity",
        "",
        "| Severity Tier | Issue Count | Description |",
        "| :--- | :---: | :--- |",
        f"| **High** | {severity_counts.get('high', 0)} | Critical defects (type contamination, 100% null columns, high duplicate rates) |",
        f"| **Medium** | {severity_counts.get('medium', 0)} | Zero-variance constant columns, negative values in positive domains |",
        f"| **Low** | {severity_counts.get('low', 0)} | Mild missingness, statistical extreme outliers, high cardinality keys |",
        "",
    ])

    # Issues detail table
    if issues:
        lines.extend([
            "### Catalog of Detected Defects",
            "",
            "| # | Severity | Defect Type | Target Column | Affected Count | Summary & Remediation |",
            "| :--- | :--- | :--- | :--- | :---: | :--- |",
        ])
        for idx, iss in enumerate(issues, start=1):
            target = f"`{iss.get('column')}`" if iss.get("column") else "*Dataset-level*"
            sev = iss.get("severity", "low").upper()
            lines.append(
                f"| {idx} | **{sev}** | `{iss.get('issue_type')}` | {target} | {iss.get('affected_count', 0)} | {iss.get('explanation')} |"
            )
        lines.append("")

    # Anomaly Detection Section (if present)
    if anomaly_summary:
        lines.extend([
            "---",
            "",
            "## 2. Unsupervised Anomaly Detection Summary",
            "",
            f"- **Detection Method:** `{anomaly_summary.get('method', 'Isolation Forest')}`",
            f"- **Features Evaluated:** `{anomaly_summary.get('features_used', [])}`",
            f"- **Total Evaluated Rows:** {anomaly_summary.get('evaluated_rows', 0):,}",
            f"- **Anomalies Identified:** {anomaly_summary.get('anomaly_count', 0):,} rows ({anomaly_summary.get('anomaly_percentage', 0.0):.2f}%)",
            f"- **Detection Latency:** {anomaly_summary.get('execution_time_ms', 0.0):.2f} ms",
            "",
            "> **Note on Statistical Anomalies:** An observation identified as anomalous is statistically unusual ",
            "> in the feature space; it does not automatically represent incorrect, fraudulent, or unusable data.",
            "",
        ])

    # Remediation & Audit Trail Section
    lines.extend([
        "---",
        "",
        "## 3. Approved Cleaning Actions & Audit Trail",
        "",
    ])

    if audit_trail and len(audit_trail) > 0:
        lines.extend([
            f"A total of **{len(audit_trail)}** operations were explicitly approved and executed in this session.",
            "",
            "| # | Proposal ID | Operation | Target Column | Affected Cells | Delta Rows | Status | Timestamp (UTC) |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |",
        ])
        for idx, entry in enumerate(audit_trail, start=1):
            col_str = f"`{entry.column}`" if entry.column else "*All*"
            row_delta = entry.rows_after - entry.rows_before
            delta_str = f"{row_delta:+d}" if row_delta != 0 else "0"
            lines.append(
                f"| {idx} | `{entry.proposal_id}` | `{entry.operation}` | {col_str} | {entry.affected_cells} | {delta_str} | **{entry.status}** | `{entry.timestamp[:19]}` |"
            )
        lines.append("")
    else:
        lines.extend([
            "*No cleaning operations were approved in this session. The dataset remains in its original form.*",
            "",
        ])

    # Quality Comparison Section
    if comparison:
        lines.extend([
            "---",
            "",
            "## 4. Before-and-After Quality Comparison",
            "",
            "| Quality Metric | Original Dataset | Cleaned Dataset | Net Change |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Dataset Rows** | {comparison.rows_before:,} | {comparison.rows_after:,} | {comparison.rows_diff:+d} |",
            f"| **Dataset Columns** | {comparison.cols_before} | {comparison.cols_after} | {comparison.cols_diff:+d} |",
            f"| **Missing Cells** | {comparison.missing_cells_before:,} | {comparison.missing_cells_after:,} | {comparison.missing_cells_diff:+d} |",
            f"| **Duplicate Rows** | {comparison.duplicate_rows_before:,} | {comparison.duplicate_rows_after:,} | {comparison.duplicate_rows_diff:+d} |",
            f"| **Remaining Defects** | {comparison.issues_count_before} | {comparison.issues_count_after} | {comparison.issues_count_diff:+d} |",
            f"| **Overall Health Score** | **{comparison.health_score_before:.1f}** | **{comparison.health_score_after:.1f}** | **{comparison.health_score_diff:+.1f}** |",
            "",
            "### Technical Evaluation Notes & Trade-offs",
            "",
        ])
        for note in comparison.evaluation_notes:
            lines.append(f"- {note}")
        lines.append("")

    # Governance & Immutability Notice
    lines.extend([
        "---",
        "",
        "## 5. Data Governance & Integrity Statement",
        "",
        "- **Zero-Tamper Guarantee:** DataDoctor AI never mutates source files or original DataFrames in place.",
        "- **Zero External Network Exfiltration:** All computations are executed 100% locally on-premise.",
        "- **Audit Trail Preserved:** Every transformation requires deliberate user authorization and produces a structured log.",
        "",
        "---",
        "*Report generated by DataDoctor AI — CognitionNext '26 Technical Assessment*",
    ])

    return "\n".join(lines)
