"""
DataDoctor AI - Exports & Downloads View
Provides secure downloads for Cleaned CSV, Original CSV, Markdown Audit Report,
and Structured Audit Trail Log.
"""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd
from core.cleaner import DataCleaner
from app.utils.report_gen import generate_markdown_audit_report


def render_export_view(
    cleaner: DataCleaner,
    metadata: dict,
    original_report: dict,
):
    """Render export options and downloads."""
    st.subheader("Data & Audit Report Downloads")

    st.markdown(
        "Download your validated datasets and compliance documentation. "
        "Original and cleaned datasets are clearly labeled to prevent version confusion."
    )

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("### 1. Dataset Exports")

        # Cleaned CSV Download
        cleaned_csv_data = cleaner.current_df.to_csv(index=False).encode("utf-8")
        is_modified = len(cleaner.audit_log) > 0
        cleaned_label = "Cleaned Dataset" if is_modified else "Current Working Dataset (Unmodified)"

        st.write(f"**{cleaned_label}** ({len(cleaner.current_df):,} rows × {len(cleaner.current_df.columns)} columns)")
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=cleaned_csv_data,
            file_name="datadoctor_cleaned_dataset.csv",
            mime="text/csv",
            type="primary" if is_modified else "secondary",
        )

        st.markdown("")

        # Original CSV Download
        original_csv_data = cleaner.original_df.to_csv(index=False).encode("utf-8")
        st.write(f"**Original Raw Dataset** ({len(cleaner.original_df):,} rows × {len(cleaner.original_df.columns)} columns)")
        st.download_button(
            label="📥 Download Untouched Original CSV",
            data=original_csv_data,
            file_name="datadoctor_original_dataset.csv",
            mime="text/csv",
        )

    with col_e2:
        st.markdown("### 2. Audit & Compliance Reports")

        # Generate Report Markdown
        comparison = cleaner.compare_quality() if cleaner.audit_log else None
        anom_res = st.session_state.get("latest_anomaly_result")
        anom_dict = anom_res.to_dict() if anom_res else None

        md_report = generate_markdown_audit_report(
            metadata=metadata,
            original_report=original_report,
            cleaned_df=cleaner.current_df,
            comparison=comparison,
            audit_trail=cleaner.audit_log,
            anomaly_summary=anom_dict,
        )

        st.write("**Data Quality & Remediation Audit Report (Markdown)**")
        st.download_button(
            label="📄 Download Quality Audit Report (.md)",
            data=md_report.encode("utf-8"),
            file_name="datadoctor_quality_audit_report.md",
            mime="text/markdown",
        )

        st.markdown("")

        # Audit Log Export
        audit_logs = cleaner.get_audit_log()
        audit_json = json.dumps(audit_logs, indent=2).encode("utf-8")
        st.write(f"**Structured Cleaning Audit Log** ({len(audit_logs)} operations recorded)")
        st.download_button(
            label="📋 Download Audit Trail Log (.json)",
            data=audit_json,
            file_name="datadoctor_audit_log.json",
            mime="application/json",
            disabled=(len(audit_logs) == 0),
        )

    # Preview of the Markdown Audit Report
    st.markdown("---")
    with st.expander("📄 Preview Generated Quality Audit Report", expanded=False):
        st.markdown(md_report)
