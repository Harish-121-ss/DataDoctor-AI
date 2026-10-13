"""
DataDoctor AI - Data Quality Analysis & Profiling View
Displays overall health score, missingness distributions, defect catalogs,
and numerical feature spread.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from core.profiler import DatasetProfile
from app.utils.visualizer import (
    build_missing_values_chart,
    build_issues_severity_chart,
    build_feature_distribution_chart,
)


def render_profiling_view(df: pd.DataFrame, report: dict, profile: DatasetProfile):
    """Render comprehensive data quality diagnosis and profiling metrics."""
    st.subheader("Data Quality Health & Profiling Diagnosis")

    health_score = report.get("health_score", 100.0)
    issues = report.get("issues", [])
    issues_by_sev = report.get("issues_by_severity", {})

    # Top Health Score Banner
    col_h1, col_h2 = st.columns([1, 2])
    with col_h1:
        st.metric("Overall Health Score", f"{health_score:.1f} / 100.0")
    with col_h2:
        st.info(
            "**Health Score Calculation:** Starts at 100.0 and applies calibrated deductions for "
            "missingness (up to -30), duplicate rows (up to -20), and defect severity "
            f"(-5 for High, -2.5 for Medium, -1 for Low). Current deductions: High ({issues_by_sev.get('high', 0)}), "
            f"Medium ({issues_by_sev.get('medium', 0)}), Low ({issues_by_sev.get('low', 0)})."
        )

    # Key Quality Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Missing Cells", f"{profile.total_missing_cells:,} ({profile.total_missing_percentage:.1f}%)")
    with c2:
        st.metric("Duplicate Rows", f"{profile.duplicate_rows:,} ({profile.duplicate_percentage:.1f}%)")
    with c3:
        st.metric("Constant Columns", f"{len(profile.constant_columns)}")
    with c4:
        st.metric("Total Issues", f"{len(issues)}")

    st.markdown("---")

    # Visualizations Row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        missing_fig = build_missing_values_chart(profile)
        if missing_fig:
            st.plotly_chart(missing_fig, use_container_width=True)
        else:
            st.success("No missing values detected across any column.")

    with chart_col2:
        sev_fig = build_issues_severity_chart(issues_by_sev)
        if sev_fig:
            st.plotly_chart(sev_fig, use_container_width=True)
        else:
            st.success("Zero diagnostic defects detected.")

    st.markdown("---")

    # Feature Distribution Inspector
    numeric_cols = [c for c, p in profile.columns.items() if p.inferred_type == "numeric"]
    if numeric_cols:
        st.markdown("### Numerical Feature Distribution Inspector")
        sel_col = st.selectbox("Select Numerical Feature to Inspect", options=numeric_cols, key="dist_inspector_col")
        dist_fig = build_feature_distribution_chart(df, sel_col)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)

    # Detailed Defects Catalog
    st.markdown("### Detected Data Quality Defects")
    if not issues:
        st.success("Dataset is clean. No defects identified.")
    else:
        for idx, iss in enumerate(issues, start=1):
            sev = iss.get("severity", "low").upper()
            badge_icon = "🔴" if sev == "HIGH" else "🟠" if sev == "MEDIUM" else "🔵"
            col_target = f"Column `{iss.get('column')}`" if iss.get("column") else "Dataset-Level"

            with st.expander(f"{badge_icon} #{idx} [{sev}] {iss.get('issue_type')} — {col_target}", expanded=(idx <= 3)):
                st.write(f"**Explanation:** {iss.get('explanation')}")
                st.write(f"**Affected Cells/Rows:** {iss.get('affected_count', 0)} ({iss.get('affected_percentage', 0.0):.1f}%)")
                st.write(f"**Suggested Remediation:** {iss.get('suggested_action')}")
                if iss.get("details"):
                    st.json(iss.get("details"))
