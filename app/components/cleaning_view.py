"""
DataDoctor AI - Reviewable Cleaning & Repair View
Integrates DataCleaner, proposed actions, dry-run previews, explicit approval guards,
audit logs, and before/after quality comparisons.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from core.cleaner import DataCleaner, CleaningProposal, ProposalNotApprovedError
from app.utils.visualizer import build_comparison_metric_chart


def render_cleaning_view(cleaner: DataCleaner):
    """Render interactive proposal generation, dry-run preview, approval, and audit trail."""
    st.subheader("Reviewable Data Cleaning & Repair Pipeline")

    st.warning(
        "**Safety Protocol:** DataDoctor AI never mutates your original uploaded data in place. "
        "Every repair operation requires your explicit approval, and transformations operate exclusively "
        "on a distinct working copy with a complete audit trail."
    )

    # 1. Proposal Generation
    col_gen1, col_gen2 = st.columns([2, 1])
    with col_gen1:
        st.markdown("### 1. Diagnostic Repair Proposals")
    with col_gen2:
        if st.button("Generate Cleaning Proposals", type="primary"):
            anom_res = st.session_state.get("latest_anomaly_result")
            anom_list = [anom_res] if anom_res else None
            cleaner.generate_proposals(anomaly_results=anom_list)
            st.success(f"Generated {len(cleaner.proposals)} cleaning proposals based on current diagnostics.")

    if not cleaner.proposals:
        st.info("No proposals generated yet. Click 'Generate Cleaning Proposals' to inspect detected defects and review safe remediation options.")
        return

    # List Proposals
    st.write(f"**Total Proposals Available:** {len(cleaner.proposals)}")

    for pid, prop in cleaner.proposals.items():
        is_applied = prop.proposal_id in cleaner._applied_ids
        status_label = "✅ APPLIED" if is_applied else "⏳ PENDING APPROVAL"
        target_str = f"Column `{prop.column}`" if prop.column else "Dataset-Level"

        with st.expander(f"[{status_label}] {prop.proposal_id} ({prop.operation}) — {target_str}", expanded=(not is_applied)):
            st.write(f"**Defect Type:** `{prop.issue_type}`")
            st.write(f"**Rationale:** {prop.rationale}")
            st.write(f"**Affected Cells/Rows:** {prop.affected_count:,} ({prop.affected_percentage:.1f}%)")
            st.error(f"⚠️ **Risks & Limitations:** {prop.risks}")

            if not is_applied:
                btn_prev, btn_app = st.columns([1, 1])

                with btn_prev:
                    if st.button("Preview Changes", key=f"prev_{pid}"):
                        preview = cleaner.preview_proposal(pid)
                        st.session_state[f"preview_result_{pid}"] = preview

                with btn_app:
                    if st.button("Approve & Apply", key=f"app_{pid}", type="primary"):
                        try:
                            cleaner.apply_proposal(pid, approved=True)
                            # Invalidate cached preview states because dataframe state changed
                            for k in list(st.session_state.keys()):
                                if k.startswith("preview_result_"):
                                    st.session_state.pop(k, None)
                            st.success(f"Successfully applied proposal {pid}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to apply proposal: {e}")

                # Display preview if generated
                if f"preview_result_{pid}" in st.session_state:
                    prev = st.session_state[f"preview_result_{pid}"]
                    st.markdown("##### Dry-Run Preview Results:")
                    st.write(f"- **Row Delta:** {prev.rows_before:,} -> {prev.rows_after:,} (net change: {prev.rows_after - prev.rows_before:+d})")
                    st.write(f"- **Explanation:** {prev.explanation}")
                    if prev.sample_diff:
                        st.caption("Sample Transformed Entries:")
                        st.dataframe(pd.DataFrame(prev.sample_diff), use_container_width=True)
            else:
                st.success(f"This proposal has been applied to the working dataset. Recorded in audit trail.")

    st.markdown("---")

    # 2. Structured Audit Trail
    st.markdown("### 2. Structured Audit Trail")
    audit_logs = cleaner.get_audit_log()
    if not audit_logs:
        st.caption("No operations have been applied yet.")
    else:
        st.dataframe(pd.DataFrame(audit_logs), use_container_width=True)

    st.markdown("---")

    # 3. Quality Comparison: Before vs After
    if cleaner.audit_log:
        st.markdown("### 3. Quality Comparison: Original vs. Cleaned Dataset")
        comparison = cleaner.compare_quality()

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Total Rows", f"{comparison.rows_after:,}", delta=f"{comparison.rows_diff:+d}")
        with q2:
            st.metric("Missing Cells", f"{comparison.missing_cells_after:,}", delta=f"{comparison.missing_cells_diff:+d}", delta_color="inverse")
        with q3:
            st.metric("Duplicate Rows", f"{comparison.duplicate_rows_after:,}", delta=f"{comparison.duplicate_rows_diff:+d}", delta_color="inverse")
        with q4:
            st.metric("Health Score", f"{comparison.health_score_after:.1f}", delta=f"{comparison.health_score_diff:+.1f}")

        # Comparison Chart
        comp_chart = build_comparison_metric_chart(comparison)
        st.plotly_chart(comp_chart, use_container_width=True)

        # Technical Trade-off Notes
        st.markdown("#### Technical Trade-off Assessment")
        for note in comparison.evaluation_notes:
            st.info(f"• {note}")
