"""
Unit tests for Streamlit application components, visualizers, and report generator.
Verifies chart construction, report generation, dashboard imports, and export formats.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import plotly.graph_objects as go

from app.utils.report_gen import generate_markdown_audit_report
from app.utils.visualizer import (
    build_anomaly_scatter_chart,
    build_comparison_metric_chart,
    build_feature_distribution_chart,
    build_issues_severity_chart,
    build_missing_values_chart,
)
from core.cleaner import DataCleaner, QualityComparison
from core.diagnostics import generate_quality_report
from core.profiler import profile_dataset


class TestUIComponentsAndReports:
    """Test suite for UI visualizers, report generation, and exports."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "age": [25.0, 30.0, None, 45.0, 50.0],
            "salary": [50000.0, 60000.0, 75000.0, 90000.0, 120000.0],
            "dept": ["Eng", "HR", "Eng", None, "Sales"],
        })

    def test_missing_values_chart(self, sample_df: pd.DataFrame):
        profile = profile_dataset(sample_df)
        fig = build_missing_values_chart(profile)

        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_issues_severity_chart(self, sample_df: pd.DataFrame):
        report = generate_quality_report(sample_df)
        fig = build_issues_severity_chart(report["issues_by_severity"])

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_feature_distribution_chart(self, sample_df: pd.DataFrame):
        fig = build_feature_distribution_chart(sample_df, "salary")
        assert fig is not None
        assert isinstance(fig, go.Figure)

        # Non-existent column returns None safely
        fig_none = build_feature_distribution_chart(sample_df, "non_existent")
        assert fig_none is None

    def test_anomaly_scatter_chart(self, sample_df: pd.DataFrame):
        preds = [False, False, False, False, True]
        fig = build_anomaly_scatter_chart(sample_df, "age", "salary", preds)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_comparison_metric_chart(self):
        comparison = QualityComparison(
            rows_before=100,
            rows_after=95,
            rows_diff=-5,
            cols_before=5,
            cols_after=5,
            cols_diff=0,
            missing_cells_before=10,
            missing_cells_after=0,
            missing_cells_diff=-10,
            duplicate_rows_before=5,
            duplicate_rows_after=0,
            duplicate_rows_diff=-5,
            issues_count_before=4,
            issues_count_after=1,
            issues_count_diff=-3,
            health_score_before=80.0,
            health_score_after=95.0,
            health_score_diff=15.0,
            applied_operations_count=2,
            evaluation_notes=["Test note"],
        )
        fig = build_comparison_metric_chart(comparison)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_markdown_audit_report_generation(self, sample_df: pd.DataFrame):
        metadata = {
            "source": "test_data.csv",
            "file_size_bytes": 1024,
            "encoding": "utf-8",
            "delimiter": ",",
        }
        report = generate_quality_report(sample_df)
        cleaner = DataCleaner(sample_df)
        cleaner.generate_proposals()

        md_output = generate_markdown_audit_report(
            metadata=metadata,
            original_report=report,
            cleaned_df=cleaner.current_df,
            comparison=None,
            audit_trail=[],
        )

        assert isinstance(md_output, str)
        assert "# DataDoctor AI — Dataset Quality & Remediation Audit Report" in md_output
        assert "Executive Summary & Quality Baseline" in md_output
        assert "Zero-Tamper Guarantee" in md_output
        assert "test_data.csv" in md_output

    def test_app_main_module_importable(self):
        # Ensure app.main and all components import cleanly without runtime errors
        import app.main as app_main
        assert hasattr(app_main, "load_dataset")
        assert hasattr(app_main, "main")

    def test_markdown_audit_report_with_remediation_and_comparison(self, sample_df: pd.DataFrame):
        metadata = {
            "source": "remediated_data.csv",
            "file_size_bytes": 2048,
            "encoding": "utf-8",
            "delimiter": ",",
        }
        report = generate_quality_report(sample_df)
        cleaner = DataCleaner(sample_df)
        cleaner.generate_proposals()

        # Find and apply an imputation proposal
        impute_pid = [pid for pid, p in cleaner.proposals.items() if p.operation.startswith("impute_")][0]
        cleaner.apply_proposal(impute_pid, approved=True)

        comparison = cleaner.compare_quality()
        md_output = generate_markdown_audit_report(
            metadata=metadata,
            original_report=report,
            cleaned_df=cleaner.current_df,
            comparison=comparison,
            audit_trail=cleaner.audit_log,
            anomaly_summary={"method": "Isolation Forest", "anomaly_count": 0, "evaluated_rows": 5, "execution_time_ms": 12.5},
        )

        assert "## 3. Approved Cleaning Actions & Audit Trail" in md_output
        assert impute_pid in md_output
        assert "## 4. Before-and-After Quality Comparison" in md_output
        assert "Overall Health Score" in md_output

    def test_json_audit_log_serialization_and_fields(self, sample_df: pd.DataFrame):
        cleaner = DataCleaner(sample_df)
        cleaner.generate_proposals()
        impute_pid = [pid for pid, p in cleaner.proposals.items() if p.operation.startswith("impute_")][0]
        cleaner.apply_proposal(impute_pid, approved=True)

        audit_dicts = cleaner.get_audit_log()
        assert len(audit_dicts) == 1
        record = audit_dicts[0]

        # Verify JSON serializability
        serialized = json.dumps(audit_dicts, indent=2)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 1
        assert deserialized[0]["proposal_id"] == impute_pid
        assert deserialized[0]["status"] == "SUCCESS"

        # Verify privacy: no raw row dump
        assert "row_data" not in record
        assert "raw_values" not in record

    def test_cleaned_csv_export_content_fidelity(self, sample_df: pd.DataFrame):
        cleaner = DataCleaner(sample_df)
        cleaner.generate_proposals()
        impute_pid = [pid for pid, p in cleaner.proposals.items() if p.operation.startswith("impute_")][0]
        cleaner.apply_proposal(impute_pid, approved=True)

        csv_str = cleaner.current_df.to_csv(index=False)
        reloaded = pd.read_csv(pd.io.common.StringIO(csv_str))

        assert len(reloaded) == len(cleaner.current_df)
        # Original had missing value, cleaned should have fewer or zero missing in target column
        target_col = cleaner.proposals[impute_pid].column
        assert reloaded[target_col].isna().sum() < sample_df[target_col].isna().sum()

