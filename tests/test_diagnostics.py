"""
Unit tests for core.diagnostics module.
Verifies rule-based diagnostics, issue classification, severity ordering,
type inconsistencies, suspicious values, and quality report generation.
"""

import numpy as np
import pandas as pd
import pytest

from core.diagnostics import (
    calculate_health_score,
    detect_suspicious_values,
    detect_type_inconsistencies,
    diagnose_dataset,
    generate_quality_report,
)
from core.profiler import profile_dataset


class TestDiagnosticsEngine:
    """Test suite for diagnostics and quality report generator."""

    def test_clean_dataset_has_no_critical_issues(self):
        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "score": [80.0, 85.0, 90.0, 95.0, 100.0],
            "category": ["A", "B", "A", "B", "A"],
        })

        report = generate_quality_report(df)
        assert report["issues_by_severity"]["high"] == 0
        assert report["health_score"] >= 95.0

    def test_duplicate_rows_issue(self):
        df = pd.DataFrame({
            "x": [1, 2, 1, 1],
            "y": ["a", "b", "a", "a"],
        })

        issues = diagnose_dataset(df)
        dup_issues = [i for i in issues if i.issue_type == "DUPLICATE_ROWS"]
        assert len(dup_issues) == 1
        assert dup_issues[0].affected_count == 2
        assert "Deduplicate" in dup_issues[0].suggested_action

    def test_all_missing_column_issue(self):
        df = pd.DataFrame({
            "valid": [1, 2, 3],
            "completely_empty": [np.nan, np.nan, np.nan],
        })

        issues = diagnose_dataset(df)
        empty_issues = [i for i in issues if i.issue_type == "ALL_MISSING_COLUMN"]
        assert len(empty_issues) == 1
        assert empty_issues[0].severity == "high"
        assert empty_issues[0].column == "completely_empty"
        assert empty_issues[0].affected_percentage == 100.0

    def test_partial_missing_values_high_vs_low(self):
        # 10 rows: col_high has 6 missing (60% -> HIGH), col_low has 2 missing (20% -> LOW)
        df = pd.DataFrame({
            "col_high": [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2, 3, 4],
            "col_low": [1, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan],
        })

        issues = diagnose_dataset(df)
        high_missing = [i for i in issues if i.issue_type == "HIGH_MISSING_VALUES"]
        low_missing = [i for i in issues if i.issue_type == "MISSING_VALUES"]

        assert len(high_missing) == 1
        assert high_missing[0].column == "col_high"
        assert high_missing[0].severity == "high"

        assert len(low_missing) == 1
        assert low_missing[0].column == "col_low"
        assert low_missing[0].severity == "low"

    def test_constant_column_issue(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "constant_flag": ["ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE"],
        })

        issues = diagnose_dataset(df)
        const_issues = [i for i in issues if i.issue_type == "CONSTANT_COLUMN"]
        assert len(const_issues) == 1
        assert const_issues[0].column == "constant_flag"
        assert const_issues[0].severity == "medium"

    def test_type_inconsistency_detection(self):
        # Numeric column contaminated with text placeholders like 'unknown', '$100'
        df = pd.DataFrame({
            "salary": ["50000", "60000", "unknown", "75000", "pending", "80000", "90000"],
        })

        issues = diagnose_dataset(df)
        type_issues = [i for i in issues if i.issue_type == "TYPE_INCONSISTENCY"]
        assert len(type_issues) == 1
        assert type_issues[0].column == "salary"
        assert type_issues[0].severity == "high"
        assert type_issues[0].affected_count == 2
        assert "sample_invalid_values" in type_issues[0].details

    def test_suspicious_negative_values_in_positive_domains(self):
        df = pd.DataFrame({
            "customer_age": [25, 30, -5, 45, -1],
            "item_price": [10.5, -2.0, 15.0, 20.0, 30.0],
            "temperature_celsius": [-10, 0, 15, 25, 30],  # Temperature can naturally be negative
        })

        issues = diagnose_dataset(df)
        neg_issues = [i for i in issues if i.issue_type == "SUSPICIOUS_NEGATIVE_VALUE"]

        flagged_cols = {i.column for i in neg_issues}
        assert "customer_age" in flagged_cols
        assert "item_price" in flagged_cols
        # temperature_celsius is NOT flagged because it is not in the non-negative list
        assert "temperature_celsius" not in flagged_cols

    def test_infinite_values_detection(self):
        df = pd.DataFrame({
            "ratio": [1.0, 2.5, np.inf, 4.0, -np.inf],
        })

        issues = diagnose_dataset(df)
        inf_issues = [i for i in issues if i.issue_type == "SUSPICIOUS_INFINITE_VALUE"]
        assert len(inf_issues) == 1
        assert inf_issues[0].affected_count == 2
        assert inf_issues[0].severity == "high"

    def test_quality_report_structure(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "val": [10, 20, 30],
        })

        report = generate_quality_report(df)
        assert "health_score" in report
        assert "total_issues" in report
        assert "issues_by_severity" in report
        assert "profile" in report
        assert "issues" in report
        assert isinstance(report["health_score"], float)
        assert isinstance(report["issues"], list)
