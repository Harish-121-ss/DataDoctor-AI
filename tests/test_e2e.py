"""
End-to-end integration tests for Stage 1.
Verifies complete flow: file on disk -> load_csv -> profile_dataset -> diagnose_dataset -> generate_quality_report.
"""

from pathlib import Path

import pandas as pd
import pytest

from core.diagnostics import generate_quality_report
from core.ingestion import load_csv
from core.profiler import profile_dataset


class TestEndToEndPipeline:
    """End-to-end integration tests for Stage 1 data-processing engine."""

    def test_end_to_end_dirty_csv_diagnosis(self, tmp_path: Path):
        # Create a realistic dirty CSV file
        raw_csv_content = (
            "patient_id;age;blood_pressure;cholesterol;city;constant_code\n"
            "1001;45;120.0;200.0;New York;CLINIC_A\n"
            "1002;-5;135.0;220.0;Chicago;CLINIC_A\n"        # Suspicious negative age
            "1003;50;;210.0;Boston;CLINIC_A\n"             # Missing blood pressure
            "1004;62;140.0;unknown;Seattle;CLINIC_A\n"     # Type inconsistency in cholesterol
            "1001;45;120.0;200.0;New York;CLINIC_A\n"     # Duplicate row
        )
        test_file = tmp_path / "patients_dirty.csv"
        test_file.write_text(raw_csv_content, encoding="utf-8")

        # 1. Ingestion
        df, metadata = load_csv(test_file)
        assert len(df) == 5
        assert metadata["delimiter"] == ";"
        assert metadata["row_count"] == 5
        assert metadata["col_count"] == 6

        # 2. Profiling
        profile = profile_dataset(df)
        assert profile.duplicate_rows == 1
        assert "constant_code" in profile.constant_columns
        assert profile.columns["blood_pressure"].missing_count == 1

        # 3. Comprehensive Quality Report & Diagnostics
        report = generate_quality_report(df)
        assert report["health_score"] < 90.0  # Quality score penalized
        assert report["total_issues"] >= 4

        issue_types = {issue["issue_type"] for issue in report["issues"]}
        assert "DUPLICATE_ROWS" in issue_types
        assert "CONSTANT_COLUMN" in issue_types
        assert "SUSPICIOUS_NEGATIVE_VALUE" in issue_types
        assert "TYPE_INCONSISTENCY" in issue_types

        # Ensure suggestions exist on every issue
        for issue in report["issues"]:
            assert "suggested_action" in issue
            assert len(issue["suggested_action"]) > 0
            assert "severity" in issue
            assert issue["severity"] in ("high", "medium", "low")
