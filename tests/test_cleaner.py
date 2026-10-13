"""
Unit tests for core.cleaner module (Reviewable Data Cleaning & Repair Engine).
Verifies proposal generation, dry-run previews, mandatory explicit approval,
idempotency, error handling, audit trail logging, before-and-after profiling,
and absolute immutability of the original dataset.
"""

import numpy as np
import pandas as pd
import pytest

from core.cleaner import (
    CleaningProposal,
    DataCleaner,
    InvalidProposalError,
    ProposalAlreadyAppliedError,
    ProposalNotApprovedError,
    apply_cleaning_proposal,
    compare_datasets,
    generate_cleaning_proposals,
    preview_cleaning_proposal,
)
from core.ml_anomaly import detect_isolation_forest_anomalies


class TestReviewableDataCleaner:
    """Test suite for DataDoctor AI reviewable cleaner."""

    @pytest.fixture
    def dirty_df(self) -> pd.DataFrame:
        """Realistic sample dirty DataFrame for test fixtures."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 1],                      # Duplicate row with index 0 and 4
            "name": [" Alice ", "Bob", "Charlie  ", "Diana", " Alice "],  # Whitespace
            "age": [25.0, -4.0, 35.0, np.nan, 25.0],    # Suspicious negative (-4) and missing
            "salary": ["$50,000", "$60,000", "$75,000", "invalid", "$50,000"],  # Type inconsistency
            "dept": ["Eng", "HR", None, "Sales", "Eng"],  # Categorical missing
            "constant_flag": ["YES", "YES", "YES", "YES", "YES"],  # Constant column
        })

    def test_proposal_generation(self, dirty_df: pd.DataFrame):
        proposals = generate_cleaning_proposals(dirty_df)

        assert len(proposals) > 0
        prop_types = {p.issue_type for p in proposals}
        operations = {p.operation for p in proposals}

        assert "DUPLICATE_ROWS" in prop_types
        assert "deduplicate_rows" in operations
        assert "CONSTANT_COLUMN" in prop_types
        assert "drop_column" in operations
        assert "TYPE_INCONSISTENCY" in prop_types
        assert "convert_numeric" in operations
        assert "INCONSISTENT_TEXT" in prop_types
        assert "normalize_text" in operations
        assert "SUSPICIOUS_NEGATIVE_VALUE" in prop_types
        assert "clip_bounds" in operations

        # Verify all proposals start in pending status
        for p in proposals:
            assert p.status == "pending"
            assert len(p.rationale) > 0
            assert len(p.risks) > 0

    def test_missing_value_proposals_numeric_and_categorical(self, dirty_df: pd.DataFrame):
        proposals = generate_cleaning_proposals(dirty_df)

        # Numeric missing in 'age': median imputation proposal
        num_prop = next(p for p in proposals if p.column == "age" and p.operation == "impute_median")
        assert num_prop.issue_type == "MISSING_VALUES"
        assert "imputed_value" in num_prop.details

        # Categorical missing in 'dept': mode imputation proposal
        cat_prop = next(p for p in proposals if p.column == "dept" and p.operation == "impute_mode")
        assert cat_prop.issue_type == "MISSING_VALUES"
        assert cat_prop.details["imputed_value"] in ("Eng", "HR", "Sales")

    def test_preview_does_not_mutate_data(self, dirty_df: pd.DataFrame):
        df_copy_before = dirty_df.copy(deep=True)
        cleaner = DataCleaner(dirty_df)
        proposals = cleaner.generate_proposals()

        dedup_prop = next(p for p in proposals if p.operation == "deduplicate_rows")
        preview = cleaner.preview_proposal(dedup_prop.proposal_id)

        assert preview.rows_before == 5
        assert preview.rows_after == 4
        assert preview.affected_rows == 1
        assert len(preview.sample_diff) > 0

        # DataFrame inside cleaner and original DataFrame must remain completely unchanged
        pd.testing.assert_frame_equal(cleaner.current_df, df_copy_before)
        pd.testing.assert_frame_equal(dirty_df, df_copy_before)

    def test_explicit_approval_requirement(self, dirty_df: pd.DataFrame):
        proposals = generate_cleaning_proposals(dirty_df)
        dedup_prop = next(p for p in proposals if p.operation == "deduplicate_rows")

        # Without approval (approved=False), must raise ProposalNotApprovedError
        with pytest.raises(ProposalNotApprovedError, match="without explicit approval"):
            apply_cleaning_proposal(dirty_df, dedup_prop, approved=False)

        # Also when using DataCleaner session
        cleaner = DataCleaner(dirty_df)
        cleaner.generate_proposals()
        with pytest.raises(ProposalNotApprovedError):
            cleaner.apply_proposal(dedup_prop.proposal_id, approved=False)

    def test_safe_type_conversion_and_numeric_casting(self, dirty_df: pd.DataFrame):
        proposals = generate_cleaning_proposals(dirty_df)
        cast_prop = next(p for p in proposals if p.operation == "convert_numeric")

        cleaned_df, audit = apply_cleaning_proposal(dirty_df, cast_prop, approved=True)

        assert pd.api.types.is_numeric_dtype(cleaned_df["salary"])
        assert cleaned_df["salary"].iloc[0] == 50000.0
        assert cleaned_df["salary"].iloc[1] == 60000.0
        # "invalid" entry was coerced safely to NaN
        assert pd.isna(cleaned_df["salary"].iloc[3])
        assert audit.status == "SUCCESS"
        assert audit.affected_cells > 0

    def test_whitespace_normalization(self, dirty_df: pd.DataFrame):
        proposals = generate_cleaning_proposals(dirty_df)
        norm_prop = next(p for p in proposals if p.column == "name" and p.operation == "normalize_text")

        cleaned_df, audit = apply_cleaning_proposal(dirty_df, norm_prop, approved=True)

        assert cleaned_df["name"].iloc[0] == "Alice"
        assert cleaned_df["name"].iloc[2] == "Charlie"
        assert audit.status == "SUCCESS"

    def test_suspicious_negative_clipping(self, dirty_df: pd.DataFrame):
        proposals = generate_cleaning_proposals(dirty_df)
        clip_prop = next(p for p in proposals if p.column == "age" and p.operation == "clip_bounds")

        cleaned_df, audit = apply_cleaning_proposal(dirty_df, clip_prop, approved=True)

        # -4.0 should be clipped to 0.0
        assert cleaned_df["age"].iloc[1] == 0.0
        assert cleaned_df["age"].iloc[0] == 25.0
        assert audit.status == "SUCCESS"

    def test_anomaly_review_proposals_integration(self):
        # Generate numeric dataset with known outlier
        df = pd.DataFrame({
            "val1": [10.0, 11.0, 10.5, 9.8, 10.2, 500.0],
            "val2": [20.0, 21.0, 20.5, 19.8, 20.2, 800.0],
        })

        anom_res = detect_isolation_forest_anomalies(df, contamination=0.20, random_state=42)
        assert anom_res.anomaly_count > 0

        proposals = generate_cleaning_proposals(df, anomaly_results=[anom_res])

        flag_prop = next(p for p in proposals if p.operation == "flag_anomalies")
        filter_prop = next(p for p in proposals if p.operation == "filter_anomalies")

        assert flag_prop.issue_type == "ANOMALY_REVIEW"
        assert filter_prop.issue_type == "ANOMALY_REVIEW"

        # Apply non-destructive flag proposal
        df_flagged, audit_flag = apply_cleaning_proposal(df, flag_prop, approved=True)
        assert "is_anomaly_isolation_forest" in df_flagged.columns
        assert len(df_flagged) == 6
        assert df_flagged["is_anomaly_isolation_forest"].sum() == anom_res.anomaly_count

        # Apply filter proposal
        df_filtered, audit_filter = apply_cleaning_proposal(df, filter_prop, approved=True)
        assert len(df_filtered) == 6 - anom_res.anomaly_count

    def test_prevent_accidental_double_application(self, dirty_df: pd.DataFrame):
        cleaner = DataCleaner(dirty_df)
        proposals = cleaner.generate_proposals()
        dedup_prop = next(p for p in proposals if p.operation == "deduplicate_rows")

        # First apply succeeds
        cleaner.apply_proposal(dedup_prop.proposal_id, approved=True)

        # Second apply of identical proposal must raise ProposalAlreadyAppliedError
        with pytest.raises(ProposalAlreadyAppliedError, match="already been applied"):
            cleaner.apply_proposal(dedup_prop.proposal_id, approved=True)

    def test_rejection_of_invalid_proposal(self, dirty_df: pd.DataFrame):
        bad_prop = CleaningProposal(
            proposal_id="PROP-BAD-999",
            issue_type="MISSING_VALUES",
            column="non_existent_column",
            operation="impute_median",
            params={"column": "non_existent_column"},
            rationale="Test bad column",
            affected_count=1,
            affected_percentage=20.0,
            risks="None",
        )

        with pytest.raises(InvalidProposalError, match="does not exist"):
            preview_cleaning_proposal(dirty_df, bad_prop)

        with pytest.raises(InvalidProposalError):
            apply_cleaning_proposal(dirty_df, bad_prop, approved=True)

    def test_audit_log_structure(self, dirty_df: pd.DataFrame):
        cleaner = DataCleaner(dirty_df)
        cleaner.generate_proposals()

        dedup_p = next(p for p in cleaner.proposals.values() if p.operation == "deduplicate_rows")
        cleaner.apply_proposal(dedup_p.proposal_id, approved=True)

        logs = cleaner.get_audit_log()
        assert len(logs) == 1
        record = logs[0]
        assert record["proposal_id"] == dedup_p.proposal_id
        assert record["operation"] == "deduplicate_rows"
        assert record["status"] == "SUCCESS"
        assert record["rows_before"] == 5
        assert record["rows_after"] == 4
        assert "timestamp" in record

    def test_quality_comparison_before_and_after(self, dirty_df: pd.DataFrame):
        cleaner = DataCleaner(dirty_df)
        cleaner.generate_proposals()

        dedup_p = next(p for p in cleaner.proposals.values() if p.operation == "deduplicate_rows")
        cleaner.apply_proposal(dedup_p.proposal_id, approved=True)

        comparison = cleaner.compare_quality()
        assert comparison.rows_before == 5
        assert comparison.rows_after == 4
        assert comparison.rows_diff == -1
        assert comparison.duplicate_rows_before == 1
        assert comparison.duplicate_rows_after == 0
        assert comparison.applied_operations_count == 1
        assert len(comparison.evaluation_notes) > 0

    def test_empty_dataframe_cleaning(self):
        empty_df = pd.DataFrame()
        cleaner = DataCleaner(empty_df)
        props = cleaner.generate_proposals()
        assert len(props) == 0

        comp = cleaner.compare_quality()
        assert comp.rows_before == 0
        assert comp.rows_after == 0
