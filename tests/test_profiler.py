"""
Unit tests for core.profiler module.
Verifies dataset profiling metrics, type inference, handling of nulls, duplicates,
constant columns, empty datasets, and immutability.
"""

import numpy as np
import pandas as pd
import pytest

from core.profiler import infer_column_type, profile_column, profile_dataset


class TestDataQualityProfiler:
    """Test suite for dataset and column profiler."""

    @pytest.fixture
    def clean_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000.0, 60000.0, 75000.0, 90000.0, 110000.0],
            "department": ["HR", "Eng", "Eng", "Sales", "HR"],
            "is_active": [True, True, False, True, False],
        })

    def test_clean_dataset_profiling(self, clean_df: pd.DataFrame):
        profile = profile_dataset(clean_df)

        assert profile.total_rows == 5
        assert profile.total_columns == 5
        assert profile.total_cells == 25
        assert profile.duplicate_rows == 0
        assert profile.duplicate_percentage == 0.0
        assert profile.total_missing_cells == 0
        assert profile.total_missing_percentage == 0.0
        assert len(profile.constant_columns) == 0
        assert len(profile.all_missing_columns) == 0
        assert "age" in profile.numeric_columns
        assert "salary" in profile.numeric_columns
        assert "department" in profile.categorical_columns
        assert "is_active" in profile.boolean_columns

    def test_missing_values_profiling(self):
        df = pd.DataFrame({
            "col_a": [1, 2, np.nan, 4, np.nan],  # 2 missing (40%)
            "col_b": [None, None, None, None, None],  # 5 missing (100%)
            "col_c": ["x", "y", "z", "w", "v"],  # 0 missing (0%)
        })

        profile = profile_dataset(df)

        assert profile.total_missing_cells == 7
        assert profile.total_missing_percentage == round(7 / 15 * 100, 2)
        assert "col_b" in profile.all_missing_columns

        prof_a = profile.columns["col_a"]
        assert prof_a.missing_count == 2
        assert prof_a.missing_percentage == 40.0
        assert prof_a.non_null_count == 3
        assert prof_a.distinct_count == 3

        prof_b = profile.columns["col_b"]
        assert prof_b.missing_count == 5
        assert prof_b.missing_percentage == 100.0
        assert prof_b.is_all_missing is True
        assert prof_b.inferred_type == "empty"

    def test_duplicate_rows_detection(self):
        df = pd.DataFrame({
            "a": [1, 2, 1, 1],
            "b": ["x", "y", "x", "x"],
        })

        profile = profile_dataset(df)
        # Row [1, 'x'] is repeated 3 times, so 2 duplicate rows exist
        assert profile.duplicate_rows == 2
        assert profile.duplicate_percentage == 50.0

    def test_constant_column_detection(self):
        df = pd.DataFrame({
            "normal": [1, 2, 3, 4],
            "constant_num": [42, 42, 42, 42],
            "constant_str": ["FIXED", "FIXED", "FIXED", "FIXED"],
        })

        profile = profile_dataset(df)
        assert "constant_num" in profile.constant_columns
        assert "constant_str" in profile.constant_columns
        assert profile.columns["constant_num"].is_constant is True
        assert profile.columns["constant_str"].is_constant is True

    def test_empty_dataframe_handling(self):
        empty_df = pd.DataFrame()
        profile = profile_dataset(empty_df)

        assert profile.total_rows == 0
        assert profile.total_columns == 0
        assert profile.total_cells == 0
        assert profile.duplicate_rows == 0
        assert profile.total_missing_cells == 0

    def test_empty_rows_dataframe_with_columns(self):
        df_cols_only = pd.DataFrame(columns=["a", "b", "c"])
        profile = profile_dataset(df_cols_only)

        assert profile.total_rows == 0
        assert profile.total_columns == 3
        assert profile.total_cells == 0
        assert profile.columns["a"].total_rows == 0
        assert profile.columns["a"].is_all_missing is True

    def test_numeric_statistics_accuracy(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        df = pd.DataFrame({"metric": values})

        profile = profile_dataset(df)
        prof_metric = profile.columns["metric"]

        stats = prof_metric.stats
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["q25"] == 20.0
        assert stats["q75"] == 40.0
        assert stats["iqr"] == 20.0
        assert stats["negative_count"] == 0
        assert stats["infinite_count"] == 0

    def test_input_dataframe_immutability(self, clean_df: pd.DataFrame):
        df_copy_before = clean_df.copy(deep=True)

        _ = profile_dataset(clean_df)

        pd.testing.assert_frame_equal(clean_df, df_copy_before)
