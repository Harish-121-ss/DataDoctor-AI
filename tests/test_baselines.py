"""
Unit tests for core.baselines (Statistical Z-Score and IQR Detectors).
Verifies detection accuracy, combination rules, missing value handling,
constant columns, and input DataFrame immutability.
"""

import numpy as np
import pandas as pd
import pytest

from core.baselines import (
    detect_iqr_anomalies,
    detect_zscore_anomalies,
    select_usable_numerical_features,
)


class TestStatisticalBaselines:
    """Test suite for Z-score and IQR anomaly detectors."""

    @pytest.fixture
    def baseline_dataset(self) -> pd.DataFrame:
        rng = np.random.default_rng(101)
        # 100 normal observations around mean=50, std=5
        x = rng.normal(50.0, 5.0, size=100)
        y = rng.normal(100.0, 10.0, size=100)
        # Inject 3 obvious outliers at indices 100, 101, 102
        x_outliers = np.array([120.0, 50.0, 150.0])   # Row 100 and 102 outlier in x
        y_outliers = np.array([100.0, 250.0, 300.0])  # Row 101 and 102 outlier in y
        all_x = np.concatenate([x, x_outliers])
        all_y = np.concatenate([y, y_outliers])
        return pd.DataFrame({"metric_x": all_x, "metric_y": all_y})

    def test_zscore_detection_on_deliberate_outliers(self, baseline_dataset: pd.DataFrame):
        res = detect_zscore_anomalies(baseline_dataset, threshold=3.0, combination_rule="any")

        assert res.total_rows == 103
        assert res.evaluated_rows == 103
        assert res.anomaly_count >= 3

        # Indices 100, 101, 102 must be flagged
        flagged = set(res.anomaly_indices)
        assert 100 in flagged
        assert 101 in flagged
        assert 102 in flagged

    def test_iqr_detection_on_deliberate_outliers(self, baseline_dataset: pd.DataFrame):
        res = detect_iqr_anomalies(baseline_dataset, k=1.5, combination_rule="any")

        assert res.total_rows == 103
        assert res.evaluated_rows == 103
        assert res.anomaly_count >= 3

        flagged = set(res.anomaly_indices)
        assert 100 in flagged
        assert 101 in flagged
        assert 102 in flagged

    def test_combination_rules_any_vs_all(self, baseline_dataset: pd.DataFrame):
        # Row 100 is outlier in x only; Row 101 in y only; Row 102 in both x and y
        res_any = detect_zscore_anomalies(baseline_dataset, threshold=3.0, combination_rule="any")
        res_all = detect_zscore_anomalies(baseline_dataset, threshold=3.0, combination_rule="all")

        # 'all' requires both features to be outliers, so row 102 should be flagged but not 100 or 101
        assert 102 in res_all.anomaly_indices
        assert 100 not in res_all.anomaly_indices
        assert 101 not in res_all.anomaly_indices
        assert len(res_all.anomaly_indices) < len(res_any.anomaly_indices)

    def test_dataframe_immutability(self, baseline_dataset: pd.DataFrame):
        df_copy_before = baseline_dataset.copy(deep=True)

        _ = detect_zscore_anomalies(baseline_dataset)
        _ = detect_iqr_anomalies(baseline_dataset)

        pd.testing.assert_frame_equal(baseline_dataset, df_copy_before)

    def test_constant_column_handling(self):
        df = pd.DataFrame({
            "constant_metric": [42.0] * 50,
            "varying_metric": list(range(50)),
        })

        res_z = detect_zscore_anomalies(df)
        res_iqr = detect_iqr_anomalies(df)

        # Constant column should be safely excluded without crash or ZeroDivisionError
        assert "varying_metric" in res_z.features_used
        assert not any("constant_metric" == f for f in res_z.features_used)
        assert "varying_metric" in res_iqr.features_used
        assert not any("constant_metric" == f for f in res_iqr.features_used)

    def test_missing_and_infinite_values_handling(self):
        df = pd.DataFrame({
            "col_a": [10.0, 12.0, np.nan, 15.0, 100.0],
            "col_b": [20.0, np.inf, 22.0, 25.0, 200.0],
        })

        res_z = detect_zscore_anomalies(df)
        res_iqr = detect_iqr_anomalies(df)

        # Rows with NaN/Inf (index 1 and 2) are not evaluated
        assert res_z.total_rows == 5
        assert res_z.evaluated_rows == 3
        assert res_z.predictions[1] is False
        assert res_z.predictions[2] is False

        assert res_iqr.total_rows == 5
        assert res_iqr.evaluated_rows == 3

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        res_z = detect_zscore_anomalies(empty_df)
        res_iqr = detect_iqr_anomalies(empty_df)

        assert res_z.total_rows == 0
        assert res_z.anomaly_count == 0
        assert res_iqr.total_rows == 0
        assert res_iqr.anomaly_count == 0
