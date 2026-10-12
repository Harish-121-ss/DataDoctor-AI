"""
Unit tests for core.ml_anomaly (Isolation Forest Anomaly Detection).
Verifies reproducibility, anomaly detection accuracy, immutability,
edge-case handling (small data, missing/infinite/constant values), and parameter validation.
"""

import numpy as np
import pandas as pd
import pytest

from core.ml_anomaly import detect_isolation_forest_anomalies


class TestIsolationForestAnomalyDetection:
    """Test suite for Isolation Forest anomaly detector."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        normal = rng.normal(loc=100.0, scale=10.0, size=(100, 2))
        # Inject 5 extreme outliers
        outliers = np.array([
            [500.0, 500.0],
            [-200.0, -200.0],
            [600.0, 100.0],
            [100.0, 600.0],
            [-300.0, 300.0],
        ])
        all_points = np.vstack([normal, outliers])
        return pd.DataFrame(all_points, columns=["feature_x", "feature_y"])

    def test_reproducibility_with_fixed_seed(self, sample_data: pd.DataFrame):
        res1 = detect_isolation_forest_anomalies(
            sample_data, contamination=0.05, random_state=42
        )
        res2 = detect_isolation_forest_anomalies(
            sample_data, contamination=0.05, random_state=42
        )

        assert res1.anomaly_indices == res2.anomaly_indices
        assert res1.predictions == res2.predictions
        assert np.allclose(res1.scores, res2.scores, atol=1e-5)

    def test_detection_of_deliberate_anomalies(self, sample_data: pd.DataFrame):
        # 105 total rows, rows 100-104 are extreme outliers
        res = detect_isolation_forest_anomalies(
            sample_data, contamination=0.05, random_state=42
        )

        assert res.evaluated_rows == 105
        assert res.anomaly_count > 0

        # Outlier rows 100..104 should be flagged
        flagged_set = set(res.anomaly_indices)
        detected_known_outliers = sum(1 for idx in range(100, 105) if idx in flagged_set)
        assert detected_known_outliers >= 4, "Isolation Forest failed to detect deliberate outliers!"

    def test_original_dataframe_immutability(self, sample_data: pd.DataFrame):
        df_copy_before = sample_data.copy(deep=True)

        _ = detect_isolation_forest_anomalies(sample_data, contamination=0.05, random_state=42)

        pd.testing.assert_frame_equal(sample_data, df_copy_before)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        res = detect_isolation_forest_anomalies(empty_df)

        assert res.total_rows == 0
        assert res.evaluated_rows == 0
        assert res.anomaly_count == 0
        assert len(res.anomaly_indices) == 0

    def test_very_small_dataset_less_than_two_samples(self):
        tiny_df = pd.DataFrame({"val": [42.0]})
        res = detect_isolation_forest_anomalies(tiny_df)

        assert res.total_rows == 1
        assert res.evaluated_rows == 1
        assert res.anomaly_count == 0
        assert "Insufficient eligible records" in res.explanation

    def test_dataset_with_missing_and_infinite_values(self):
        df = pd.DataFrame({
            "feat_a": [10.0, 12.0, 11.0, np.nan, 13.0, np.inf, 100.0],
            "feat_b": [20.0, 22.0, 21.0, 23.0, 24.0, 25.0, 200.0],
        })

        res = detect_isolation_forest_anomalies(df, contamination=0.20, random_state=42)

        assert res.total_rows == 7
        # Rows with NaN or Inf (rows 3 and 5) should be skipped from training/inference
        assert res.evaluated_rows == 5
        assert res.predictions[3] is False
        assert res.predictions[5] is False

    def test_constant_column_and_id_exclusion(self):
        df = pd.DataFrame({
            "patient_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],  # Identifier
            "constant_col": [99.0] * 10,  # Zero variance
            "actual_metric": [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 10.2, 11.8, 12.2, 95.0],
        })

        res = detect_isolation_forest_anomalies(df, contamination=0.10, random_state=42)

        assert "actual_metric" in res.features_used
        assert not any("patient_id" == f for f in res.features_used)
        assert any("patient_id" in exc for exc in res.excluded_features)
        assert any("constant_col" in exc for exc in res.excluded_features)

    def test_invalid_contamination_parameters(self, sample_data: pd.DataFrame):
        # Contamination <= 0
        with pytest.raises(ValueError, match="Contamination"):
            detect_isolation_forest_anomalies(sample_data, contamination=-0.05)

        # Contamination > 0.5
        with pytest.raises(ValueError, match="Contamination"):
            detect_isolation_forest_anomalies(sample_data, contamination=0.75)

        # Invalid string
        with pytest.raises(ValueError, match="Contamination"):
            detect_isolation_forest_anomalies(sample_data, contamination="invalid_string")
