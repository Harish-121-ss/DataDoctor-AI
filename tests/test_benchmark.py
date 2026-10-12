"""
Unit tests for evaluation suite (Synthetic Benchmark & Benchmark Runner).
Verifies deterministic data generation, label separation, metric formulas,
zero-division safety, and end-to-end benchmark execution.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.benchmark_runner import calculate_metrics, run_anomaly_benchmark
from evaluation.synthetic_benchmark import generate_synthetic_anomaly_dataset


class TestEvaluationBenchmark:
    """Test suite for synthetic benchmark and evaluation runner."""

    def test_synthetic_data_generation_properties(self):
        features_df, ground_truth, meta = generate_synthetic_anomaly_dataset(
            n_samples=500, contamination=0.10, random_seed=42
        )

        assert len(features_df) == 500
        assert len(ground_truth) == 500
        assert meta["total_samples"] == 500
        assert meta["total_anomalies"] == 50
        assert ground_truth.sum() == 50

        # DATA LEAKAGE PREVENTION: Label must NOT be in the features DataFrame
        assert "is_ground_truth_anomaly" not in features_df.columns
        assert list(features_df.columns) == ["feat_alpha", "feat_beta", "feat_gamma"]

    def test_deterministic_generation_reproducibility(self):
        f1, g1, _ = generate_synthetic_anomaly_dataset(n_samples=200, random_seed=123)
        f2, g2, _ = generate_synthetic_anomaly_dataset(n_samples=200, random_seed=123)

        pd.testing.assert_frame_equal(f1, f2)
        pd.testing.assert_series_equal(g1, g2)

    def test_metric_calculations_standard_case(self):
        # 10 samples: 3 true anomalies, detector flags 3 items (2 TP, 1 FP)
        y_true = [True, True, True, False, False, False, False, False, False, False]
        y_pred = [True, True, False, True, False, False, False, False, False, False]

        m = calculate_metrics(y_true, y_pred, "TestDetector", execution_time_ms=10.0)

        assert m.true_positives == 2
        assert m.false_positives == 1
        assert m.false_negatives == 1
        assert m.true_negatives == 6
        assert m.precision == round(2 / 3, 4)
        assert m.recall == round(2 / 3, 4)
        assert m.f1_score == round(2 / 3, 4)
        assert m.specificity == round(6 / 7, 4)

    def test_metric_calculations_zero_division_cases(self):
        # Case A: Model flags nothing (TP=0, FP=0) -> Precision and Recall should be 0.0, no ZeroDivisionError
        y_true = [True, True, False, False]
        y_pred_none = [False, False, False, False]

        m_none = calculate_metrics(y_true, y_pred_none, "FlagsNothing", execution_time_ms=1.0)
        assert m_none.true_positives == 0
        assert m_none.false_positives == 0
        assert m_none.precision == 0.0
        assert m_none.recall == 0.0
        assert m_none.f1_score == 0.0
        assert m_none.specificity == 1.0

        # Case B: Ground truth has zero anomalies (TP=0, FN=0)
        y_true_none = [False, False, False, False]
        y_pred = [True, False, False, False]

        m_no_truth = calculate_metrics(y_true_none, y_pred, "NoTruthAnomalies", execution_time_ms=1.0)
        assert m_no_truth.recall == 0.0
        assert m_no_truth.f1_score == 0.0

    def test_end_to_end_benchmark_run(self):
        # Run small benchmark
        results = run_anomaly_benchmark(n_samples=200, contamination=0.10, random_seed=42)

        assert "dataset_metadata" in results
        assert "metrics" in results
        assert "raw_results" in results

        metrics = results["metrics"]
        assert "isolation_forest" in metrics
        assert "z_score" in metrics
        assert "iqr" in metrics

        # Verify all metrics exist and are within [0.0, 1.0]
        for name, m in metrics.items():
            assert 0.0 <= m["precision"] <= 1.0
            assert 0.0 <= m["recall"] <= 1.0
            assert 0.0 <= m["f1_score"] <= 1.0
            assert 0.0 <= m["specificity"] <= 1.0
            assert m["execution_time_ms"] > 0.0
