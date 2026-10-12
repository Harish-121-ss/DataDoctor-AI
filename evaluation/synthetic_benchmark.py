"""
DataDoctor AI - Deterministic Synthetic Benchmark Generator
Generates reproducible synthetic datasets with documented ground-truth anomalies.
Guarantees zero data leakage by keeping ground-truth labels completely separate from feature data.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def generate_synthetic_anomaly_dataset(
    n_samples: int = 1000,
    contamination: float = 0.10,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Generate a deterministic synthetic dataset containing normal observations
    and deliberately injected anomalies across three distinct defect modes.

    Anomalies generated:
    1. Univariate Extreme: Values far outside normal variance (5-7 stds).
    2. Multivariate Correlation Violations: Values within univariate range, but
       violating the strong joint covariance structure between correlated features.
    3. Sparse Cluster Outliers: A small, isolated grouping in low-density feature space.

    Parameters
    ----------
    n_samples : int, default 1000
        Total number of observations to generate.
    contamination : float, default 0.10
        Fraction of observations that are anomalies (100 out of 1000 by default).
    random_seed : int, default 42
        Seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]
        - features_df: DataFrame containing ONLY feature columns (no labels).
        - ground_truth_labels: pd.Series of booleans (True for anomaly, False for normal).
        - metadata: Dictionary detailing generation parameters and anomaly breakdown.
    """
    rng = np.random.default_rng(random_seed)

    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies

    # 1. Normal data generation: 3 correlated features
    # feat_alpha and feat_beta have strong linear correlation (r ~ 0.85)
    mean = np.array([50.0, 100.0, 30.0])
    cov = np.array([
        [25.0,  42.5,   5.0],   # std(alpha)=5, std(beta)=10, cov(alpha,beta)=42.5 (r=0.85)
        [42.5, 100.0,  10.0],
        [ 5.0,  10.0,  16.0],   # std(gamma)=4
    ])

    normal_data = rng.multivariate_normal(mean, cov, size=n_normal)

    # 2. Anomaly generation
    # Mode A: Univariate extreme outliers (~40% of anomalies)
    n_univariate = int(n_anomalies * 0.40)
    univariate_data = rng.multivariate_normal(mean, cov, size=n_univariate)
    # Inject massive shifts (6-8 std deviations)
    for i in range(n_univariate):
        col_to_spike = i % 3
        spike_direction = 1 if rng.random() > 0.5 else -1
        if col_to_spike == 0:
            univariate_data[i, 0] += spike_direction * 35.0  # 7 * 5
        elif col_to_spike == 1:
            univariate_data[i, 1] += spike_direction * 70.0  # 7 * 10
        else:
            univariate_data[i, 2] += spike_direction * 28.0  # 7 * 4

    # Mode B: Multivariate correlation violations (~40% of anomalies)
    # Both alpha and beta stay within standard min/max ranges, but break the correlation!
    n_multivariate = int(n_anomalies * 0.40)
    alpha_high = rng.uniform(58.0, 64.0, size=n_multivariate)    # High alpha
    beta_low = rng.uniform(70.0, 78.0, size=n_multivariate)       # But LOW beta! (Normal would be > 115)
    gamma_normal = rng.normal(30.0, 4.0, size=n_multivariate)
    multivariate_data = np.column_stack([alpha_high, beta_low, gamma_normal])

    # Mode C: Isolated cluster outliers (remaining anomalies)
    n_cluster = n_anomalies - n_univariate - n_multivariate
    cluster_mean = np.array([20.0, 160.0, 60.0])  # Far detached from main cluster
    cluster_cov = np.diag([4.0, 9.0, 4.0])
    cluster_data = rng.multivariate_normal(cluster_mean, cluster_cov, size=n_cluster)

    # 3. Combine and shuffle deterministically
    all_data = np.vstack([normal_data, univariate_data, multivariate_data, cluster_data])
    labels = np.array(
        [False] * n_normal
        + [True] * n_univariate
        + [True] * n_multivariate
        + [True] * n_cluster,
        dtype=bool,
    )

    # Deterministic permutation
    perm = rng.permutation(n_samples)
    shuffled_data = all_data[perm]
    shuffled_labels = labels[perm]

    features_df = pd.DataFrame(
        shuffled_data,
        columns=["feat_alpha", "feat_beta", "feat_gamma"],
    )
    ground_truth_series = pd.Series(shuffled_labels, name="is_ground_truth_anomaly")

    metadata = {
        "total_samples": n_samples,
        "normal_samples": n_normal,
        "total_anomalies": n_anomalies,
        "contamination_rate": contamination,
        "random_seed": random_seed,
        "anomaly_types": {
            "univariate_extreme": n_univariate,
            "multivariate_correlation_violation": n_multivariate,
            "sparse_cluster_outliers": n_cluster,
        },
        "features": list(features_df.columns),
    }

    return features_df, ground_truth_series, metadata
