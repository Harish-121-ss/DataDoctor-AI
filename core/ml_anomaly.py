"""
DataDoctor AI - Machine Learning Anomaly Detection (Isolation Forest)
Implements unsupervised multidimensional anomaly detection using Scikit-learn.
Operates immutably, defends against degenerate data, and explains flagged records.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from core.baselines import AnomalyDetectionResult, select_usable_numerical_features


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    contamination: Union[float, str] = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
    exclude_identifiers: bool = True,
) -> AnomalyDetectionResult:
    """
    Detect multivariate anomalies using Scikit-learn's Isolation Forest algorithm.

    Isolation Forest isolates observations by randomly selecting a feature and
    then randomly selecting a split value between the minimum and maximum values of the selected feature.
    Anomalies have significantly shorter average path lengths in the isolation trees.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (guaranteed not to be mutated).
    columns : sequence of str, optional
        Numerical columns to evaluate. If None, auto-selects valid numerical features.
    contamination : float or 'auto', default 0.05
        The expected proportion of outliers in the data set (must be in (0.0, 0.5] if float).
    n_estimators : int, default 100
        Number of isolation trees in the ensemble.
    random_state : int, default 42
        Random seed for reproducible results.
    exclude_identifiers : bool, default True
        Automatically exclude ID/key columns from the feature matrix.

    Returns
    -------
    AnomalyDetectionResult
        Standardized detection results including indices, scores, and explanations.

    Raises
    ------
    ValueError
        If contamination is not 'auto' and not between 0.0 and 0.5.
    """
    start_time = time.perf_counter()
    total_rows = len(df)

    # Validate contamination parameter
    if isinstance(contamination, (int, float)):
        if not (0.0 < float(contamination) <= 0.5):
            raise ValueError(
                f"Contamination must be in range (0.0, 0.5] or 'auto', got {contamination}."
            )
    elif contamination != "auto":
        raise ValueError(
            f"Contamination must be a float in (0.0, 0.5] or 'auto', got '{contamination}'."
        )

    if total_rows == 0:
        return AnomalyDetectionResult(
            method="isolation_forest",
            total_rows=0,
            evaluated_rows=0,
            anomaly_count=0,
            anomaly_percentage=0.0,
            features_used=[],
            excluded_features=[],
            anomaly_indices=[],
            predictions=[],
            scores=[],
            explanation="Dataset is empty (0 rows).",
        )

    # 1. Feature selection & validation
    usable_cols, excluded_cols = select_usable_numerical_features(
        df, specified_columns=columns, exclude_identifiers=exclude_identifiers
    )

    if not usable_cols:
        return AnomalyDetectionResult(
            method="isolation_forest",
            total_rows=total_rows,
            evaluated_rows=0,
            anomaly_count=0,
            anomaly_percentage=0.0,
            features_used=[],
            excluded_features=excluded_cols,
            anomaly_indices=[],
            predictions=[False] * total_rows,
            scores=[0.0] * total_rows,
            explanation="No usable numerical features found for Isolation Forest anomaly detection.",
        )

    # 2. Identify complete & finite cases for training and inference
    valid_mask = pd.Series(True, index=df.index)
    for col in usable_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid_mask &= (s.notna() & np.isfinite(s))

    evaluated_indices = df.index[valid_mask].tolist()
    evaluated_count = len(evaluated_indices)

    # Minimum sample threshold: Isolation Forest requires at least 2 samples
    if evaluated_count < 2:
        return AnomalyDetectionResult(
            method="isolation_forest",
            total_rows=total_rows,
            evaluated_rows=evaluated_count,
            anomaly_count=0,
            anomaly_percentage=0.0,
            features_used=usable_cols,
            excluded_features=excluded_cols,
            anomaly_indices=[],
            predictions=[False] * total_rows,
            scores=[0.0] * total_rows,
            explanation=(
                f"Insufficient eligible records ({evaluated_count} rows). "
                f"Isolation Forest requires at least 2 complete, finite numerical observations."
            ),
        )

    # 3. Prepare feature matrix (copy to ensure original DataFrame immutability)
    X_raw = df.loc[valid_mask, usable_cols].to_numpy(dtype=float)

    # Scale features for numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 4. Train Isolation Forest
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_scaled)

    # 5. Predict: in sklearn, -1 = anomaly, 1 = normal
    raw_preds = model.predict(X_scaled)
    is_anomaly_evaluated = (raw_preds == -1)

    # Raw decision function: lower = more anomalous
    raw_decision = model.decision_function(X_scaled)
    # Convert to normalized anomaly score where higher = more anomalous (0.0 to 1.0 approximately)
    # Offset is typically 0.0 under standard threshold
    scores_evaluated = np.round(-raw_decision, 4)

    # 6. Per-feature attribution for flagged anomalies
    # Measures normalized Z-deviation of each feature to understand why it was flagged
    means = np.mean(X_raw, axis=0)
    stds = np.std(X_raw, axis=0, ddof=1)
    stds[stds == 0.0] = 1.0  # Guard against zero division

    feature_flags: Dict[str, List[bool]] = {col: [False] * total_rows for col in usable_cols}

    for col_idx, col in enumerate(usable_cols):
        # Flag feature if its individual deviation > 2.0 std
        feat_dev = np.abs((X_raw[:, col_idx] - means[col_idx]) / stds[col_idx])
        col_flag_evaluated = (feat_dev > 2.0) & is_anomaly_evaluated
        for row_pos, idx in enumerate(evaluated_indices):
            if col_flag_evaluated[row_pos]:
                feature_flags[col][idx] = True

    # 7. Assemble full result aligned with original DataFrame rows
    full_predictions = [False] * total_rows
    full_scores = [0.0] * total_rows
    anomaly_indices: List[int] = []

    for row_pos, idx in enumerate(evaluated_indices):
        is_anom = bool(is_anomaly_evaluated[row_pos])
        score = float(scores_evaluated[row_pos])
        full_predictions[idx] = is_anom
        full_scores[idx] = score
        if is_anom:
            anomaly_indices.append(idx)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    anom_count = len(anomaly_indices)
    anom_pct = round(anom_count / evaluated_count * 100.0, 2) if evaluated_count > 0 else 0.0

    return AnomalyDetectionResult(
        method="isolation_forest",
        total_rows=total_rows,
        evaluated_rows=evaluated_count,
        anomaly_count=anom_count,
        anomaly_percentage=anom_pct,
        features_used=usable_cols,
        excluded_features=excluded_cols,
        anomaly_indices=anomaly_indices,
        predictions=full_predictions,
        scores=full_scores,
        feature_flags=feature_flags,
        metadata={
            "contamination": contamination,
            "n_estimators": n_estimators,
            "random_state": random_state,
            "skipped_rows_missing_finite": total_rows - evaluated_count,
        },
        execution_time_ms=round(elapsed_ms, 2),
        explanation=(
            f"Isolation Forest flagged {anom_count} multivariate anomalies "
            f"({anom_pct}% of {evaluated_count} evaluated rows) using features {usable_cols}. "
            f"Note: An anomaly represents an observation that is statistically unusual in the "
            f"feature space; it does not automatically imply incorrectness or fraud."
        ),
    )
