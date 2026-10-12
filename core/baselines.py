"""
DataDoctor AI - Statistical Baseline Anomaly Detectors
Implements univariate Z-score and Tukey's IQR detection methods.
Independent of machine learning models, providing transparent benchmarks.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class AnomalyDetectionResult:
    """
    Standardized, documented result returned by all anomaly detectors.
    """
    method: str
    total_rows: int
    evaluated_rows: int
    anomaly_count: int
    anomaly_percentage: float
    features_used: List[str]
    excluded_features: List[str]
    anomaly_indices: List[int]
    predictions: List[bool]  # True for anomaly, False for normal
    scores: List[float]       # Continuous anomaly score (higher = more anomalous)
    feature_flags: Dict[str, List[bool]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_usable_numerical_features(
    df: pd.DataFrame,
    specified_columns: Optional[Sequence[str]] = None,
    exclude_identifiers: bool = True,
) -> tuple[List[str], List[str]]:
    """
    Select usable continuous numerical columns, filtering non-numeric columns,
    zero-variance constants, and potential identifier columns.

    Returns
    -------
    tuple[List[str], List[str]]
        (usable_features, excluded_features)
    """
    candidate_cols = list(specified_columns) if specified_columns is not None else list(df.columns)
    usable: List[str] = []
    excluded: List[str] = []

    for col in candidate_cols:
        if col not in df.columns:
            excluded.append(f"{col} (not in dataframe)")
            continue

        series = df[col]

        # 1. Must be numeric or coercible to numeric
        if not pd.api.types.is_numeric_dtype(series):
            # Check if coercible
            non_null = series.dropna()
            if len(non_null) > 0 and pd.to_numeric(non_null, errors="coerce").notna().mean() >= 0.95:
                # Could be coerced, but for raw detection safety, we only use clean numeric
                pass
            excluded.append(f"{col} (non-numeric dtype: {series.dtype})")
            continue

        # 2. Exclude identifiers if requested
        if exclude_identifiers:
            col_lower = str(col).lower().replace("_", " ").replace("-", " ")
            id_keywords = ["id", "uuid", "guid", "key", "pk", "fk", "token", "hash"]
            words = col_lower.split()
            if any(kw in words for kw in id_keywords) or col_lower.endswith(" id"):
                excluded.append(f"{col} (detected identifier)")
                continue

            # Sequential integer index check only when column name suggests an index/sequence
            index_hints = ["index", "idx", "row", "seq", "sequence"]
            if any(hint in words for hint in index_hints):
                non_null = series.dropna()
                if len(non_null) > 10 and non_null.nunique() == len(non_null):
                    diffs = np.diff(np.sort(non_null.values))
                    if np.all(diffs == 1):
                        excluded.append(f"{col} (sequential unique integer index)")
                        continue

        # 3. Exclude zero variance / all null
        non_null_finite = series.dropna()
        non_null_finite = non_null_finite[np.isfinite(non_null_finite)]
        if len(non_null_finite) < 1:
            excluded.append(f"{col} (no non-null finite values)")
            continue

        if len(non_null_finite) > 1 and non_null_finite.nunique() <= 1:
            excluded.append(f"{col} (constant zero variance)")
            continue

        usable.append(str(col))

    return usable, excluded


def detect_zscore_anomalies(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    threshold: float = 3.0,
    combination_rule: Literal["any", "all", "majority"] = "any",
    exclude_identifiers: bool = True,
) -> AnomalyDetectionResult:
    """
    Detect anomalies using the Gaussian Z-score method.

    Assumptions:
    - Features follow an approximately normal (Gaussian) distribution.
    - Outliers lie beyond `threshold` standard deviations from the feature mean (|z| > threshold).
    - Features are evaluated independently and combined using `combination_rule`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (remains unmodified).
    columns : sequence of str, optional
        Numerical columns to evaluate. If None, auto-selects valid numeric features.
    threshold : float, default 3.0
        Number of standard deviations beyond which a value is considered an outlier.
    combination_rule : {'any', 'all', 'majority'}, default 'any'
        Logic to combine multi-column outlier flags into a row-level flag:
        - 'any': row is an anomaly if at least 1 feature exceeds threshold.
        - 'all': row is an anomaly only if all features exceed threshold.
        - 'majority': row is an anomaly if > 50% of features exceed threshold.
    exclude_identifiers : bool, default True
        Automatically exclude ID/key columns.
    """
    start_time = time.perf_counter()
    total_rows = len(df)

    if total_rows == 0:
        return AnomalyDetectionResult(
            method="z_score",
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

    usable_cols, excluded_cols = select_usable_numerical_features(
        df, specified_columns=columns, exclude_identifiers=exclude_identifiers
    )

    if not usable_cols:
        return AnomalyDetectionResult(
            method="z_score",
            total_rows=total_rows,
            evaluated_rows=0,
            anomaly_count=0,
            anomaly_percentage=0.0,
            features_used=[],
            excluded_features=excluded_cols,
            anomaly_indices=[],
            predictions=[False] * total_rows,
            scores=[0.0] * total_rows,
            explanation="No usable numerical features found for Z-score anomaly detection.",
        )

    # Eligible rows: must have non-null, finite values across all evaluated columns
    valid_mask = pd.Series(True, index=df.index)
    for col in usable_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid_mask &= (s.notna() & np.isfinite(s))

    evaluated_indices = df.index[valid_mask].tolist()
    evaluated_count = len(evaluated_indices)

    if evaluated_count == 0:
        return AnomalyDetectionResult(
            method="z_score",
            total_rows=total_rows,
            evaluated_rows=0,
            anomaly_count=0,
            anomaly_percentage=0.0,
            features_used=usable_cols,
            excluded_features=excluded_cols,
            anomaly_indices=[],
            predictions=[False] * total_rows,
            scores=[0.0] * total_rows,
            explanation="Zero rows have complete, finite numerical data across evaluated features.",
        )

    feature_flags: Dict[str, List[bool]] = {}
    z_scores_dict: Dict[str, np.ndarray] = {}

    sub_df = df.loc[valid_mask, usable_cols]

    for col in usable_cols:
        vals = sub_df[col].to_numpy(dtype=float)
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0

        if std_val == 0.0:
            z_scores = np.zeros_like(vals)
            flags = np.zeros_like(vals, dtype=bool)
        else:
            z_scores = np.abs((vals - mean_val) / std_val)
            flags = z_scores > threshold

        z_scores_dict[col] = z_scores

        # Map to full dataframe length
        full_flags = [False] * total_rows
        for row_pos, idx in enumerate(evaluated_indices):
            full_flags[idx] = bool(flags[row_pos])
        feature_flags[col] = full_flags

    # Combine flags and calculate row-level scores
    flags_matrix = np.column_stack([z_scores_dict[col] > threshold for col in usable_cols])
    scores_matrix = np.column_stack([z_scores_dict[col] for col in usable_cols])

    if combination_rule == "all":
        row_anomalies_evaluated = np.all(flags_matrix, axis=1)
    elif combination_rule == "majority":
        row_anomalies_evaluated = (np.sum(flags_matrix, axis=1) / len(usable_cols)) > 0.5
    else:  # "any"
        row_anomalies_evaluated = np.any(flags_matrix, axis=1)

    max_z_scores_evaluated = np.max(scores_matrix, axis=1)

    full_predictions = [False] * total_rows
    full_scores = [0.0] * total_rows
    anomaly_indices: List[int] = []

    for row_pos, idx in enumerate(evaluated_indices):
        is_anom = bool(row_anomalies_evaluated[row_pos])
        score = float(max_z_scores_evaluated[row_pos])
        full_predictions[idx] = is_anom
        full_scores[idx] = score
        if is_anom:
            anomaly_indices.append(idx)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    anom_count = len(anomaly_indices)
    anom_pct = round(anom_count / evaluated_count * 100.0, 2) if evaluated_count > 0 else 0.0

    return AnomalyDetectionResult(
        method="z_score",
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
            "threshold": threshold,
            "combination_rule": combination_rule,
            "skipped_rows_missing_finite": total_rows - evaluated_count,
        },
        execution_time_ms=round(elapsed_ms, 2),
        explanation=(
            f"Z-Score detected {anom_count} anomalies ({anom_pct}% of {evaluated_count} evaluated rows) "
            f"exceeding |Z| > {threshold} across features {usable_cols} (rule: '{combination_rule}'). "
            f"Note: An anomaly indicates statistically unusual values, not confirmed error."
        ),
    )


def detect_iqr_anomalies(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    k: float = 1.5,
    combination_rule: Literal["any", "all", "majority"] = "any",
    exclude_identifiers: bool = True,
) -> AnomalyDetectionResult:
    """
    Detect anomalies using Tukey's Interquartile Range (IQR) fencing method.

    Assumptions:
    - Non-parametric: does NOT assume a normal distribution.
    - Outliers lie beyond [Q1 - k*IQR, Q3 + k*IQR].
    - Robust against extreme skewness and heavy tails.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (remains unmodified).
    columns : sequence of str, optional
        Numerical columns to evaluate.
    k : float, default 1.5
        Multiplier for the IQR fence (1.5 for standard outliers, 3.0 for extreme outliers).
    combination_rule : {'any', 'all', 'majority'}, default 'any'
        Logic to combine multi-column outlier flags.
    exclude_identifiers : bool, default True
        Automatically exclude ID/key columns.
    """
    start_time = time.perf_counter()
    total_rows = len(df)

    if total_rows == 0:
        return AnomalyDetectionResult(
            method="iqr",
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

    usable_cols, excluded_cols = select_usable_numerical_features(
        df, specified_columns=columns, exclude_identifiers=exclude_identifiers
    )

    if not usable_cols:
        return AnomalyDetectionResult(
            method="iqr",
            total_rows=total_rows,
            evaluated_rows=0,
            anomaly_count=0,
            anomaly_percentage=0.0,
            features_used=[],
            excluded_features=excluded_cols,
            anomaly_indices=[],
            predictions=[False] * total_rows,
            scores=[0.0] * total_rows,
            explanation="No usable numerical features found for IQR anomaly detection.",
        )

    # Eligible rows: must have non-null, finite values across all evaluated columns
    valid_mask = pd.Series(True, index=df.index)
    for col in usable_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid_mask &= (s.notna() & np.isfinite(s))

    evaluated_indices = df.index[valid_mask].tolist()
    evaluated_count = len(evaluated_indices)

    if evaluated_count == 0:
        return AnomalyDetectionResult(
            method="iqr",
            total_rows=total_rows,
            evaluated_rows=0,
            anomaly_count=0,
            anomaly_percentage=0.0,
            features_used=usable_cols,
            excluded_features=excluded_cols,
            anomaly_indices=[],
            predictions=[False] * total_rows,
            scores=[0.0] * total_rows,
            explanation="Zero rows have complete, finite numerical data across evaluated features.",
        )

    feature_flags: Dict[str, List[bool]] = {}
    iqr_distance_dict: Dict[str, np.ndarray] = {}

    sub_df = df.loc[valid_mask, usable_cols]

    for col in usable_cols:
        vals = sub_df[col].to_numpy(dtype=float)
        q25 = np.percentile(vals, 25)
        q75 = np.percentile(vals, 75)
        iqr = q75 - q25

        if iqr == 0.0:
            # When IQR is 0, points different from median can be checked, or 0 distance
            distances = np.zeros_like(vals)
            flags = np.zeros_like(vals, dtype=bool)
        else:
            lower_fence = q25 - (k * iqr)
            upper_fence = q75 + (k * iqr)

            flags = (vals < lower_fence) | (vals > upper_fence)
            # Distance beyond fence normalized by IQR
            dist_below = np.maximum(0, (lower_fence - vals) / iqr)
            dist_above = np.maximum(0, (vals - upper_fence) / iqr)
            distances = dist_below + dist_above

        iqr_distance_dict[col] = distances

        full_flags = [False] * total_rows
        for row_pos, idx in enumerate(evaluated_indices):
            full_flags[idx] = bool(flags[row_pos])
        feature_flags[col] = full_flags

    flags_matrix = np.column_stack([iqr_distance_dict[col] > 0 for col in usable_cols])
    scores_matrix = np.column_stack([iqr_distance_dict[col] for col in usable_cols])

    if combination_rule == "all":
        row_anomalies_evaluated = np.all(flags_matrix, axis=1)
    elif combination_rule == "majority":
        row_anomalies_evaluated = (np.sum(flags_matrix, axis=1) / len(usable_cols)) > 0.5
    else:  # "any"
        row_anomalies_evaluated = np.any(flags_matrix, axis=1)

    max_distances_evaluated = np.max(scores_matrix, axis=1)

    full_predictions = [False] * total_rows
    full_scores = [0.0] * total_rows
    anomaly_indices: List[int] = []

    for row_pos, idx in enumerate(evaluated_indices):
        is_anom = bool(row_anomalies_evaluated[row_pos])
        score = float(max_distances_evaluated[row_pos])
        full_predictions[idx] = is_anom
        full_scores[idx] = score
        if is_anom:
            anomaly_indices.append(idx)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    anom_count = len(anomaly_indices)
    anom_pct = round(anom_count / evaluated_count * 100.0, 2) if evaluated_count > 0 else 0.0

    return AnomalyDetectionResult(
        method="iqr",
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
            "k_multiplier": k,
            "combination_rule": combination_rule,
            "skipped_rows_missing_finite": total_rows - evaluated_count,
        },
        execution_time_ms=round(elapsed_ms, 2),
        explanation=(
            f"IQR Fencing (k={k}) detected {anom_count} anomalies ({anom_pct}% of {evaluated_count} evaluated rows) "
            f"across features {usable_cols} (rule: '{combination_rule}'). "
            f"Note: An anomaly indicates statistically unusual values, not confirmed error."
        ),
    )
