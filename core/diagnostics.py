"""
DataDoctor AI - Rule-Based Diagnostics Engine
Evaluates dataset profiles to detect concrete data quality defects,
inconsistencies, and suspicious values, producing actionable repair recommendations.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.profiler import ColumnProfile, DatasetProfile, profile_dataset

# Domain terms where negative values are suspicious
NON_NEGATIVE_NAME_PATTERNS = [
    r"\bage\b",
    r"\bcount\b",
    r"\bquantity\b",
    r"\bprice\b",
    r"\brevenue\b",
    r"\bsalary\b",
    r"\bincome\b",
    r"\bcost\b",
    r"\bheight\b",
    r"\bweight\b",
    r"\bdistance\b",
    r"\bduration\b",
    r"\bspeed\b",
    r"\btax\b",
    r"\bamount\b",
]


@dataclass
class DiagnosticIssue:
    """Represents a single detected data quality issue."""
    issue_type: str
    severity: str  # "high", "medium", "low"
    column: Optional[str]
    affected_count: int
    affected_percentage: float
    explanation: str
    suggested_action: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_name_typically_non_negative(column_name: str) -> bool:
    """Check if column name suggests naturally non-negative values."""
    col_clean = column_name.lower().replace("_", " ").replace("-", " ")
    for pattern in NON_NEGATIVE_NAME_PATTERNS:
        if re.search(pattern, col_clean):
            return True
    return False


def detect_type_inconsistencies(series: pd.Series, col_prof: ColumnProfile) -> Optional[DiagnosticIssue]:
    """
    Detect columns stored as object/string that are predominantly numeric,
    identifying the offending non-numeric tokens.
    """
    is_textual = (
        (series.dtype == object)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    )
    if not is_textual:
        return None

    non_null = series.dropna()
    if len(non_null) == 0:
        return None

    # Try numeric conversion
    coerced = pd.to_numeric(non_null, errors="coerce")
    valid_numeric_count = int(coerced.notna().sum())
    non_numeric_count = int(coerced.isna().sum())

    # If at least 60% of values are numeric, but there are non-numeric strings
    if non_numeric_count > 0 and (valid_numeric_count / len(non_null)) >= 0.60:
        # Find sample offending values
        offending_series = non_null[coerced.isna()]
        sample_invalids = offending_series.drop_duplicates().head(5).tolist()

        return DiagnosticIssue(
            issue_type="TYPE_INCONSISTENCY",
            severity="high",
            column=col_prof.name,
            affected_count=non_numeric_count,
            affected_percentage=round(non_numeric_count / len(non_null) * 100.0, 2),
            explanation=(
                f"Column '{col_prof.name}' is stored as text/object, but {valid_numeric_count} "
                f"({valid_numeric_count / len(non_null) * 100:.1f}%) values are numeric. "
                f"Found {non_numeric_count} non-numeric tokens such as: {sample_invalids}."
            ),
            suggested_action=(
                "Clean stray characters (currency signs, commas, text placeholders) "
                "and cast column to float or int."
            ),
            details={"sample_invalid_values": sample_invalids},
        )

    return None


def detect_suspicious_values(series: pd.Series, col_prof: ColumnProfile) -> List[DiagnosticIssue]:
    """
    Detect suspicious numerical values such as negative values in naturally positive domains,
    infinite numbers, or extreme deviations (>5 IQR).
    """
    issues: List[DiagnosticIssue] = []
    if col_prof.inferred_type != "numeric" or col_prof.non_null_count == 0:
        return issues

    num_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(num_series) == 0:
        return issues

    # 1. Infinite values
    inf_count = int(np.isinf(num_series).sum())
    if inf_count > 0:
        issues.append(
            DiagnosticIssue(
                issue_type="SUSPICIOUS_INFINITE_VALUE",
                severity="high",
                column=col_prof.name,
                affected_count=inf_count,
                affected_percentage=round(inf_count / len(num_series) * 100.0, 2),
                explanation=f"Column '{col_prof.name}' contains {inf_count} infinite (inf / -inf) values.",
                suggested_action="Replace infinite values with NaN for imputation or clip to column finite bounds.",
                details={"infinite_count": inf_count},
            )
        )

    # 2. Negative values in naturally positive domains
    if is_name_typically_non_negative(col_prof.name):
        neg_count = int((num_series < 0).sum())
        if neg_count > 0:
            min_val = float(num_series.min())
            issues.append(
                DiagnosticIssue(
                    issue_type="SUSPICIOUS_NEGATIVE_VALUE",
                    severity="medium",
                    column=col_prof.name,
                    affected_count=neg_count,
                    affected_percentage=round(neg_count / len(num_series) * 100.0, 2),
                    explanation=(
                        f"Column '{col_prof.name}' represents a naturally non-negative domain "
                        f"but contains {neg_count} negative entries (minimum value: {min_val})."
                    ),
                    suggested_action="Review domain logic: replace negative values with absolute values, NaN, or clip to zero.",
                    details={"min_value": min_val, "negative_count": neg_count},
                )
            )

    # 3. Extreme outliers (> 5 * IQR beyond quartiles)
    stats = col_prof.stats
    if stats and "iqr" in stats and stats["iqr"] > 0:
        q25 = stats["q25"]
        q75 = stats["q75"]
        iqr = stats["iqr"]
        lower_bound = q25 - (5.0 * iqr)
        upper_bound = q75 + (5.0 * iqr)

        extreme_mask = (num_series < lower_bound) | (num_series > upper_bound)
        extreme_count = int(extreme_mask.sum())

        if extreme_count > 0:
            sample_extremes = num_series[extreme_mask].head(5).tolist()
            issues.append(
                DiagnosticIssue(
                    issue_type="SUSPICIOUS_EXTREME_OUTLIER",
                    severity="low",
                    column=col_prof.name,
                    affected_count=extreme_count,
                    affected_percentage=round(extreme_count / len(num_series) * 100.0, 2),
                    explanation=(
                        f"Column '{col_prof.name}' contains {extreme_count} extreme statistical outliers "
                        f"exceeding 5x IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]."
                    ),
                    suggested_action="Inspect values using anomaly detection view; consider winsorization or capping.",
                    details={
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "sample_outliers": sample_extremes,
                    },
                )
            )

    return issues


def diagnose_dataset(
    df: pd.DataFrame,
    profile: Optional[DatasetProfile] = None,
) -> List[DiagnosticIssue]:
    """
    Run deterministic diagnostic rules over the dataset and its profile.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    profile : DatasetProfile, optional
        Precomputed dataset profile. If not provided, will be computed.

    Returns
    -------
    List[DiagnosticIssue]
        List of all detected issues sorted by severity (high -> medium -> low).
    """
    if profile is None:
        profile = profile_dataset(df)

    issues: List[DiagnosticIssue] = []

    # 1. Dataset-level: Duplicate rows
    if profile.duplicate_rows > 0:
        severity = "high" if profile.duplicate_percentage >= 10.0 else "medium"
        issues.append(
            DiagnosticIssue(
                issue_type="DUPLICATE_ROWS",
                severity=severity,
                column=None,
                affected_count=profile.duplicate_rows,
                affected_percentage=profile.duplicate_percentage,
                explanation=(
                    f"Dataset contains {profile.duplicate_rows} duplicate rows "
                    f"({profile.duplicate_percentage}% of total dataset)."
                ),
                suggested_action="Deduplicate rows keeping the first occurrence.",
                details={"duplicate_rows": profile.duplicate_rows},
            )
        )

    # 2. Column-level diagnostics
    for col_name, col_prof in profile.columns.items():
        # A. All-missing columns
        if col_prof.is_all_missing:
            issues.append(
                DiagnosticIssue(
                    issue_type="ALL_MISSING_COLUMN",
                    severity="high",
                    column=col_name,
                    affected_count=col_prof.missing_count,
                    affected_percentage=100.0,
                    explanation=f"Column '{col_name}' is completely empty (100% missing values).",
                    suggested_action="Drop this column as it contains no useful information.",
                )
            )
            continue  # No need for further checks on 100% null column

        # B. Partial missing values
        if col_prof.missing_count > 0:
            if col_prof.missing_percentage >= 50.0:
                issues.append(
                    DiagnosticIssue(
                        issue_type="HIGH_MISSING_VALUES",
                        severity="high",
                        column=col_name,
                        affected_count=col_prof.missing_count,
                        affected_percentage=col_prof.missing_percentage,
                        explanation=(
                            f"Column '{col_name}' has {col_prof.missing_count} missing entries "
                            f"({col_prof.missing_percentage}%)."
                        ),
                        suggested_action="Evaluate column relevance; consider dropping column or using median/mode imputation.",
                    )
                )
            else:
                issues.append(
                    DiagnosticIssue(
                        issue_type="MISSING_VALUES",
                        severity="low",
                        column=col_name,
                        affected_count=col_prof.missing_count,
                        affected_percentage=col_prof.missing_percentage,
                        explanation=(
                            f"Column '{col_name}' has {col_prof.missing_count} missing entries "
                            f"({col_prof.missing_percentage}%)."
                        ),
                        suggested_action="Impute with median/mean for numeric or mode for categorical, or drop affected rows.",
                    )
                )

        # C. Constant columns (zero variance)
        if col_prof.is_constant:
            issues.append(
                DiagnosticIssue(
                    issue_type="CONSTANT_COLUMN",
                    severity="medium",
                    column=col_name,
                    affected_count=col_prof.total_rows,
                    affected_percentage=100.0,
                    explanation=(
                        f"Column '{col_name}' has only 1 distinct non-null value across all rows "
                        f"(zero variance)."
                    ),
                    suggested_action="Drop this constant column as it provides no predictive variance.",
                )
            )

        # D. Type inconsistencies (e.g. numeric disguised as object)
        if col_name in df.columns:
            type_issue = detect_type_inconsistencies(df[col_name], col_prof)
            if type_issue:
                issues.append(type_issue)

        # E. Suspicious values (negative in positive domains, inf, extreme outliers)
        if col_name in df.columns:
            susp_issues = detect_suspicious_values(df[col_name], col_prof)
            issues.extend(susp_issues)

        # F. High-cardinality string identifiers
        if (
            col_prof.inferred_type in ("text", "categorical")
            and col_prof.total_rows >= 20
            and col_prof.distinct_percentage >= 95.0
        ):
            issues.append(
                DiagnosticIssue(
                    issue_type="HIGH_CARDINALITY_IDENTIFIER",
                    severity="low",
                    column=col_name,
                    affected_count=col_prof.distinct_count,
                    affected_percentage=col_prof.distinct_percentage,
                    explanation=(
                        f"Column '{col_name}' has {col_prof.distinct_count} unique values "
                        f"({col_prof.distinct_percentage}% unique). Likely an ID, key, or free text."
                    ),
                    suggested_action="Verify if this column is an identifier/hash; exclude from ML feature sets.",
                )
            )

    # Sort issues by severity: high first, then medium, then low
    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: severity_order.get(x.severity, 3))
    return issues


def calculate_health_score(profile: DatasetProfile, issues: List[DiagnosticIssue]) -> float:
    """
    Calculate an honest dataset health score from 0.0 to 100.0.
    Score starts at 100 and applies calibrated penalties based on defect severity.
    """
    if profile.total_rows == 0 or profile.total_columns == 0:
        return 0.0

    score = 100.0

    # Penalty for missing values overall (up to -30)
    score -= min(profile.total_missing_percentage * 0.6, 30.0)

    # Penalty for duplicate rows (up to -20)
    score -= min(profile.duplicate_percentage * 0.5, 20.0)

    # Penalty per issue severity
    for issue in issues:
        if issue.severity == "high":
            score -= 5.0
        elif issue.severity == "medium":
            score -= 2.5
        elif issue.severity == "low":
            score -= 1.0

    return max(round(score, 1), 0.0)


def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an integrated, serializable dataset quality report.
    """
    profile = profile_dataset(df)
    issues = diagnose_dataset(df, profile=profile)
    health_score = calculate_health_score(profile, issues)

    return {
        "health_score": health_score,
        "total_issues": len(issues),
        "issues_by_severity": {
            "high": sum(1 for i in issues if i.severity == "high"),
            "medium": sum(1 for i in issues if i.severity == "medium"),
            "low": sum(1 for i in issues if i.severity == "low"),
        },
        "profile": profile.to_dict(),
        "issues": [issue.to_dict() for issue in issues],
    }
