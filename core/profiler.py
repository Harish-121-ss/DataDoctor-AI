"""
DataDoctor AI - Data Quality Profiler
Computes detailed statistical and structural profiles for datasets and individual columns.
Handles empty datasets, all-null columns, and mixed data types defensively.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Detailed profile metrics for an individual column."""
    name: str
    pandas_dtype: str
    inferred_type: str
    total_rows: int
    non_null_count: int
    missing_count: int
    missing_percentage: float
    distinct_count: int
    distinct_percentage: float
    is_constant: bool
    is_all_missing: bool
    sample_values: List[Any] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetProfile:
    """Summary profile metrics for the entire dataset."""
    total_rows: int
    total_columns: int
    total_cells: int
    memory_bytes: int
    duplicate_rows: int
    duplicate_percentage: float
    total_missing_cells: int
    total_missing_percentage: float
    constant_columns: List[str]
    all_missing_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    boolean_columns: List[str]
    text_columns: List[str]
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["columns"] = {col: prof.to_dict() if hasattr(prof, "to_dict") else prof
                          for col, prof in self.columns.items()}
        return res


def infer_column_type(series: pd.Series) -> str:
    """
    Infer semantic data type of a Series without mutating underlying data.
    Returns one of: 'empty', 'boolean', 'numeric', 'datetime', 'categorical', 'text'.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"

    # 1. Native boolean or boolean-like objects
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    is_textual = (
        (series.dtype == object)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    )

    if is_textual:
        # Check if values are boolean-like strings
        unique_lower = {str(x).strip().lower() for x in non_null.unique()[:20]}
        if unique_lower.issubset({"true", "false", "yes", "no", "t", "f", "1", "0", "y", "n"}):
            return "boolean"

    # 2. Native numeric dtypes
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # 3. Check if text/object series is disguised numeric
    if is_textual:
        # Sample non-null values to test numeric coercion
        sample = non_null.head(100)
        coerced = pd.to_numeric(sample, errors="coerce")
        # If >= 90% of non-null sample can be converted to numbers
        if coerced.notna().mean() >= 0.90:
            return "numeric"

    # 4. Native datetime dtypes
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # 5. Check if text/object series is parseable datetime
    if is_textual:
        sample = non_null.head(50)
        # Avoid checking single numbers or short words as datetime
        has_date_delimiters = any("-" in str(x) or "/" in str(x) for x in sample.head(5))
        if has_date_delimiters:
            try:
                coerced_date = pd.to_datetime(sample, errors="coerce")
                if coerced_date.notna().mean() >= 0.85:
                    return "datetime"
            except Exception:
                pass

    # 6. Categorical vs free-form text
    distinct_ratio = non_null.nunique() / max(len(non_null), 1)
    if distinct_ratio < 0.20 or non_null.nunique() <= 30:
        return "categorical"

    return "text"


def profile_column(series: pd.Series, total_rows: int) -> ColumnProfile:
    """
    Compute comprehensive quality profile metrics for a single column.
    """
    col_name = str(series.name)
    pandas_dtype = str(series.dtype)

    missing_count = int(series.isna().sum())
    non_null_count = total_rows - missing_count
    missing_percentage = round((missing_count / total_rows * 100.0), 2) if total_rows > 0 else 0.0

    non_null_series = series.dropna()
    distinct_count = int(non_null_series.nunique())
    distinct_percentage = (
        round((distinct_count / non_null_count * 100.0), 2) if non_null_count > 0 else 0.0
    )

    is_all_missing = (non_null_count == 0)
    is_constant = (distinct_count == 1)

    inferred_type = infer_column_type(series)

    # Clean sample values for serialization
    raw_samples = non_null_series.drop_duplicates().head(5).tolist()
    sample_values = []
    for s in raw_samples:
        if isinstance(s, (np.integer, np.floating)):
            sample_values.append(s.item() if not np.isnan(s) else None)
        elif pd.isna(s):
            sample_values.append(None)
        else:
            sample_values.append(str(s))

    stats: Dict[str, Any] = {}

    if inferred_type == "numeric" and non_null_count > 0:
        # Coerce numeric values cleanly for statistics
        num_series = pd.to_numeric(non_null_series, errors="coerce").dropna()
        if len(num_series) > 0:
            finite_series = num_series[np.isfinite(num_series)]
            if len(finite_series) > 0:
                q25 = float(finite_series.quantile(0.25))
                q75 = float(finite_series.quantile(0.75))
                stats = {
                    "min": float(finite_series.min()),
                    "max": float(finite_series.max()),
                    "mean": round(float(finite_series.mean()), 4),
                    "std": round(float(finite_series.std()), 4) if len(finite_series) > 1 else 0.0,
                    "median": float(finite_series.median()),
                    "q25": q25,
                    "q75": q75,
                    "iqr": round(q75 - q25, 4),
                    "zeros_count": int((finite_series == 0).sum()),
                    "negative_count": int((finite_series < 0).sum()),
                    "infinite_count": int(np.isinf(num_series).sum()),
                }
            else:
                stats = {
                    "min": float("-inf") if -np.inf in num_series.values else float("inf"),
                    "max": float("inf") if np.inf in num_series.values else float("-inf"),
                    "mean": 0.0,
                    "std": 0.0,
                    "median": 0.0,
                    "q25": 0.0,
                    "q75": 0.0,
                    "iqr": 0.0,
                    "zeros_count": 0,
                    "negative_count": int((num_series < 0).sum()),
                    "infinite_count": int(np.isinf(num_series).sum()),
                }

    elif inferred_type in ("categorical", "text", "boolean") and non_null_count > 0:
        val_counts = non_null_series.astype(str).value_counts()
        if not val_counts.empty:
            top_val = str(val_counts.index[0])
            top_freq = int(val_counts.iloc[0])
            stats = {
                "top_value": top_val,
                "top_frequency": top_freq,
                "top_frequency_percentage": round(top_freq / non_null_count * 100.0, 2),
            }

    return ColumnProfile(
        name=col_name,
        pandas_dtype=pandas_dtype,
        inferred_type=inferred_type,
        total_rows=total_rows,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        distinct_count=distinct_count,
        distinct_percentage=distinct_percentage,
        is_constant=is_constant,
        is_all_missing=is_all_missing,
        sample_values=sample_values,
        stats=stats,
    )


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """
    Profile an entire DataFrame safely and non-destructively.
    Handles empty DataFrames (0 rows or 0 columns) gracefully.
    """
    total_rows = int(len(df))
    total_cols = int(len(df.columns))
    total_cells = total_rows * total_cols

    memory_bytes = int(df.memory_usage(deep=True).sum()) if total_cols > 0 else 0

    duplicate_rows = int(df.duplicated().sum()) if total_rows > 0 and total_cols > 0 else 0
    duplicate_percentage = (
        round((duplicate_rows / total_rows * 100.0), 2) if total_rows > 0 else 0.0
    )

    total_missing_cells = int(df.isna().sum().sum()) if total_cols > 0 else 0
    total_missing_percentage = (
        round((total_missing_cells / total_cells * 100.0), 2) if total_cells > 0 else 0.0
    )

    column_profiles: Dict[str, ColumnProfile] = {}
    constant_columns: List[str] = []
    all_missing_columns: List[str] = []
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    datetime_columns: List[str] = []
    boolean_columns: List[str] = []
    text_columns: List[str] = []

    for col in df.columns:
        prof = profile_column(df[col], total_rows=total_rows)
        column_profiles[str(col)] = prof

        if prof.is_constant:
            constant_columns.append(str(col))
        if prof.is_all_missing:
            all_missing_columns.append(str(col))

        if prof.inferred_type == "numeric":
            numeric_columns.append(str(col))
        elif prof.inferred_type == "categorical":
            categorical_columns.append(str(col))
        elif prof.inferred_type == "datetime":
            datetime_columns.append(str(col))
        elif prof.inferred_type == "boolean":
            boolean_columns.append(str(col))
        elif prof.inferred_type == "text":
            text_columns.append(str(col))

    return DatasetProfile(
        total_rows=total_rows,
        total_columns=total_cols,
        total_cells=total_cells,
        memory_bytes=memory_bytes,
        duplicate_rows=duplicate_rows,
        duplicate_percentage=duplicate_percentage,
        total_missing_cells=total_missing_cells,
        total_missing_percentage=total_missing_percentage,
        constant_columns=constant_columns,
        all_missing_columns=all_missing_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        boolean_columns=boolean_columns,
        text_columns=text_columns,
        columns=column_profiles,
    )
