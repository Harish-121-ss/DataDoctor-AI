"""
DataDoctor AI - Overview & Ingestion View
Displays dataset dimensions, sample records, inferred data types, and ingestion metadata.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from core.profiler import DatasetProfile


def render_overview_view(df: pd.DataFrame, metadata: dict, profile: DatasetProfile):
    """Render dataset overview metrics, table preview, and column metadata."""
    st.subheader("Dataset Ingestion & Structural Overview")

    # Ingestion Metadata banner
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Rows", f"{profile.total_rows:,}")
    with col_m2:
        st.metric("Total Columns", f"{profile.total_columns:,}")
    with col_m3:
        size_kb = metadata.get("file_size_bytes", profile.memory_bytes) / 1024.0
        st.metric("File Size", f"{size_kb:.1f} KB")
    with col_m4:
        st.metric("Encoding / Delimiter", f"{metadata.get('encoding', 'utf-8')} / '{metadata.get('delimiter', ',')}'")

    st.markdown("---")

    # Interactive Table Preview
    st.markdown("### Raw Dataset Preview")
    preview_limit = st.slider("Preview Rows", min_value=5, max_value=min(100, max(5, len(df))), value=min(10, len(df)), step=5)
    st.dataframe(df.head(preview_limit), use_container_width=True)

    # Column Schema and Inferred Types Table
    st.markdown("### Inferred Schema & Column Attributes")
    schema_rows = []
    for col_name, cp in profile.columns.items():
        sample_disp = ", ".join(str(v) for v in cp.sample_values[:3] if v is not None)
        schema_rows.append({
            "Column Name": col_name,
            "Pandas Dtype": cp.pandas_dtype,
            "Inferred Semantic Type": cp.inferred_type.capitalize(),
            "Non-Null Count": f"{cp.non_null_count:,}",
            "Missing Count": f"{cp.missing_count:,} ({cp.missing_percentage:.1f}%)",
            "Unique Count": f"{cp.distinct_count:,}",
            "Sample Values": sample_disp or "-",
        })

    schema_df = pd.DataFrame(schema_rows)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
