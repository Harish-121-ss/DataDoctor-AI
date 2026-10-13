"""
DataDoctor AI - Streamlit Dashboard Main Entry Point
Intelligent Tabular Dataset Quality Diagnosis, Anomaly Detection & Safe Repair.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from core.cleaner import DataCleaner
from core.diagnostics import generate_quality_report
from core.ingestion import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    EmptyFileError,
    FileTooLargeError,
    MalformedFileError,
    load_csv,
)
from core.profiler import profile_dataset

from app.components.overview_view import render_overview_view
from app.components.profiling_view import render_profiling_view
from app.components.anomaly_view import render_anomaly_view
from app.components.cleaning_view import render_cleaning_view
from app.components.export_view import render_export_view

# Page Configuration
st.set_page_config(
    page_title="DataDoctor AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_dataset(file_source, max_mb: int = 50):
    """Load and initialize session state for a new dataset."""
    try:
        max_bytes = max_mb * 1024 * 1024
        df, metadata = load_csv(file_source, max_file_size_bytes=max_bytes)
        init_profile = profile_dataset(df)
        init_report = generate_quality_report(df)

        st.session_state["raw_df"] = df
        st.session_state["metadata"] = metadata
        st.session_state["initial_profile"] = init_profile
        st.session_state["initial_quality_report"] = init_report
        st.session_state["profile"] = init_profile
        st.session_state["quality_report"] = init_report
        st.session_state["cleaner"] = DataCleaner(df)
        st.session_state.pop("latest_anomaly_result", None)
        st.session_state.pop("benchmark_results", None)

        # Clear any cached dry-run preview states from prior datasets
        for k in list(st.session_state.keys()):
            if k.startswith("preview_result_"):
                st.session_state.pop(k, None)

        return True, "Dataset loaded successfully."
    except FileTooLargeError as e:
        return False, f"File too large: {e}"
    except EmptyFileError as e:
        return False, f"Empty file: {e}"
    except MalformedFileError as e:
        return False, f"Malformed file: {e}"
    except Exception as e:
        return False, f"Failed to load dataset: {e}"


def main():
    # Sidebar
    st.sidebar.title("🩺 DataDoctor AI")
    st.sidebar.caption("Intelligent Tabular Quality Diagnosis & Safe Repair")
    st.sidebar.markdown("---")

    # Ingestion Controls
    st.sidebar.markdown("### 📥 Data Ingestion")
    max_size_mb = st.sidebar.number_input("Max File Size (MB)", min_value=1, max_value=100, value=50, step=5)
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"], help="Supports UTF-8, UTF-8 BOM, Latin-1, and comma/semicolon/tab/pipe delimiters.")

    # Check if a new file is uploaded
    if uploaded_file is not None:
        last_file = st.session_state.get("_last_uploaded_name")
        if last_file != uploaded_file.name:
            success, msg = load_dataset(uploaded_file, max_mb=max_size_mb)
            if success:
                st.session_state["_last_uploaded_name"] = uploaded_file.name
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
    else:
        # Reset uploader state if user removed the uploaded file
        if st.session_state.get("_last_uploaded_name") not in (None, "sample_dirty_dataset.csv"):
            st.session_state["_last_uploaded_name"] = None

    # Demo Dataset Loader
    demo_path = PROJECT_ROOT / "examples" / "sample_dirty_dataset.csv"
    if demo_path.exists():
        if st.sidebar.button("Load Sample Dirty Dataset", use_container_width=True):
            success, msg = load_dataset(str(demo_path), max_mb=max_size_mb)
            if success:
                st.session_state["_last_uploaded_name"] = "sample_dirty_dataset.csv"
                st.sidebar.success("Loaded demo dataset!")
                st.rerun()

    st.sidebar.markdown("---")

    # Navigation
    st.sidebar.markdown("### 🧭 Navigation")
    nav_options = [
        "1. Overview & Ingestion",
        "2. Quality Profiling",
        "3. Anomaly Detection",
        "4. Reviewable Cleaning",
        "5. Export & Audit Reports",
    ]
    selected_nav = st.sidebar.radio("Select View", nav_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("Cognext's CognitionNext '26 Technical Assessment")

    # Main Area
    st.title("🩺 DataDoctor AI")
    st.caption("AI/ML dataset quality diagnosis, unsupervised anomaly isolation, and reviewable repair engine.")

    # Check for Active Dataset
    if "cleaner" not in st.session_state or st.session_state.get("cleaner") is None:
        st.info("👋 **Welcome to DataDoctor AI!** Please upload a CSV dataset using the sidebar, or click **'Load Sample Dirty Dataset'** to explore quality diagnosis and safe cleaning.")

        col_feat1, col_feat2, col_feat3 = st.columns(3)
        with col_feat1:
            st.markdown("#### 🔍 Structural Profiling")
            st.write("Automatically detect missingness distributions, duplicates, type contamination, constant columns, and calculate transparent health scores.")
        with col_feat2:
            st.markdown("#### 🤖 Anomaly Isolation")
            st.write("Unsupervised multidimensional anomaly detection using Scikit-learn's Isolation Forest alongside statistical Z-Score and Tukey's IQR baselines.")
        with col_feat3:
            st.markdown("#### 🛡️ Safe Repair & Audit")
            st.write("Dry-run preview changes before applying. Require explicit user approval for transformations and preserve an untouched original dataset.")
        return

    cleaner: DataCleaner = st.session_state["cleaner"]
    df: pd.DataFrame = cleaner.current_df
    metadata: dict = st.session_state.get("metadata", {})
    initial_report = st.session_state.get("initial_quality_report", st.session_state.get("quality_report"))

    # Compute reactive current profile and quality report if cleaning operations have been applied
    if len(cleaner.audit_log) > 0:
        current_profile = profile_dataset(df)
        current_quality_report = generate_quality_report(df)
    else:
        current_profile = st.session_state.get("initial_profile", st.session_state.get("profile"))
        current_quality_report = initial_report

    st.session_state["profile"] = current_profile
    st.session_state["quality_report"] = current_quality_report

    # Routing based on selected navigation
    if selected_nav.startswith("1."):
        render_overview_view(df, metadata, current_profile)
    elif selected_nav.startswith("2."):
        render_profiling_view(df, current_quality_report, current_profile, initial_report=initial_report)
    elif selected_nav.startswith("3."):
        render_anomaly_view(df)
    elif selected_nav.startswith("4."):
        render_cleaning_view(cleaner)
    elif selected_nav.startswith("5."):
        render_export_view(cleaner, metadata, initial_report)


if __name__ == "__main__":
    main()
