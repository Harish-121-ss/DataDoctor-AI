"""
DataDoctor AI - Anomaly Detection View
Integrates Scikit-learn Isolation Forest with statistical Z-score and IQR baselines.
Provides interactive feature selection, parameter tuning, 2D scatter visualization,
and documented synthetic benchmark comparison.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from core.baselines import (
    detect_iqr_anomalies,
    detect_zscore_anomalies,
    select_usable_numerical_features,
    AnomalyDetectionResult,
)
from core.ml_anomaly import detect_isolation_forest_anomalies
from evaluation.benchmark_runner import run_anomaly_benchmark
from app.utils.visualizer import build_anomaly_scatter_chart


def render_anomaly_view(df: pd.DataFrame):
    """Render interactive anomaly detection configuration and results."""
    st.subheader("Multidimensional & Statistical Anomaly Detection")

    st.info(
        "**Core Principle:** An anomaly indicates an observation that is statistically unusual "
        "or occupies a low-density region of the feature space. **It is not automatically incorrect, "
        "fraudulent, or corrupt.** Always review domain context before deciding whether to clean or remove."
    )

    # 1. Feature Selection
    usable_cols, excluded_cols = select_usable_numerical_features(df, exclude_identifiers=True)

    if not usable_cols:
        st.warning("No continuous numerical features found suitable for anomaly detection. Ensure your dataset contains numeric measurements.")
        if excluded_cols:
            st.caption(f"Excluded columns: {', '.join(excluded_cols)}")
        return

    st.markdown("### 1. Configure Anomaly Detector")
    selected_features = st.multiselect(
        "Select Numerical Features for Modeling",
        options=usable_cols,
        default=usable_cols,
        help="Identifiers, keys, and zero-variance constant columns are automatically excluded."
    )

    if not selected_features:
        st.warning("Please select at least one numerical feature.")
        return

    # Method Selector & Hyperparameters
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        method = st.selectbox(
            "Anomaly Detection Algorithm",
            options=["Isolation Forest (Scikit-learn ML)", "Z-Score (Gaussian Baseline)", "Tukey's IQR Fencing (Robust Baseline)"],
            index=0,
        )

    with col_opt2:
        if "Isolation Forest" in method:
            contamination = st.slider("Contamination Rate (Expected Anomaly %)", min_value=0.01, max_value=0.30, value=0.05, step=0.01)
            n_trees = st.slider("Ensemble Trees (n_estimators)", min_value=50, max_value=300, value=100, step=50)
        elif "Z-Score" in method:
            threshold = st.slider("Standard Deviation Threshold (|Z| > threshold)", min_value=1.5, max_value=5.0, value=3.0, step=0.25)
            combination_rule = st.selectbox("Multi-Feature Combination Rule", options=["any", "all", "majority"], index=0)
        else:
            k_val = st.slider("IQR Multiplier (k * IQR Fence)", min_value=1.0, max_value=3.5, value=1.5, step=0.25)
            combination_rule = st.selectbox("Multi-Feature Combination Rule", options=["any", "all", "majority"], index=0)

    # Execution Trigger
    if st.button("Run Anomaly Detection", type="primary"):
        with st.spinner("Executing anomaly model..."):
            if "Isolation Forest" in method:
                result = detect_isolation_forest_anomalies(
                    df,
                    columns=selected_features,
                    contamination=contamination,
                    n_estimators=n_trees,
                    random_state=42,
                )
            elif "Z-Score" in method:
                result = detect_zscore_anomalies(
                    df,
                    columns=selected_features,
                    threshold=threshold,
                    combination_rule=combination_rule,
                )
            else:
                result = detect_iqr_anomalies(
                    df,
                    columns=selected_features,
                    k=k_val,
                    combination_rule=combination_rule,
                )

            # Store in session state for cross-component access
            st.session_state["latest_anomaly_result"] = result

    # Display Results if Available
    if "latest_anomaly_result" in st.session_state:
        res: AnomalyDetectionResult = st.session_state["latest_anomaly_result"]

        st.markdown("---")
        st.markdown("### 2. Detection Results on Uploaded Dataset")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Flagged Anomalies", f"{res.anomaly_count:,} rows")
        with r2:
            st.metric("Anomaly Rate", f"{res.anomaly_percentage:.1f}%")
        with r3:
            st.metric("Evaluated Rows", f"{res.evaluated_rows:,}")
        with r4:
            st.metric("Latency", f"{res.execution_time_ms:.1f} ms")

        st.write(f"**Model Explanation:** {res.explanation}")

        # 2D Scatter Visualization
        if len(selected_features) >= 2:
            st.markdown("#### Bivariate Feature Space Visualization")
            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                x_axis = st.selectbox("X-Axis Feature", options=selected_features, index=0, key="scat_x")
            with sc_col2:
                y_axis = st.selectbox("Y-Axis Feature", options=selected_features, index=min(1, len(selected_features) - 1), key="scat_y")

            scatter_fig = build_anomaly_scatter_chart(df, x_axis, y_axis, res.predictions, method_name=res.method)
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)

        # Flagged Records Table
        if res.anomaly_count > 0:
            st.markdown("#### Flagged Records Preview")
            anom_df = df.iloc[res.anomaly_indices].copy()
            anom_df.insert(0, "Anomaly Score", [res.scores[i] for i in res.anomaly_indices])
            st.dataframe(anom_df.head(25), use_container_width=True)
        else:
            st.success("Zero anomalies detected under current parameter settings.")

    # Synthetic Benchmark Section
    st.markdown("---")
    with st.expander("🔬 View Reproducible Ground-Truth Benchmark Results (Empirical Validation)"):
        st.markdown(
            "Below are actual, reproducible comparative metrics computed on a deterministic synthetic benchmark "
            "(1,000 samples, 100 ground-truth anomalies including univariate spikes, correlation violations, and cluster outliers)."
        )

        if st.button("Run / Refresh Ground-Truth Benchmark", key="run_synth_bench"):
            with st.spinner("Evaluating models against synthetic ground truth..."):
                bench_res = run_anomaly_benchmark(n_samples=1000, contamination=0.10, random_seed=42)
                st.session_state["benchmark_results"] = bench_res

        if "benchmark_results" in st.session_state:
            b_metrics = st.session_state["benchmark_results"]["metrics"]
            bench_table = []
            for key, m in b_metrics.items():
                bench_table.append({
                    "Method": m["method_name"],
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "F1-Score": f"{m['f1_score']:.4f}",
                    "Specificity": f"{m['specificity']:.4f}",
                    "True Positives": m["true_positives"],
                    "False Positives": m["false_positives"],
                    "Latency (ms)": f"{m['execution_time_ms']:.1f}",
                })
            st.dataframe(pd.DataFrame(bench_table), use_container_width=True, hide_index=True)
            st.caption(
                "**Key Insight:** Univariate baselines (Z-score, IQR) achieve high precision and low latency but completely miss "
                "correlation violations. Isolation Forest captures multi-dimensional correlation structures at the cost of higher latency. "
                "*Note: Synthetic benchmarks illustrate algorithmic trade-offs; they do not establish identical accuracy on real-world data.*"
            )
