"""
DataDoctor AI - Plotly Visualization Helpers
Generates clean, accessible, well-labeled Plotly figures for data profiling,
anomalies, distributions, and before/after comparisons.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.cleaner import QualityComparison
from core.profiler import DatasetProfile

# Professional data-science theme palette
PRIMARY_COLOR = "#1F77B4"
ACCENT_COLOR = "#FF7F0E"
ALERT_COLOR = "#D62728"
SUCCESS_COLOR = "#2CA02C"
NEUTRAL_COLOR = "#7F7F7F"
BACKGROUND_COLOR = "rgba(0,0,0,0)"


def build_missing_values_chart(profile: DatasetProfile) -> Optional[go.Figure]:
    """
    Horizontal bar chart showing missing value percentage for columns with missing data.
    """
    col_names = []
    missing_pcts = []
    missing_counts = []

    for name, p in profile.columns.items():
        if p.missing_count > 0:
            col_names.append(name)
            missing_pcts.append(p.missing_percentage)
            missing_counts.append(p.missing_count)

    if not col_names:
        return None

    # Sort ascending for horizontal bar chart
    sorted_data = sorted(zip(col_names, missing_pcts, missing_counts), key=lambda x: x[1])
    s_names, s_pcts, s_counts = zip(*sorted_data)

    colors = [ALERT_COLOR if pct >= 50.0 else ACCENT_COLOR for pct in s_pcts]

    fig = go.Figure(
        go.Bar(
            x=s_pcts,
            y=s_names,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{count} rows ({pct:.1f}%)" for count, pct in zip(s_counts, s_pcts)],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Missing: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Missing Values by Column</b>",
        xaxis_title="Missing Percentage (%)",
        yaxis_title="Column",
        xaxis=dict(range=[0, max(max(s_pcts) * 1.15, 10)]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=max(220, len(s_names) * 35),
        template="plotly_white",
    )
    return fig


def build_issues_severity_chart(issues_by_severity: Dict[str, int]) -> Optional[go.Figure]:
    """
    Donut chart displaying breakdown of diagnostic defects by severity tier.
    """
    labels = []
    values = []
    colors = []

    palette = {
        "high": ALERT_COLOR,
        "medium": ACCENT_COLOR,
        "low": PRIMARY_COLOR,
    }

    for sev in ("high", "medium", "low"):
        count = issues_by_severity.get(sev, 0)
        if count > 0:
            labels.append(sev.capitalize())
            values.append(count)
            colors.append(palette.get(sev, NEUTRAL_COLOR))

    if not values:
        return None

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+value",
            hovertemplate="<b>%{label} Severity</b><br>Count: %{value} (%{percent})<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Detected Issues by Severity</b>",
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
        template="plotly_white",
    )
    return fig


def build_feature_distribution_chart(df: pd.DataFrame, column: str) -> Optional[go.Figure]:
    """
    Histogram and KDE box plot showing numeric feature distribution and spread.
    """
    if column not in df.columns:
        return None

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    series = series[np.isfinite(series)]

    if len(series) == 0:
        return None

    fig = px.histogram(
        series,
        x=column,
        marginal="box",
        nbins=min(40, max(10, len(series) // 5)),
        title=f"<b>Distribution of {column}</b>",
        color_discrete_sequence=[PRIMARY_COLOR],
        opacity=0.8,
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        template="plotly_white",
    )
    return fig


def build_anomaly_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    predictions: Sequence[bool],
    method_name: str = "Isolation Forest",
) -> Optional[go.Figure]:
    """
    2D scatter plot visualizing normal observations vs detected anomalies in feature space.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return None

    x_vals = pd.to_numeric(df[x_col], errors="coerce")
    y_vals = pd.to_numeric(df[y_col], errors="coerce")

    valid_mask = x_vals.notna() & np.isfinite(x_vals) & y_vals.notna() & np.isfinite(y_vals)
    if valid_mask.sum() == 0:
        return None

    sub_x = x_vals[valid_mask].to_numpy()
    sub_y = y_vals[valid_mask].to_numpy()
    sub_preds = np.asarray(predictions)[valid_mask.to_numpy()]

    fig = go.Figure()

    # Normal points
    normal_mask = ~sub_preds
    if np.any(normal_mask):
        fig.add_trace(
            go.Scatter(
                x=sub_x[normal_mask],
                y=sub_y[normal_mask],
                mode="markers",
                name="Normal Records",
                marker=dict(color=PRIMARY_COLOR, size=7, opacity=0.6),
                hovertemplate=f"<b>Normal</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
            )
        )

    # Anomalies
    anomaly_mask = sub_preds
    if np.any(anomaly_mask):
        fig.add_trace(
            go.Scatter(
                x=sub_x[anomaly_mask],
                y=sub_y[anomaly_mask],
                mode="markers",
                name=f"Anomalies ({method_name})",
                marker=dict(
                    color=ALERT_COLOR,
                    symbol="diamond",
                    size=10,
                    line=dict(width=1, color="black"),
                ),
                hovertemplate=f"<b>ANOMALY</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"<b>Bivariate Anomaly Scatter ({x_col} vs {y_col})</b>",
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(l=20, r=20, t=40, b=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def build_comparison_metric_chart(comparison: QualityComparison) -> go.Figure:
    """
    Grouped bar chart comparing key metrics between Original and Cleaned datasets.
    """
    metrics = ["Rows", "Missing Cells", "Duplicate Rows", "Detected Issues"]
    before_vals = [
        comparison.rows_before,
        comparison.missing_cells_before,
        comparison.duplicate_rows_before,
        comparison.issues_count_before,
    ]
    after_vals = [
        comparison.rows_after,
        comparison.missing_cells_after,
        comparison.duplicate_rows_after,
        comparison.issues_count_after,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Original (Before)",
            x=metrics,
            y=before_vals,
            marker_color=NEUTRAL_COLOR,
            text=before_vals,
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Cleaned (After)",
            x=metrics,
            y=after_vals,
            marker_color=SUCCESS_COLOR,
            text=after_vals,
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>Dataset Metrics: Before vs. After Cleaning</b>",
        barmode="group",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig
