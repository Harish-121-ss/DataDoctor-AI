# DataDoctor AI — Live Screenshot Capture Checklist

This checklist guides capturing high-resolution, presentation-ready screenshots from the live running Streamlit application (`http://localhost:8501`).

---

## Prerequisites
- Resolution: Browser maximized at 1920x1080 (100% zoom).
- Dataset: Sample dirty dataset loaded (`examples/sample_dirty_dataset.csv`).
- Theme: Default Light Theme or Clean Dark Theme (ensure text contrast is sharp).

---

## Screenshot Inventory

### 1. Ingestion & Structural Overview View
- **File Name:** `screenshot_01_overview.png`
- **Location:** View `1. Overview & Ingestion`
- **Elements to Show:**
  - Sidebar ingestion banner with `Max File Size (MB)` and `Upload CSV Dataset`.
  - Top metric cards: `Total Rows: 10`, `Total Columns: 7`, `File Size: 0.6 KB`, `Encoding / Delimiter: utf-8 / ','`.
  - Inferred Schema & Column Attributes table showing semantic types (`Numeric`, `Text`, etc.).
  - Interactive table head preview.

### 2. Quality Profiling & Defect Diagnostics
- **File Name:** `screenshot_02_profiling_health.png`
- **Location:** View `2. Quality Profiling`
- **Elements to Show:**
  - Overall Health Score metric card (`78.1 / 100.0`) and the calculation info callout.
  - Plotly horizontal bar chart for missing values by column (`department`).
  - Plotly donut chart showing defects by severity (High, Medium, Low).
  - Expand the first defect card (`TYPE_INCONSISTENCY` in `salary` or `SUSPICIOUS_NEGATIVE_VALUE` in `age`) showing affected counts and suggested remediation.

### 3. Isolation Forest Anomaly Detection & 2D Scatter
- **File Name:** `screenshot_03_anomaly_scatter.png`
- **Location:** View `3. Anomaly Detection`
- **Elements to Show:**
  - Algorithm selector set to `Isolation Forest (Scikit-learn ML)` with contamination `0.05`.
  - Metric cards: `Flagged Anomalies: 1 rows`, `Anomaly Rate: 10.0%`, `Evaluated Rows: 10`, `Latency: ~X ms`.
  - Plotly 2D bivariate scatter plot (`age` vs. `salary`) with yellow star highlighting the flagged anomaly record.
  - Flagged Records Preview table below the scatter plot.

### 4. Empirical Ground-Truth Synthetic Benchmark
- **File Name:** `screenshot_04_benchmark_table.png`
- **Location:** View `3. Anomaly Detection` (bottom expander)
- **Elements to Show:**
  - Expander titled `View Reproducible Ground-Truth Benchmark Results (Empirical Validation)`.
  - Benchmark comparison table showing Isolation Forest, Z-Score, and IQR metrics:
    - Precision, Recall, F1-Score, Specificity, True Positives, False Positives, Latency.
  - Caption highlighting the trade-off between 1D latency and multivariate correlation isolation.

### 5. Reviewable Cleaning & Dry-Run Preview
- **File Name:** `screenshot_05_cleaning_preview.png`
- **Location:** View `4. Reviewable Cleaning`
- **Elements to Show:**
  - Safety protocol warning banner.
  - Cleaning proposals catalog (`PROP-DEDUP-001`, `PROP-CAST-NUM-002`, etc.).
  - Expanded dry-run preview showing row delta (`10 -> 9 (net change: -1)`), explanation, and sample transformed entries.
  - Distinct `Preview Changes` and `Approve & Apply` buttons.

### 6. Audit Trail & Before-vs-After Comparison
- **File Name:** `screenshot_06_audit_comparison.png`
- **Location:** View `4. Reviewable Cleaning` (after approving 2-3 proposals)
- **Elements to Show:**
  - Active Health Score delta: `93.8 / 100.0` (+15.7 from baseline 78.1).
  - Structured Audit Trail table with `record_id`, `proposal_id`, `operation`, `status: SUCCESS`, and UTC timestamp.
  - Before-and-after quality comparison metric cards:
    - Total Rows: `9 (-1)`
    - Duplicate Rows: `0 (-1)`
    - Health Score: `93.8 (+15.7)`
  - Grouped Plotly comparison bar chart.

### 7. Export & Downloads Center
- **File Name:** `screenshot_07_export_reports.png`
- **Location:** View `5. Export & Audit Reports`
- **Elements to Show:**
  - Download buttons:
    - `Download Cleaned CSV`
    - `Download Untouched Original CSV`
    - `Download Quality Audit Report (.md)`
    - `Download Audit Trail Log (.json)`
  - Expander previewing the rendered Markdown Quality & Remediation Audit Report.
