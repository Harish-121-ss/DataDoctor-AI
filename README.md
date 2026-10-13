# DataDoctor AI

**DataDoctor AI** is an intelligent tabular dataset quality diagnosis, anomaly detection, and reviewable repair web application designed to provide genuine AI/ML proof-of-work for Cognext's *CognitionNext '26* screening.

It pairs deterministic structural profiling and rule-based defect diagnosis with unsupervised machine learning (Isolation Forest), statistical baselines (Z-Score and Tukey's IQR), and a safe, auditable cleaning engine that requires explicit user approval before applying any transformation. Original datasets are guaranteed to remain completely immutable.

---

## Architecture Overview

```
DataDoctor-AI/
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
├── app/
│   ├── __init__.py
│   ├── main.py                # Streamlit application entry point
│   ├── components/            # UI components per view
│   │   ├── __init__.py
│   │   ├── overview_view.py   # Ingestion metadata & raw table preview
│   │   ├── profiling_view.py  # Health score, nulls, duplicates, diagnostics
│   │   ├── anomaly_view.py    # Isolation Forest & baseline anomaly detection
│   │   ├── cleaning_view.py   # Reviewable proposals, previews, approvals
│   │   └── export_view.py     # Cleaned CSV & markdown quality report exports
│   └── utils/
│       ├── __init__.py
│       ├── visualizer.py      # Plotly interactive chart builders
│       └── report_gen.py      # Markdown quality & remediation audit generator
├── core/
│   ├── __init__.py
│   ├── ingestion.py           # Resilient CSV reader (delimiters, BOM, size limit)
│   ├── profiler.py            # Structural & statistical column/dataset profiling
│   ├── diagnostics.py         # Rule-based diagnostics & 0-100 health scoring
│   ├── baselines.py           # Statistical baselines: Z-Score & Tukey's IQR
│   ├── ml_anomaly.py          # Isolation Forest anomaly detection (Scikit-learn)
│   └── cleaner.py             # Reviewable cleaning, explicit approval, audit log
├── evaluation/
│   ├── __init__.py
│   ├── synthetic_benchmark.py  # Deterministic ground-truth anomaly generator
│   └── benchmark_runner.py     # Benchmark comparator & metrics calculator
├── examples/
│   ├── run_profiling_demo.py   # Ingestion & quality profiling demo
│   ├── run_cleaning_demo.py    # Reviewable cleaning & audit trail demo
│   └── sample_dirty_dataset.csv
└── tests/
    ├── __init__.py
    ├── test_ingestion.py       # Ingestion & file resilience tests
    ├── test_profiler.py        # Statistical profiling & null tests
    ├── test_diagnostics.py     # Rule diagnostic & severity tests
    ├── test_baselines.py       # Z-Score & IQR tests
    ├── test_ml_anomaly.py      # Isolation Forest tests
    ├── test_benchmark.py       # Benchmark & metric calculation tests
    ├── test_cleaner.py         # Reviewable cleaning, approval & audit tests
    ├── test_ui.py              # Visualizers, report generator, app imports
    └── test_e2e.py             # End-to-end integration tests
```

---

## Core Capabilities by Stage

### 1. Data Ingestion & Quality Profiling (Stage 1)
- **Resilient CSV Ingestion** (`core/ingestion.py`): Delimiter auto-detection (`,`, `;`, `\t`, `|`), UTF-8/BOM/Latin-1 encoding handling, binary/null-byte rejection, 50MB safety limit, and guaranteed file immutability.
- **Structural Profiler** (`core/profiler.py`): Shape, memory footprint, null counts, duplicate rows, constant columns, distinctness, and semantic type inference compatible with Pandas 3.0 `StringDtype`.
- **Deterministic Diagnostics** (`core/diagnostics.py`): Severity-tiered issues (`high`, `medium`, `low`), type contamination detection, suspicious negative values in naturally positive domains, infinite numbers, extreme outliers, and overall Dataset Health Score (0–100).

### 2. Anomaly Detection & Statistical Baselines (Stage 2)
- **Isolation Forest** (`core/ml_anomaly.py`): Unsupervised multidimensional anomaly detection using Scikit-learn's `IsolationForest`. Auto-selects valid continuous numerical features, excludes identifiers/keys, standardizes inputs, and provides per-feature anomaly attribution.
- **Statistical Baselines** (`core/baselines.py`):
  - **Z-Score Method**: Parametric outlier detection assuming Gaussian distribution ($|Z| > \text{threshold}$, default 3.0).
  - **Tukey's IQR Fencing**: Non-parametric outlier detection robust to heavy tails ($[Q_1 - k \cdot \text{IQR}, Q_3 + k \cdot \text{IQR}]$, default $k=1.5$).
  - **Multi-Feature Aggregation**: Configurable combination rules (`any`, `all`, `majority`).

### 3. Reviewable Cleaning & Audit Trail (Stage 3)
- **Cleaning Proposal Engine** (`core/cleaner.py`): Converts diagnosed defects and anomaly results into structured, reviewable proposals (`CleaningProposal`).
- **Supported Operations**:
  - `deduplicate_rows`: Remove exact duplicate records while preserving a documented representative row (`keep="first"` or `"last"`).
  - `impute_median` / `impute_mean`: Fill missing numerical values with robust column medians or means.
  - `impute_mode` / `impute_constant`: Fill missing categorical/text values with dominant class or user-defined placeholders.
  - `drop_missing_rows`: Discard incomplete observations for complete-case analysis.
  - `drop_column`: Remove completely empty or zero-variance constant columns.
  - `convert_numeric`: Clean stray currency symbols (`$`, `€`, `£`, `¥`), commas, and whitespace, safely coercing text to numeric float/int.
  - `normalize_text`: Strip leading/trailing whitespace and collapse redundant internal spaces.
  - `clip_bounds`: Enforce domain boundaries (e.g., non-negative lower bound of 0.0) without discarding records.
  - `replace_with_nan`: Convert infinite or unparseable values to NaN for domain verification.
  - `flag_anomalies`: Non-destructive tagging of records identified as anomalous by ML models.
  - `filter_anomalies`: Remove flagged outlier records for models sensitive to extreme distributions.
- **Explicit Approval Guard**: Operations require explicit `approved=True`; attempts to apply without approval raise `ProposalNotApprovedError`.
- **Dry-Run Previews**: `preview_proposal` calculates rows/columns before and after, affected counts, and sample diffs on an isolated copy without mutating data.
- **Idempotency & Double-Application Protection**: Cleaner sessions track applied proposal IDs and reject duplicate applications via `ProposalAlreadyAppliedError`.
- **Structured Audit Trail**: Records proposal ID, operation, parameter changes, affected row/cell counts, and ISO timestamp. Excludes raw sensitive data by default.
- **Before-and-After Quality Comparison**: Computes before/after profiles and articulates technical trade-offs honestly.

### 4. Interactive Streamlit Dashboard (Stage 4)
- **Single-Command Launch**: Run `streamlit run app/main.py`.
- **5 Integrated Views**:
  1. **Overview & Ingestion**: Ingestion metadata, dataset dimensions, inferred types, and interactive data preview.
  2. **Quality Profiling**: Transparent Health Score (0–100) with penalty breakdown, Plotly missingness charts, defect severity donut charts, and expandable defect catalog.
  3. **Anomaly Detection**: Interactive Isolation Forest and baseline configuration, 2D feature-space scatter visualization, flagged records table, and reproducible synthetic benchmark results.
  4. **Reviewable Cleaning**: Proposal catalog, dry-run change preview, explicit approval buttons, live audit trail log, and before-and-after quality comparison cards.
  5. **Export & Audit Reports**: Download Cleaned CSV, Untouched Original CSV, formatted Markdown Quality Audit Report, and structured JSON Audit Log.

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14.0 Windows 64-bit)

### Installation
```bash
cd "DataDoctor-AI"
pip install -r requirements.txt
```

---

## Quickstart: Launching the Dashboard

Launch the interactive web application using Streamlit:
```bash
streamlit run app/main.py
```

Once running, navigate to `http://localhost:8501` in your web browser.

### Using the Dashboard:
1. **Upload Dataset**: Upload any CSV file via the sidebar (or click **"Load Sample Dirty Dataset"** for instant demonstration).
2. **Inspect Quality**: Navigate to **"2. Quality Profiling"** to view missingness, type errors, duplicate counts, and health score deductions.
3. **Isolate Anomalies**: Go to **"3. Anomaly Detection"** to configure Isolation Forest or statistical baselines, select numerical features, and inspect the 2D feature-space scatter plot.
4. **Review & Approve Repairs**: In **"4. Reviewable Cleaning"**, click **"Generate Cleaning Proposals"**, click **"Preview Changes"** to inspect diffs, and click **"Approve & Apply"** to execute approved operations.
5. **Download Cleaned Data & Reports**: Open **"5. Export & Audit Reports"** to download the cleaned CSV, the untouched original CSV, the comprehensive Markdown Audit Report, and the JSON Audit Log.

---

## Running Verification & Automated Tests

### Run All 70 Automated Tests
```bash
pytest -v
```

### Run Anomaly Detection Benchmark
```bash
python evaluation/benchmark_runner.py
```

### Run Profiling Demo Script
```bash
python examples/run_profiling_demo.py
```

### Run Reviewable Cleaning Demo Script
```bash
python examples/run_cleaning_demo.py
```

---

## Profiling Issues vs. Statistical Anomalies

| Dimension | Profiling Issues (Rule-Based) | Statistical Anomalies (ML & Baselines) |
| :--- | :--- | :--- |
| **Nature** | Syntactic, structural, or domain rule violations. | Probabilistic deviations in numerical feature space. |
| **Examples** | Missing values, duplicate rows, text in numeric columns, negative age. | An observation where feature combinations are rare. |
| **Ground Truth** | Objectively defective according to data schema. | **Unusual, not automatically erroneous or fraudulent.** |
| **Detection** | Deterministic heuristics and pattern matching. | Unsupervised models (Isolation Forest, Z-Score, IQR). |

---

## Privacy & Data Governance Statement

- **100% On-Premise Execution**: Zero external telemetry, zero LLM APIs, and zero third-party cloud data transfer.
- **Strict Immutability**: Uploaded files and original DataFrames are never modified in place.
- **Audit Logging**: Every approved operation produces a structured audit record with UTC timestamp and parameter details.
- **Sensitive Data Privacy**: Audit logs and reports exclude full row contents by default.

---

## Implemented vs. Deferred Features

| Feature | Status | Notes |
| :--- | :---: | :--- |
| Resilient CSV Ingestion (BOM, Delimiters, Size Limit) | ✅ Implemented | Stage 1 (`core/ingestion.py`) |
| Data Quality Profiling & Health Scoring | ✅ Implemented | Stage 1 (`core/profiler.py`, `core/diagnostics.py`) |
| Isolation Forest & Statistical Baselines | ✅ Implemented | Stage 2 (`core/ml_anomaly.py`, `core/baselines.py`) |
| Reproducible Synthetic Anomaly Benchmark | ✅ Implemented | Stage 2 (`evaluation/benchmark_runner.py`) |
| Reviewable Cleaning, Approval Guard & Audit Trail | ✅ Implemented | Stage 3 (`core/cleaner.py`) |
| Interactive Streamlit Web Dashboard | ✅ Implemented | Stage 4 (`app/main.py`, `app/components/`) |
| Cleaned CSV & Markdown Quality Report Exporters | ✅ Implemented | Stage 4 (`app/components/export_view.py`) |
| User Authentication & Cloud Databases | ⏳ Deferred | Out of scope for standalone screening MVP |
| Unstructured Text / Image Data Processing | ⏳ Deferred | Focus is strictly on tabular CSV data |
