# DataDoctor AI

**DataDoctor AI** is an intelligent tabular dataset quality diagnosis, anomaly detection, and reviewable repair application designed to provide genuine AI/ML proof-of-work for Cognext's *CognitionNext '26* screening.

It pairs deterministic structural profiling and rule-based defect diagnosis with unsupervised machine learning (Isolation Forest), statistical baselines (Z-Score and Tukey's IQR), and a safe, auditable cleaning engine that requires explicit user approval before applying any transformation. Original datasets are guaranteed to remain completely immutable.

---

## Architecture Overview

```
DataDoctor-AI/
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
├── core/
│   ├── __init__.py
│   ├── ingestion.py          # Resilient CSV reader (delimiters, BOM, size limit)
│   ├── profiler.py           # Structural & statistical column/dataset profiling
│   ├── diagnostics.py        # Rule-based diagnostics & 0-100 health scoring
│   ├── baselines.py          # Statistical baselines: Z-Score & Tukey's IQR
│   ├── ml_anomaly.py         # Isolation Forest anomaly detection (Scikit-learn)
│   └── cleaner.py            # Reviewable cleaning, explicit approval, audit log
├── evaluation/
│   ├── __init__.py
│   ├── synthetic_benchmark.py # Deterministic ground-truth anomaly generator
│   └── benchmark_runner.py    # Benchmark comparator & metrics calculator
├── examples/
│   ├── run_profiling_demo.py  # Ingestion & quality profiling demo
│   ├── run_cleaning_demo.py   # Reviewable cleaning & audit trail demo
│   └── sample_dirty_dataset.csv
└── tests/
    ├── __init__.py
    ├── test_ingestion.py      # Ingestion & file resilience tests
    ├── test_profiler.py       # Statistical profiling & null tests
    ├── test_diagnostics.py    # Rule diagnostic & severity tests
    ├── test_baselines.py      # Z-Score & IQR tests
    ├── test_ml_anomaly.py     # Isolation Forest tests
    ├── test_benchmark.py      # Benchmark & metric calculation tests
    ├── test_cleaner.py        # Reviewable cleaning, approval & audit tests
    └── test_e2e.py            # End-to-end integration tests
```

---

## Core Capabilities

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

---

## Example: Proposal-Review-Approval Workflow

```python
from core.ingestion import load_csv
from core.cleaner import DataCleaner
from core.ml_anomaly import detect_isolation_forest_anomalies

# 1. Ingest original dataset safely (remains untouched)
df, metadata = load_csv("data.csv")

# 2. Run optional anomaly detection
anomaly_result = detect_isolation_forest_anomalies(df)

# 3. Initialize cleaner session and generate proposals
cleaner = DataCleaner(df)
proposals = cleaner.generate_proposals(anomaly_results=[anomaly_result])

# 4. Preview a specific proposal without applying it
preview = cleaner.preview_proposal("PROP-DEDUP-001")
print(f"Proposed change: {preview.rows_before} -> {preview.rows_after} rows")
print(f"Explanation: {preview.explanation}")
print(f"Sample Diffs: {preview.sample_diff}")

# 5. Explicitly approve and apply
cleaned_df, audit_record = cleaner.apply_proposal("PROP-DEDUP-001", approved=True)

# 6. Inspect structured audit log
print(cleaner.get_audit_log())

# 7. Review before-and-after quality comparison
comparison = cleaner.compare_quality()
print(f"Health Score change: {comparison.health_score_before} -> {comparison.health_score_after}")
for note in comparison.evaluation_notes:
    print(f"Trade-off: {note}")
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

## Known Limitations and Cleaning Risks

1. **Information Loss from Dropping Rows**: Dropping rows with missing values or anomalies reduces sample size, reduces statistical power, and can introduce severe selection bias if missingness is not Missing Completely At Random (MCAR).
2. **Variance Compression from Imputation**: Mean/median/mode imputation preserves row count but artificially reduces feature variance and can distort covariances between features.
3. **Anomalies are Not Automatically Errors**: An anomaly flag simply indicates a point in a low-density region of the feature space. In real-world data, anomalies frequently represent critical edge cases, high-value customers, rare failures, or genuine discoveries. Removing them blindly distorts ground-truth populations.
4. **Always Retain Original Copy**: DataDoctor AI enforces that the original DataFrame is never modified in place, but practitioners should always archive the original untouched raw files.

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

## Verification & Testing

### Run All Automated Tests
```bash
pytest -v
```

### Run Profiling Demo
```bash
python examples/run_profiling_demo.py
```

### Run Anomaly Detection Benchmark
```bash
python evaluation/benchmark_runner.py
```

### Run Reviewable Cleaning & Audit Demo
```bash
python examples/run_cleaning_demo.py
```

---

## Roadmap

1. **Interactive UI**: Scheduled for Stage 5 (Streamlit dashboard with Plotly visualizers, interactive proposal checklist, diff inspector, and exportable report).
2. **Export Engine**: Cleaned CSV exporter and markdown health audit report generator.
