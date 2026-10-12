# DataDoctor AI

**DataDoctor AI** is an intelligent tabular dataset quality diagnosis and repair application designed to provide genuine AI/ML proof-of-work for Cognext's *CognitionNext '26* screening.

It pairs deterministic structural profiling and rule-based defect diagnosis with unsupervised machine learning (Isolation Forest) and statistical baselines (Z-Score and Tukey's IQR) to analyze, explain, and benchmark dataset health while guaranteeing that original datasets remain completely immutable.

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
│   └── ml_anomaly.py         # Isolation Forest anomaly detection (Scikit-learn)
├── evaluation/
│   ├── __init__.py
│   ├── synthetic_benchmark.py # Deterministic ground-truth anomaly generator
│   └── benchmark_runner.py    # Benchmark comparator & metrics calculator
├── examples/
│   ├── run_profiling_demo.py  # End-to-end terminal demo
│   └── sample_dirty_dataset.csv
└── tests/
    ├── __init__.py
    ├── test_ingestion.py      # Ingestion & file resilience tests
    ├── test_profiler.py       # Statistical profiling & null tests
    ├── test_diagnostics.py    # Rule diagnostic & severity tests
    ├── test_baselines.py      # Z-Score & IQR tests
    ├── test_ml_anomaly.py     # Isolation Forest tests
    ├── test_benchmark.py      # Benchmark & metric calculation tests
    └── test_e2e.py            # End-to-end integration tests
```

---

## Core Capabilities

### 1. Data Ingestion & Quality Profiling (Stage 1)
- **Resilient CSV Ingestion** (`core/ingestion.py`): Delimiter auto-detection (`,`, `;`, `\t`, `|`), UTF-8/BOM/Latin-1 encoding handling, binary/null-byte rejection, 50MB safety limit, and guaranteed file immutability.
- **Structural Profiler** (`core/profiler.py`): Shape, memory footprint, null counts, duplicate rows, constant columns, distinctness, and semantic type inference compatible with Pandas 3.0 `StringDtype`.
- **Deterministic Diagnostics** (`core/diagnostics.py`): Severity-tiered issues (`high`, `medium`, `low`), type contamination detection (e.g. numeric columns containing string tokens), suspicious negative values in naturally positive domains, infinite numbers, extreme outliers, and overall Dataset Health Score (0–100).

### 2. Anomaly Detection & Statistical Baselines (Stage 2)
- **Isolation Forest** (`core/ml_anomaly.py`): Unsupervised multidimensional anomaly detection using Scikit-learn's `IsolationForest`. Auto-selects valid continuous numerical features, excludes identifiers/keys, standardizes inputs, and provides per-feature anomaly attribution.
- **Statistical Baselines** (`core/baselines.py`):
  - **Z-Score Method**: Parametric outlier detection assuming Gaussian distribution ($|Z| > \text{threshold}$, default 3.0).
  - **Tukey's IQR Fencing**: Non-parametric outlier detection robust to heavy tails ($[Q_1 - k \cdot \text{IQR}, Q_3 + k \cdot \text{IQR}]$, default $k=1.5$).
  - **Multi-Feature Aggregation**: Configurable combination rules (`any`, `all`, `majority`).

---

## Profiling Issues vs. Statistical Anomalies

Understanding the distinction is fundamental to DataDoctor AI:

| Dimension | Profiling Issues (Rule-Based) | Statistical Anomalies (ML & Baselines) |
| :--- | :--- | :--- |
| **Nature** | Syntactic, structural, or domain rule violations. | Probabilistic deviations in numerical feature space. |
| **Examples** | Missing values, duplicate rows, text in numeric columns, negative age. | An observation where feature combinations are rare. |
| **Ground Truth** | Objectively defective according to data schema. | **Unusual, not automatically erroneous or fraudulent.** |
| **Detection** | Deterministic heuristics and pattern matching. | Unsupervised models (Isolation Forest, Z-Score, IQR). |

---

## Reproducible Anomaly Detection Benchmark

To provide authentic, reproducible proof-of-work, DataDoctor AI includes a deterministic synthetic benchmark (`evaluation/`):
- **1,000 observations** (900 normal, 100 ground-truth anomalies = 10% contamination).
- **Three Anomaly Types**:
  1. *Univariate Spikes* (40): Individual features spiked 5–7 standard deviations away.
  2. *Multivariate Violations* (40): Observations within marginal min/max ranges, but violating strong joint correlation ($r \approx 0.85$).
  3. *Sparse Cluster Outliers* (20): Isolated pocket in low-density feature space.
- **Zero Data Leakage**: Ground-truth labels are generated separately and never passed to detectors.

### Actual Benchmark Results

Run command:
```bash
python evaluation/benchmark_runner.py
```

Output:
```
================================================================================
        DATADOCTOR AI: ANOMALY DETECTION BENCHMARK REPORT
================================================================================
Dataset Size:        1000 rows (900 normal, 100 ground-truth anomalies)
Ground Contamination:10.0%
Anomaly Breakdown:   Univariate Spikes: 40 | Multivariate Violations: 40 | Cluster: 20
--------------------------------------------------------------------------------
Method                         | Precision | Recall   | F1-Score | Spec     | TP   | FP   | Time (ms)
--------------------------------------------------------------------------------
Isolation Forest (ML)          | 0.7800    | 0.7800   | 0.7800   | 0.9756   | 78   | 22   | 1085.25 
Z-Score Baseline (|Z| > 3.0)   | 1.0000    | 0.5800   | 0.7342   | 1.0000   | 58   | 0    | 22.88   
IQR Fencing Baseline (k=1.5)   | 0.8333    | 0.6000   | 0.6977   | 0.9867   | 60   | 12   | 25.24   
================================================================================
```

### Empirical Technical Trade-offs
1. **Univariate Baselines (Z-score, IQR)**: Blazing fast (<25 ms) and zero false positives for Z-score on Gaussian data. However, they **completely miss correlation-violating anomalies** because each marginal feature value is individually unremarkable.
2. **Isolation Forest (ML)**: Successfully isolates multi-feature correlation anomalies (Recall: 0.78 vs 0.58), but incurs higher execution latency (~1,000 ms) and requires contamination tuning.
3. **Caveat**: Synthetic benchmark results illustrate algorithmic trade-offs under controlled assumptions; they do not establish identical performance across arbitrary real-world datasets.

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

---

## Known Limitations & Roadmap

1. **Interactive UI**: Scheduled for Stage 5 (Streamlit dashboard with Plotly visualizers).
2. **Safe Cleaning Engine**: Scheduled for Stage 4 (immutable repairs with before/after diffs).
3. **High Dimensionality**: For datasets with >100 numeric columns, Isolation Forest tree depth and feature subspace sampling may require PCA or feature selection.
