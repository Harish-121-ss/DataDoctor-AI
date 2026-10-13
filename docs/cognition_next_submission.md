# DataDoctor AI — CognitionNext '26 Proof-of-Work Summary

**Candidate / Author:** Harish S  
**Project:** DataDoctor AI  
**Repository:** `DataDoctor-AI`  
**Target Program:** Cognext CognitionNext '26 Technical Screening  
**Verification Status:** Verified locally (Python 3.14.0 on Windows AMD64)  
**Test Suite:** 73 passing unit/integration tests (0 failures, 0 warnings)  

---

## 1. Executive Summary & Problem Formulation

High-stakes machine learning models in finance, healthcare, and credit scoring routinely suffer from silent data degradation. Tabular data entering production environments contains missing entries, type contamination (e.g., currency symbols disguising numbers), duplicate records, invalid ranges, and multidimensional outliers.

Existing data preparation tools suffer from two critical architectural shortcomings:
1. **Unsafe Silent Mutation:** Heuristic or black-box scripts automatically drop rows or impute values without dry-run diffs, risk disclosure, or auditable user authorization.
2. **Defect Conflation:** Many tools treat statistical outliers identically to corrupted data, blindly removing rare but legitimate high-value observations.

**DataDoctor AI** is an intelligent, auditable tabular quality diagnosis, anomaly isolation, and reviewable repair system designed from first principles. It operates 100% locally with zero cloud dependencies, guarantees data immutability, and introduces an auditable governance pipeline: **Diagnose -> Propose -> Preview -> Explicitly Approve -> Audit**.

---

## 2. Technical Contributions & Architecture

DataDoctor AI is organized into clean, decoupled layers:

### A. Resilient Ingestion & Profiling (`core/ingestion.py`, `core/profiler.py`)
- Resilient CSV loading supporting UTF-8, UTF-8-BOM, Latin-1, and automatic delimiter inference (comma, semicolon, tab, pipe).
- Defensive guards enforcing a configurable 50 MB file size limit and raising explicit domain exceptions (`EmptyFileError`, `FileTooLargeError`, `MalformedFileError`).
- Comprehensive profiling: row/column counts, memory usage, sparsity patterns, and semantic type inference (`numeric`, `categorical`, `datetime`, `boolean`, `text`).

### B. Calibrated Health Scoring & Diagnostics (`core/diagnostics.py`)
- Transparent, bounded Quality Health Score ($0.0 \le S \le 100.0$):
  $$S = \max\left(0.0, 100.0 - \min(30, P_{\text{missing}}) - \min(20, P_{\text{duplicate}}) - \sum_{i} W(s_i)\right)$$
  where weights $W(\text{high}) = 5.0$, $W(\text{medium}) = 2.5$, $W(\text{low}) = 1.0$.
- Rule diagnostics for 100% null columns, partial missingness, exact/subset duplicates, currency/punctuation type inconsistencies, zero-variance constant columns, and domain-range violations (e.g. negative ages).

### C. Machine Learning Anomaly Isolation (`core/ml_anomaly.py`, `core/baselines.py`)
- Unsupervised **Isolation Forest** ensemble (`sklearn.ensemble.IsolationForest`) operating on scaled numerical feature spaces.
- Statistical baselines:
  - Gaussian Z-Score ($|Z| > 3.0$).
  - Tukey's Interquartile Range Fencing ($[Q_1 - 1.5 \cdot \text{IQR},\; Q_3 + 1.5 \cdot \text{IQR}]$).
- Explicit governance distinction: **Anomalies are statistically unusual in feature space; they are not automatically errors.**

### D. Safe Reviewable Cleaning & Audit Trail (`core/cleaner.py`)
- Converts diagnostics and anomalies into structured `CleaningProposal` records.
- Dry-run preview (`ProposalPreview`) computing exact before/after row deltas and sample diffs on an isolated copy without mutating data.
- Explicit approval guard: `apply_proposal(pid, approved=True)` raises `ProposalNotApprovedError` if approval is omitted.
- Double-application protection (`ProposalAlreadyAppliedError`).
- Immutable `AuditRecord` logging with UTC timestamps, excluding raw sensitive cell contents to preserve privacy.
- Before-and-after quality comparison (`compare_datasets`) calculating metric deltas and evaluating statistical trade-offs (sample power loss vs. variance compression).

### E. Interactive Streamlit Dashboard (`app/`)
- Five decoupled views: Overview, Quality Profiling, Anomaly Detection, Reviewable Cleaning, Export & Audit Reports.
- Plotly visualizers for missingness distributions, defect severity breakdown, and 2D anomaly scatter projections.
- Four separate download streams: Cleaned CSV, Untouched Original CSV, Markdown Quality Audit Report, and structured JSON Audit Log.

---

## 3. Empirical Benchmark Methodology & Actual Measured Results

To prevent fabricated performance claims, we developed a deterministic synthetic benchmark (`evaluation/synthetic_benchmark.py`, `evaluation/benchmark_runner.py`) containing 1,000 samples and 100 ground-truth anomalies (40 univariate spikes, 40 correlation-violating outliers, 20 low-density clusters).

### Measured Benchmark Output (Python 3.14.0 Windows AMD64):

| Method | Precision | Recall | F1-Score | Specificity | True Positives | False Positives | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest (ML)** | **0.7800** | **0.7800** | **0.7800** | **0.9756** | **78** | **22** | 1043.27 |
| **Z-Score Baseline (|Z| > 3.0)** | 1.0000 | 0.5800 | 0.7342 | 1.0000 | 58 | 0 | 37.52 |
| **Tukey's IQR Fencing (k=1.5)** | 0.8333 | 0.6000 | 0.6977 | 0.9867 | 60 | 12 | 26.49 |

### Empirical Insights & Trade-offs:
1. **Univariate Baselines (Z-score, IQR)** execute in under 40 ms with zero or low false positives, but completely fail to detect correlation-violating anomalies (recall capped at 58%–60%) because individual feature values remain within normal 1D margins.
2. **Isolation Forest** successfully isolates multidimensional correlation structure violations, achieving a balanced 0.78 F1-score, but requires ~1,000 ms execution time and hyperparameter tuning.
3. **Synthetic vs. Real-World Disclosure:** These results benchmark algorithmic trade-offs under known distributions; they do not guarantee identical accuracy across arbitrary real-world enterprise datasets.

---

## 4. Verification & Testing Evidence

- **Test Suite:** 73 passed in 31.93s (`pytest -q` across 9 test modules).
- **Core Coverage:** Ingestion, Profiler, Diagnostics, Baselines, ML Anomaly, Benchmark, Cleaner, UI Components, E2E Pipeline.
- **Privacy & Security:** Zero remote network requests; zero external LLM API dependencies; zero hardcoded local machine paths; zero sensitive cell contents recorded in audit logs.
- **Immutability:** Verified via memory copy checks and automated tests that `cleaner.original_df` and disk files are never altered.

---

## 5. Scope Boundaries, Limitations & Future Work

### Implemented & Verified in Release v1.0.0:
- CSV ingestion with encoding and delimiter auto-detection.
- Full structural profiling and calibrated health scoring.
- Multidimensional Isolation Forest and statistical baselines.
- Reviewable cleaning engine with dry-run previews, approval guards, and audit trail.
- Interactive Streamlit dashboard with Plotly charts and markdown/JSON report exports.

### Intentionally Deferred / Future Scope:
- Multi-user authentication and role-based access control (RBAC).
- Direct connectors to relational and cloud data warehouses (PostgreSQL, BigQuery, Snowflake).
- Distributed dataset execution (Apache Spark / Dask) for multi-gigabyte datasets.
- Semi-supervised anomaly detection when partial label feedback is available.
