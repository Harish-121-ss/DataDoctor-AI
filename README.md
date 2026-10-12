# DataDoctor AI

**DataDoctor AI** is a dataset quality diagnosis and repair application designed to provide genuine AI/ML proof-of-work for Cognext's *CognitionNext '26* screening.

It automates the detection, profiling, and repair recommendations for tabular dataset defects using deterministic profiling, rule-based diagnostic verification, and unsupervised machine learning anomaly detection.

---

## Stage 1: Data Ingestion & Quality Profiling

Stage 1 implements the core data processing foundation without UI or ML models:
1. **Resilient CSV Ingestion** (`core/ingestion.py`):
   - Automatic delimiter inference (supporting `,`, `;`, `\t`, `|`).
   - Encoding resilience with transparent UTF-8 with BOM (`utf-8-sig`) and fallback detection.
   - Configurable file-size thresholds (50 MB default) to prevent out-of-memory crashes.
   - Defensive error classification (`EmptyFileError`, `FileTooLargeError`, `MalformedFileError`).
   - Non-destructive processing that guarantees disk file immutability.
2. **Quality Profiling Engine** (`core/profiler.py`):
   - Computes structural metrics: row count, column count, memory footprint, missingness %, duplicate %, and distinctness.
   - Semantic type inference: categorizes columns into `numeric`, `categorical`, `text`, `boolean`, `datetime`, or `empty`.
   - Comprehensive statistical summaries (min, max, mean, std, median, quartiles, IQR, zero count, negative count, infinite count).
3. **Structured Diagnostics Engine** (`core/diagnostics.py`):
   - Classifies defects into severity tiers (`high`, `medium`, `low`).
   - Detects dataset duplicates, all-missing columns, high/low null fractions, constant columns (zero variance), disguised numeric types containing string tokens, suspicious negative values in naturally positive domains, infinite values, and extreme outliers.
   - Computes an overall **Dataset Health Score (0–100)**.
   - Supplies non-destructive, actionable repair suggestions for each detected defect.

---

## Directory Structure

```
DataDoctor-AI/
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
├── core/
│   ├── __init__.py
│   ├── ingestion.py          # Resilient CSV parser & error classifier
│   ├── profiler.py           # Dataset & column profiling engine
│   └── diagnostics.py        # Rule-based diagnostics & health scoring
├── examples/
│   ├── run_profiling_demo.py # End-to-end terminal demo
│   └── sample_dirty_dataset.csv
└── tests/
    ├── __init__.py
    ├── test_ingestion.py     # Ingestion & file resilience tests
    ├── test_profiler.py      # Statistical profiling & null tests
    ├── test_diagnostics.py   # Rule diagnostic & severity tests
    └── test_e2e.py           # End-to-end integration tests
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.14.0 Windows 64-bit)

### Installation
Clone or navigate to the project directory:
```bash
cd "DataDoctor-AI"
```

Install Stage 1 dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Running the Automated Test Suite
Run all tests with `pytest`:
```bash
pytest -v
```

### 2. Running the Stage 1 Interactive Demo
Run the included end-to-end demo on a sample dirty dataset:
```bash
python examples/run_profiling_demo.py
```

### 3. Programmatic Usage in Python
```python
from core.ingestion import load_csv
from core.diagnostics import generate_quality_report

# Load CSV safely
df, metadata = load_csv("path/to/dataset.csv")

# Run full quality audit
report = generate_quality_report(df)

print(f"Health Score: {report['health_score']}/100.0")
for issue in report["issues"]:
    print(f"[{issue['severity'].upper()}] {issue['issue_type']}: {issue['explanation']}")
    print(f"Suggested Action: {issue['suggested_action']}")
```

---

## Known Limitations & Next Steps

1. **Unsupervised ML Anomaly Detection**: Scheduled for Stage 2 / Stage 3 (Isolation Forest integration comparing against statistical baselines).
2. **Interactive UI**: Scheduled for Stage 5 (Streamlit dashboard with Plotly charts and downloadable repair reports).
3. **Data Scope**: Stage 1 exclusively targets tabular CSV formats. Parquet, Excel (.xlsx), and JSON ingestion will be added in later extensions.
4. **Natural Domain Heuristics**: Domain-specific suspicious value detection relies on keyword patterns (e.g. `age`, `salary`, `price`). Column names in unfamiliar languages or obscure naming conventions require custom rule overrides.
