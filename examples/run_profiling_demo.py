"""
DataDoctor AI - Stage 1 End-to-End Demo Script
Demonstrates loading a dirty CSV, calculating profiling metrics,
and printing structured data quality diagnostics.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ingestion import load_csv
from core.diagnostics import generate_quality_report


def main():
    csv_path = PROJECT_ROOT / "examples" / "sample_dirty_dataset.csv"
    print("=" * 70)
    print("          DATADOCTOR AI - STAGE 1 QUALITY PROFILING DEMO          ")
    print("=" * 70)
    print(f"\n[1] Ingesting CSV: {csv_path.name}")

    try:
        df, metadata = load_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return 1

    print(f"    - Ingestion Status: SUCCESS")
    print(f"    - Detected Delimiter: '{metadata['delimiter']}'")
    print(f"    - Detected Encoding:  {metadata['encoding']}")
    print(f"    - Rows Loaded:        {metadata['row_count']}")
    print(f"    - Columns Loaded:     {metadata['col_count']}")
    print(f"    - File Size on Disk:  {metadata['file_size_bytes']} bytes")

    print("\n[2] Executing Profiler & Quality Diagnostics...")
    report = generate_quality_report(df)

    profile = report["profile"]
    print(f"\n[3] Dataset Profile Summary:")
    print(f"    - Overall Health Score:   {report['health_score']} / 100.0")
    print(f"    - Total Cells:            {profile['total_cells']}")
    print(f"    - Total Missing Cells:    {profile['total_missing_cells']} ({profile['total_missing_percentage']}%)")
    print(f"    - Duplicate Rows:         {profile['duplicate_rows']} ({profile['duplicate_percentage']}%)")
    print(f"    - Constant Columns:       {profile['constant_columns']}")
    print(f"    - Memory Footprint:       {profile['memory_bytes']} bytes")

    print(f"\n[4] Detected Data Quality Issues ({report['total_issues']} found):")
    for idx, issue in enumerate(report["issues"], start=1):
        col_str = f" [Column: {issue['column']}]" if issue['column'] else " [Dataset Level]"
        print(f"\n    Issue #{idx}: [{issue['severity'].upper()}] {issue['issue_type']}{col_str}")
        print(f"    - Details:    {issue['explanation']}")
        print(f"    - Suggestion: {issue['suggested_action']}")

    print("\n" + "=" * 70)
    print("Demo completed successfully. Original data remains pristine and unchanged.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
