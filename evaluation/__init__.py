"""
DataDoctor AI - Evaluation and Benchmarking Suite
"""

from evaluation.synthetic_benchmark import generate_synthetic_anomaly_dataset
from evaluation.benchmark_runner import run_anomaly_benchmark, BenchmarkMetrics

__all__ = [
    "generate_synthetic_anomaly_dataset",
    "run_anomaly_benchmark",
    "BenchmarkMetrics",
]
