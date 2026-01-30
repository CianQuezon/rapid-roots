"""
Main code for running accuracy benchmark
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from accuracy._plot import plot_error_distribution_boxplot
from accuracy._benchmark import run_functions_accuracy_benchmark


def main():
    benchmark_results = run_functions_accuracy_benchmark(n_samples=50, seed=42)
    plot_error_distribution_boxplot(benchmark_results)


if __name__ == "__main__":
    main()
