"""
Main program to plot and benchmark the solvers throughput.

Author: Cian Quezon
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from throughput._benchmark import run_benchmark_functions_throughput
from throughput._plot import plot_line_graph_throughput


def main():
    
    throughput_results = run_benchmark_functions_throughput(problem_size=[10_000, 100_000, 1_000_000, 10_000_000],
                                                            seed=42, n_samples=13, n_warmup=3)
    
    plot_line_graph_throughput(throughput_results=throughput_results,
                               output_dir='benchmark/generated/plots')
    
if __name__ == "__main__":
    main()