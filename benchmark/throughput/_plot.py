"""
Plots throughput line graph.

Authour: Cian Quezon
"""
import sys
from typing import Dict
from pathlib import Path

import seaborn as sns
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from throughput._benchmark import run_benchmark_functions_throughput

def plot_line_graph_throughput(thorughput_results: Dict, output_dir = 'benchmark/generated/plots'):
    """
    Create line plot showing throughput scaling across problem sizes.
    
    Generates a log-scale line plot comparing Brent, Bisection, and Newton-Raphson
    throughput performance across varying problem sizes. Shows how solver throughput
    (problems/second) scales with problem size.
    
    Parameters
    ----------
    throughput_results : Dict
        Results from run_benchmark_functions_throughput().
        Structure: {problem_size: {'brent': {...}, 'newton': {...}, 'bisect': {...}}}
        Each method dict contains 'throughput' and 'n_problem_size' keys.
    output_dir : str, default='benchmark/generated/plots'
        Directory to save the plot
    
    Returns
    -------
    FacetGrid
        Seaborn FacetGrid object containing the throughput line plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    throughputs = []
    methods = []
    problem_sizes = []


    for sample_size, throughput_data in thorughput_results.items():

        for method in ['brent', 'bisect', 'newton']:
            method_results = throughput_data.get(method, None)
            method_throughput = method_results.get('throughput', None)
            method_problem_size = method_results.get('n_problem_size', None)

            throughputs.append(method_throughput)
            methods.append(method)
            problem_sizes.append(method_problem_size)
        
        

    print(f"Total data points: {len(throughputs)}")
    print(f"Problem sizes: {problem_sizes}")
    print(f"Methods: {set(methods)}")

    g = sns.catplot(
        x=problem_sizes,
        y=throughputs,
        hue=methods,
        kind='point',
        palette={'brent': '#FF6B6B', 'bisect': '#4ECDC4', 'newton': '#95E1D3'},
        height=7,
        aspect=2,
        legend_out=False

    )

    g.set(yscale='log')
    g.set_axis_labels("Sample Size", "Throughput (solves/sec)", fontweight='bold')
    g.figure.suptitle('Solver Throughput using Lambert W Equation',
                      fontweight='bold', fontsize=15, y=1.02)
    g.ax.grid(True, alpha=0.3, axis='y', which='both', linestyle='--')
    g.ax.tick_params(axis='x', rotation=45)

    g.add_legend(title='Method', loc='upper left', frameon=True)
    g.legend.set_title('Method')
    sns.move_legend(g, "upper left", frameon=True)

    g.savefig(f"{output_dir}/method_throughput_line_plot.png",
              dpi=300, bbox_inches='tight')
    print(f"Saved:{output_dir}/method_throughput_line_plot.png")

    return g
        