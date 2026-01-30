"""
Docstring for benchmark.accuracy._plot
"""

from typing import Dict
from pathlib import Path

import seaborn as sns
import numpy as np


def plot_error_distribution_boxplot(
    benchmark_results: Dict, output_dir="benchmark/generated/plots"
):
    """
    Create box plot showing distribution of individual sample errors
    across function categories.

    Each point represents the absolute error for ONE sample from the
    50 samples per function. All samples from functions in the same
    category are grouped together.

    Parameters
    ----------
    all_results : dict
        Results from run_accuracy_benchmark() containing individual
        sample errors for each function and method.

    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    categories = []
    methods = []
    errors = []

    for func_name, func_data in benchmark_results.items():
        category = func_data["category"].capitalize()
        for method in ["brent", "bisect", "newton"]:
            method_results = func_data["results"].get(method, None)
            sample_errors = method_results.get("abs_errors", None)

            if sample_errors is not None:
                for error in sample_errors:
                    if np.isfinite(error):
                        categories.append(category)
                        methods.append(method)
                        errors.append(max(error, 1e-20))

    print(f"Total data points: {len(errors)}")
    print(f"Categories: {set(categories)}")
    print(f"Methods: {set(methods)}")

    g = sns.catplot(
        x=categories,
        y=errors,
        hue=methods,
        kind="box",
        palette={"brent": "#FF6B6B", "bisect": "#4ECDC4", "newton": "#95E1D3"},
        height=7,
        aspect=2,
        legend_out=False,
    )

    g.set(yscale="log")
    g.set_axis_labels(
        "Function Category", "Absolute Error", fontweight="bold", fontsize=13
    )
    g.figure.suptitle(
        "Error Distribution Across All Samples by Category",
        fontweight="bold",
        fontsize=15,
        y=1.02,
    )
    g.ax.grid(True, alpha=0.3, axis="y", which="both", linestyle="--")
    g.ax.tick_params(axis="x", rotation=45)

    g.add_legend(title="Method", loc="upper left", frameon=True)
    g.legend.set_title("Method")
    sns.move_legend(g, "upper left", frameon=True)

    g.savefig(
        f"{output_dir}/error_distribution_boxplot.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved:{output_dir}/error_distribution_boxplot.png")

    return g
