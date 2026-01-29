import numpy as np
import time
from scipy.optimize import brentq, bisect, newton
from typing import Dict, List, Tuple
import sys
from pathlib import Path

from tqdm import tqdm
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rapid_roots.solvers.core import RootSolvers
from accuracy._calculations import calculate_error_metrics
from shared._run_benchmark import (
    _benchmark_rapid_roots_bisect, 
    _benchmark_rapid_roots_brent,
    _benchmark_rapid_roots_newton,
    _benchmark_scipy_bisect,
    _benchmark_scipy_brent,
    _benchmark_scipy_newton
)
from shared._generate import generate_test_samples
from shared.functions import (
    FUNCTIONS_LIST,
)

def run_functions_accuracy_benchmark(n_samples: int = 50, seed: int = 42) -> Dict:
    """
    Run complete accuracy benchmark on all 25 functions.
    
    Parameters
    ----------
    n_samples : int
        Number of samples per function
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Complete results for all functions and methods
    """
    
    all_results = {}
    
    for func_dict in tqdm(FUNCTIONS_LIST, desc=' Solving Functions'):
        name = func_dict['name']
        category = func_dict['category']
        difficulty = func_dict['difficulty']
        
        # Run benchmark
        start_time = time.time()
        results = run_benchmark_single_function(func_dict, n_samples, seed)
        elapsed = time.time() - start_time
        
        # Store results
        all_results[name] = {
            'category': category,
            'difficulty': difficulty,
            'description': func_dict['description'],
            'results': results,
            'time': elapsed
        }
    
    return all_results



def run_benchmark_single_function(func_dict: Dict, n_samples: int = 50,
                              seed: int = 42) -> Dict:
    """
    Benchmark a single function across all methods.
    
    Parameters
    ----------
    func_dict : dict
        Function dictionary from FUNCTIONS_LIST
    n_samples : int
        Number of random samples to test
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Results for all methods containing error metrics
    """
    name = func_dict['name']
    
    params, a_bounds, b_bounds = generate_test_samples(func_dict, n_samples, seed)
    
    x0_values = (a_bounds + b_bounds) / 2.0
    
    results = {}
    
    # Brent Method
    scipy_brent_results = _benchmark_scipy_brent(func_dict, params, a_bounds, b_bounds)
    rapid_roots_brent_results = _benchmark_rapid_roots_brent(func_dict, params, a_bounds, b_bounds)
    
    brent_error_metrics = calculate_error_metrics(scipy_brent_results, rapid_roots_brent_results)
    results['brent'] = {
        'scipy_converged': np.sum(np.isfinite(scipy_brent_results)),
        'rapid_converged': np.sum(np.isfinite(rapid_roots_brent_results)),
        **brent_error_metrics
    }
    
    # Bisection Method
    scipy_bisect_results = _benchmark_scipy_bisect(func_dict, params, a_bounds, b_bounds)
    rapid_roots_bisect_results = _benchmark_rapid_roots_bisect(func_dict, params, a_bounds, b_bounds)
    
    bisect_error_metrics = calculate_error_metrics(scipy_bisect_results, rapid_roots_bisect_results)
    results['bisect'] = {
        'scipy_converged': np.sum(np.isfinite(scipy_bisect_results)),
        'rapid_converged': np.sum(np.isfinite(rapid_roots_bisect_results)),
        **bisect_error_metrics
    }
    
    # Newton Method
    scipy_newton_results = _benchmark_scipy_newton(func_dict, params, x0_values)
    rapid_roots_newton_results = _benchmark_rapid_roots_newton(func_dict, params, x0_values)
    
    newton_error_metrics = calculate_error_metrics(scipy_newton_results, rapid_roots_newton_results)
    results['newton'] = {
        'scipy_converged': np.sum(np.isfinite(scipy_newton_results)),
        'rapid_converged': np.sum(np.isfinite(rapid_roots_newton_results)),
        **newton_error_metrics
    }

    return results
