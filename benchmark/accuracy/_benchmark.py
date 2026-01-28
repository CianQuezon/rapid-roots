"""
Docstring for benchmark.accuracy._benchmark
"""
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, bisect, newton

from rapid_roots.solvers.core import RootSolvers

sys.path.append(str(Path(__file__).parent.parent))

from _calculations import calculate_error_metrics
from shared._generate import generate_test_samples
from shared.functions import (
    ACCURACY_TEST_FUNCTIONS,
    get_function_by_name
)

def run_benchmark_single_function(func_dict: Dict, n_samples: int = 50,
                                  seed: int = 42) -> Dict:
    """
    Benchmarks a single function which uses all methods.
    
    Parameters
    ----------
    func_dict : dict
        Function dictionary from ACCURACY_TEST_FUNCTIONS
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
    category = func_dict['category']

    params, a_bounds, b_bounds = generate_test_samples(func_dict=func_dict, n_samples=n_samples, seed=seed)

    x0_values = (a_bounds + b_bounds) / 2.0

    results = {}

    # Bisection results
    scipy_bisect_results = _benchmark_scipy_bisect(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds)
    rapid_roots_bisect_results = _benchmark_rapid_roots_bisect(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds)

    bisect_error_metrics = calculate_error_metrics(scipy_results=scipy_bisect_results, rapid_roots_results=rapid_roots_bisect_results)
    
    results['bisect'] = {
        'rapid_roots_converged': np.sum(np.isfinite(rapid_roots_bisect_results)),
        'scipy_converged': np.sum(np.isfinite(scipy_bisect_results)),
        **bisect_error_metrics
    }

    # Newton results
    scipy_newton_results = _benchmark_scipy_newton(func_dict=func_dict, params=params, x0_values=x0_values)
    rapid_roots_newton_results = _benchmark_rapid_roots_newton(func_dict=func_dict, params=params, x0_values=x0_values)

    newton_error_metrics = calculate_error_metrics(scipy_results=scipy_newton_results, rapid_roots_results=rapid_roots_newton_results)
    results['newton'] = {
        'scipy_converged': np.sum(np.isfinite(scipy_newton_results)),
        'rapid_roots_converged': np.sum(np.isfinite(rapid_roots_newton_results)),
        **newton_error_metrics
    }

    # Brent results
    scipy_brent_results = _benchmark_scipy_brent(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds)
    rapid_roots_brent_results = _benchmark_rapid_roots_brent(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds)



def _benchmark_scipy_newton(func_dict: Dict, params: npt.NDArray,
                           x0_values: npt.NDArray) -> npt.NDArray:
    """
    Runs Scipy Newton Raphson on all samples
    """
    n_samples = len(params)
    results = np.full(n_samples, np.nan)

    func_scipy = func_dict['func_scipy']
    func_prime_scipy = func_dict['func_prime_scipy']

    for i in range(n_samples):
        try:
            result = newton(
                func_scipy,
                x0_values[i],
                fprime=func_prime_scipy,
                args=tuple(params[i]),
                tol = 1e-12,
                maxiter=100
            )
            results[i] = results
        except:
            pass
    
    return results

def _benchmark_scipy_brent(func_dict: Dict, params: npt.NDArray,
                          a_bounds: npt.NDArray, b_bounds: npt.NDArray) -> npt.NDArray:
    """
    Runs scipy bisection on all samples
    """
    n_samples = len(params)
    results = np.full(n_samples, np.nan)

    func_scipy = func_dict['func_scipy']

    for i in range(n_samples):
        try:
            result = brentq(
                func_scipy,
                a_bounds[i],
                b_bounds[i],
                args=tuple(params[i]),
                xtol=1e-12,
                maxiter=100
            )

            results[i] = result
        
        except:
            pass
    
    return results

def _benchmark_scipy_bisect(func_dict: Dict, params: npt.NDArray,
                           a_bounds: npt.NDArray, b_bounds: npt.NDArray):
    """
    Run scipy bisection on all samples.
    """
    n_samples = len(params)
    results = np.full(n_samples, np.nan)

    func_scipy = func_dict['func_scipy']

    for i in range(n_samples):
        
        try:
            result = bisect(
                func_scipy,
                a_bounds[i],
                b_bounds[i],
                args=tuple(params[i]),
                xtol=1e-12,
                maxiter=100
            )
            
            results[i] = result
        except:
            pass
    
    return results

def _benchmark_rapid_roots_brent(func_dict: Dict, params: npt.NDArray,
                                a_bounds: npt.NDArray, b_bounds: npt.NDArray) -> npt.NDArray:
    """
    Run rapid-roots Brent on all samples
    """
    func = func_dict['func']

    try:
        results, iters, converged = RootSolvers.get_root(
            func=func,
            a=a_bounds,
            b=b_bounds,
            func_params=params,
            tol=1e-12,
            max_iter=100,
            use_backup=False
        )

        results[np.logical_not(converged)] = np.nan

        return results
    
    except Exception as e:
        print(f" rapid-roots Brent failed: {e}")
        return np.full(len(params, np.nan))

def _benchmark_rapid_roots_bisect(func_dict: Dict, params: npt.NDArray,
                                 a_bounds: npt.NDArray, b_bounds: npt.NDArray) -> npt.NDArray:
    """
    Run rapid-roots Bisection on all samples
    """
    func = func_dict['func']
    
    try:
        results, iters, converged = RootSolvers.get_root(
            func=func,
            a=a_bounds,
            b=b_bounds,
            func_params=params,
            main_solver='bisection',
            tol=1e-12,
            max_iter=100,
            use_backup=False
        )

        results[np.logical_not(converged)] = np.nan

        return results
    
    except Exception as e:
        print(f" rapid-roots Bisection failed: {e}")
        return np.full(len(params), np.nan)
    
def _benchmark_rapid_roots_newton(func_dict: Dict, params: npt.NDArray,
                                  x0_values: npt.NDArray) -> npt.NDArray:
    """
    Run rapid-roots Newton on all samples
    """
    func = func_dict['func']
    func_prime = func_dict['func_prime']

    try: 
        results, iters, converged = RootSolvers.get_root(
            func=func,
            x0=x0_values,
            func_prime=func_prime,
            func_params=params,
            main_solver="newton",
            tol=1e-12,
            max_iter=100,
            use_backup=False
        )

        results[np.logical_not(converged)] = np.nan
        return results
    
    except Exception as e:
        print(f" rapid-roots Newton failed: {e}")
        return np.full(len(params), np.nan)
    