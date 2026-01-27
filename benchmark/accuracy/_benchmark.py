"""
Docstring for benchmark.accuracy._benchmark
"""

from typing import Dict

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, bisect, newton

from rapid_roots.solvers.core import RootSolvers
from _generate import generate_test_samples
from functions import (
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
    