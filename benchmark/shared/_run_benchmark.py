"""
Code for benchmarking using rapid-roots and scipy.

Author: Cian Quezon
"""
import numpy as np
import time
from scipy.optimize import brentq, bisect, newton
from typing import Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rapid_roots.solvers.core import RootSolvers


def _benchmark_scipy_brent(func_dict: Dict, params: np.ndarray,
                          a_bounds: np.ndarray, b_bounds: np.ndarray) -> np.ndarray:
    """Run SciPy Brent on all samples."""
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


def _benchmark_scipy_bisect(func_dict: Dict, params: np.ndarray,
                           a_bounds: np.ndarray, b_bounds: np.ndarray) -> np.ndarray:
    """Run SciPy Bisection on all samples."""
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


def _benchmark_scipy_newton(func_dict: Dict, params: np.ndarray,
                           x0_values: np.ndarray) -> np.ndarray:
    """Run SciPy Newton on all samples."""
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
                tol=1e-12,
                maxiter=100
            )
            results[i] = result
        except:
            pass
    
    return results


def _benchmark_rapid_roots_brent(func_dict: Dict, params: np.ndarray,
                                 a_bounds: np.ndarray, b_bounds: np.ndarray) -> np.ndarray:
    """Run rapid-roots Brent on all samples."""
    func = func_dict['func']
    
    try:
        results, iters, converged = RootSolvers.get_root(
            func=func,
            a=a_bounds,
            b=b_bounds,
            func_params=params,
            main_solver='brent',
            tol=1e-12,
            max_iter=100,
            use_backup=False
        )
        
        # Set non-converged to NaN
        results[np.logical_not(converged)] = np.nan
        
        return results
    except Exception as e:
        print(f"  rapid-roots Brent failed: {e}")
        return np.full(len(params), np.nan)


def _benchmark_rapid_roots_bisect(func_dict: Dict, params: np.ndarray,
                                  a_bounds: np.ndarray, b_bounds: np.ndarray) -> np.ndarray:
    """Run rapid-roots Bisection on all samples."""
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
        
        # Set non-converged to NaN
        results[np.logical_not(converged)] = np.nan
        
        return results
    except Exception as e:
        print(f"  rapid-roots Bisection failed: {e}")
        return np.full(len(params), np.nan)


def _benchmark_rapid_roots_newton(func_dict: Dict, params: np.ndarray,
                                  x0_values: np.ndarray) -> np.ndarray:
    """Run rapid-roots Newton on all samples."""
    func = func_dict['func']
    func_prime = func_dict['func_prime']
    
    try:
        results, iters, converged = RootSolvers.get_root(
            func=func,
            x0=x0_values,
            func_prime=func_prime,
            func_params=params,
            main_solver='newton',
            tol=1e-12,
            max_iter=100,
            use_backup=False
        )
        
        # Set non-converged to NaN
        results[np.logical_not(converged)] = np.nan
        return results
    except Exception as e:
        print(f"  rapid-roots Newton failed: {e}")
        return np.full(len(params), np.nan)
