"""
Benchmark code to measure solver throughput

Author: Cian Quezon
"""
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import numpy.typing as npt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared._run_benchmark import (
    _benchmark_rapid_roots_bisect,
    _benchmark_rapid_roots_newton,
    _benchmark_rapid_roots_brent
)
from shared._generate import generate_test_samples
from shared.functions import (
    get_function_by_name
)
def run_benchmark_functions_throughput(problem_size: list[int] = [10_000, 100_000, 1_000_000, 10_000_000], seed: int = 42,
                                       n_samples: int = 10, n_warmup: int = 3):
    """
    Benchmark rapid-roots throughput across multiple problem sizes.
    
    Tests Brent, Bisection, and Newton-Raphson methods on the Lambert W
    equation with varying problem sizes. Each benchmark runs multiple times
    with warmup to measure steady-state performance.
    
    Parameters
    ----------
    problem_size : list[int], optional
        Problem sizes to benchmark. Default: [10K, 100K, 1M, 10M]
    seed : int, default=42
        Random seed for reproducible test data generation
    n_samples : int, default=10
        Total benchmark runs per problem size
    n_warmup : int, default=3
        Number of initial runs to discard as JIT warmup
    
    Returns
    -------
    dict
        Nested dictionary with structure:
        {
            problem_size_1: {
                'newton': {'median_time': float, 'throughput': float, 'n_problem_size': int},
                'brent': {'median_time': float, 'throughput': float, 'n_problem_size': int},
                'bisect': {'median_time': float, 'throughput': float, 'n_problem_size': int}
            },
            problem_size_2: {...},
            ...
        }
    """
    func_dict = get_function_by_name(name='lambert_w_equation')

    results = {}
    for n_problem_samples in problem_size:
        
        params, a_bounds, b_bounds = generate_test_samples(func_dict=func_dict, n_samples=n_problem_samples, seed=seed)
        x0_values = (a_bounds + b_bounds)/2.0
        
        newton_results = _benchmark_rapid_roots_newton_throughput(func_dict=func_dict, params=params, x0_values=x0_values, n_problem_samples=n_problem_samples, 
                                                                  n_samples=n_samples, n_warmup=n_warmup)
        brent_results = _benchmark_rapid_roots_brent_throughput(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds, n_problem_samples=n_problem_samples, 
                                                                n_samples=n_samples, n_warmup=n_warmup)
        bisect_results = _benchmark_rapid_roots_bisect_throughput(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds, n_problem_samples=n_problem_samples,
                                                 n_samples=n_samples, n_warmup=n_warmup)
        results[n_problem_samples] = {
            'newton' : {**newton_results},
            'brent' : {**brent_results},
            'bisect': {**bisect_results}
        }
    
    return results

def _benchmark_rapid_roots_newton_throughput(func_dict: Dict, params: npt.NDArray, x0_values: npt.NDArray, n_problem_samples: int, n_samples: int = 10,
                                             n_warmup: int = 3):
    """
    Benchmark rapid-roots Newton throughput with JIT warmup.
    
    Runs the Newton solver multiple times, discards the first n_warmup runs
    (JIT compilation warmup), and returns the median time and throughput
    of the remaining runs.
    
    Parameters
    ----------
    func_dict : Dict
        Function dictionary containing 'func' (JIT function) and metadata
    params : ndarray, shape (n_problem_samples, n_params)
        Parameters for each root-finding problem
    a_bounds : ndarray, shape (n_problem_samples,)
        Lower bracket bounds for each problem
    b_bounds : ndarray, shape (n_problem_samples,)
        Upper bracket bounds for each problem
    n_problem_samples : int
        Number of problems being solved (for throughput calculation)
    n_samples : int, default=10
        Total benchmark runs
    n_warmup : int, default=3
        Number of initial runs to discard as warmup
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'median_time': Median time (seconds) of steady-state runs
        - 'throughput': Problems solved per second
        - 'n_problem_size': Number of problems (same as n_problem_samples)
    """
    times = []
    
    for i in range(n_samples):
        start = time.perf_counter()
        _ = _benchmark_rapid_roots_newton(func_dict=func_dict, params=params, x0_values=x0_values)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # ignore 3 runs as caching and warmup for jit
    steady_test = times[n_warmup:]
    steady_state_median_time = np.median(steady_test)
    steady_state_throughput = n_problem_samples / steady_state_median_time

    return {
        'median_time': steady_state_median_time,
        'throughput': steady_state_throughput,
        'n_problem_size': n_problem_samples
    }

def _benchmark_rapid_roots_brent_throughput(func_dict: Dict, params: npt.NDArray, a_bounds: npt.NDArray, b_bounds:npt.ArrayLike, n_problem_samples: int,
                                            n_samples: int = 10, n_warmup: int = 3):
    """
    Benchmark rapid-roots Brent throughput with JIT warmup.
    
    Runs the Brent solver multiple times, discards the first n_warmup runs
    (JIT compilation warmup), and returns the median time and throughput
    of the remaining runs.
    
    Parameters
    ----------
    func_dict : Dict
        Function dictionary containing 'func' (JIT function) and metadata
    params : ndarray, shape (n_problem_samples, n_params)
        Parameters for each root-finding problem
    a_bounds : ndarray, shape (n_problem_samples,)
        Lower bracket bounds for each problem
    b_bounds : ndarray, shape (n_problem_samples,)
        Upper bracket bounds for each problem
    n_problem_samples : int
        Number of problems being solved (for throughput calculation)
    n_samples : int, default=10
        Total benchmark runs
    n_warmup : int, default=3
        Number of initial runs to discard as warmup
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'median_time': Median time (seconds) of steady-state runs
        - 'throughput': Problems solved per second
        - 'n_problem_size': Number of problems (same as n_problem_samples)
    """
    
    times = []

    for i in range(n_samples):
        start = time.perf_counter()
        _ = _benchmark_rapid_roots_brent(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # ignore 3 runs as caching and warmup for jit
    steady_test = times[n_warmup:]
    steady_state_median_time = np.median(steady_test)
    steady_state_throughput = n_problem_samples / steady_state_median_time

    return {
        'median_time': steady_state_median_time,
        'throughput': steady_state_throughput,
        'n_problem_size': n_problem_samples
    }

def _benchmark_rapid_roots_bisect_throughput(func_dict: Dict, params: npt.NDArray, a_bounds: npt.NDArray, b_bounds: npt.NDArray, n_problem_samples: int,
                                             n_samples: int = 10, n_warmup: int = 3):
    """
    Benchmark rapid-roots Bisection throughput with JIT warmup.
    
    Runs the Bisection solver multiple times, discards the first n_warmup runs
    (JIT compilation warmup), and returns the median time and throughput
    of the remaining runs.
    
    Parameters
    ----------
    func_dict : Dict
        Function dictionary containing 'func' (JIT function) and metadata
    params : ndarray, shape (n_problem_samples, n_params)
        Parameters for each root-finding problem
    a_bounds : ndarray, shape (n_problem_samples,)
        Lower bracket bounds for each problem
    b_bounds : ndarray, shape (n_problem_samples,)
        Upper bracket bounds for each problem
    n_problem_samples : int
        Number of problems being solved (for throughput calculation)
    n_samples : int, default=10
        Total benchmark runs
    n_warmup : int, default=3
        Number of initial runs to discard as warmup
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'median_time': Median time (seconds) of steady-state runs
        - 'throughput': Problems solved per second
        - 'n_problem_size': Number of problems (same as n_problem_samples)
    """

    times = []
    
    for i in range(n_samples):
        start = time.perf_counter()
        _ = _benchmark_rapid_roots_bisect(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # ignore 3 runs as caching and warmup for jit
    steady_test = times[n_warmup:]
    steady_state_median_time = np.median(steady_test)
    steady_state_throughput = n_problem_samples / steady_state_median_time

    return {
        'median_time': steady_state_median_time,
        'throughput': steady_state_throughput,
        'n_problem_size': n_problem_samples
    }  

if __name__ == "__main__":
    run_benchmark_functions_throughput(problem_size=[10_000, 100_000, 1_000_000, 10_000_000], seed=42,
                                       n_samples=13, n_warmup=3)




