"""
Benchmark code to measure throughput

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
def run_benchmark_functions_throughput(problem_size: list[int] = [10_000, 100_000, 1_000_000, 10_000_000], seed: int = 42):
    """
    
    
    
    """
    func_dict = get_function_by_name(name='lambert_w_equation')

    results = {}
    for n_samples in problem_size:
        
        params, a_bounds, b_bounds = generate_test_samples(func_dict=func_dict, n_samples=n_samples, seed=42)
        x0_values = (a_bounds + b_bounds)/2.0
        
        newton_results = _benchmark_rapid_roots_newton_throughput(func_dict=func_dict, params=params, x0_values=x0_values, problem_sample=n_samples, n_samples=10)
        results['newton'] = {
            **newton_results
        }
        
        brent_results = _benchmark_rapid_roots_brent_throughput(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds, problem_sample=n_samples, 
                                                                n_samples=10)
        results['brent'] = {
            **brent_results
        }

        print(results)

def _benchmark_rapid_roots_newton_throughput(func_dict: Dict, params: npt.NDArray, x0_values: npt.NDArray, problem_sample: int, n_samples: int = 10):
    """
    Docstring for _benchmark_rapid_roots_newton_throughput
    
    :param func_dict: Description
    :type func_dict: Dict
    :param params: Description
    :type params: npt.NDArray
    :param x0_values: Description
    :type x0_values: npt.NDArray
    :param n_samples: Description
    :type n_samples: int
    """
    times = []
    
    for i in range(n_samples):
        start = time.perf_counter()
        _ = _benchmark_rapid_roots_newton(func_dict=func_dict, params=params, x0_values=x0_values)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # ignore 3 runs as caching and warmup for jit
    steady_test = times[3:]
    steady_state_median_time = np.median(steady_test)
    steady_state_throughput = problem_sample / steady_state_median_time

    return {
        'median_time': steady_state_median_time,
        'throughput': steady_state_throughput,
        'n_problem_size': problem_sample
    }

def _benchmark_rapid_roots_brent_throughput(func_dict: Dict, params: npt.NDArray, a_bounds: npt.NDArray, b_bounds:npt.ArrayLike, problem_sample: int,
                                            n_samples: int = 10,):
    """
    Docstring for _benchmark_rapid_roots_brent_throughput
    
    :param func_dict: Description
    :type func_dict: Dict
    :param parame: Description
    :type parame: npt.NDArray
    :param x0_values: Description
    :type x0_values: npt.NDArray
    :param n_samples: Description
    :type n_samples: int
    """
    
    times = []

    for i in range(n_samples):
        start = time.perf_counter()
        _ = _benchmark_rapid_roots_brent(func_dict=func_dict, params=params, a_bounds=a_bounds, b_bounds=b_bounds)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # ignore 3 runs as caching and warmup for jit
    steady_test = times[3:]
    steady_state_median_time = np.median(steady_test)
    steady_state_throughput = problem_sample / steady_state_median_time

    return {
        'median_time': steady_state_median_time,
        'throughput': steady_state_throughput,
        'n_problem_size': problem_sample
    }


if __name__ == "__main__":
    run_benchmark_functions_throughput()




