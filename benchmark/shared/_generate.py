"""
Generates the required data for benchmarks
"""
from typing import Dict

import numpy as np
import numpy.typing as npt

def generate_test_samples(func_dict: Dict, n_samples: int = 50,
                          seed: int = 42) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Generate random test samples for functions
    
    Parameters
    ----------
    func_dict : dict
        Function dictionary from ACCURACY_TEST_FUNCTIONS
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    params : ndarray, shape (n_samples, n_params)
        Random parameter values
    a_bounds : ndarray, shape (n_samples,)
        Lower bounds for bracket methods
    b_bounds : ndarray, shape (n_samples,)
        Upper bounds for bracket methods
    """
    np.random.seed(seed)

    params_range = func_dict['params_range']
    n_params = len(params_range)

    params = np.zeros((n_samples, n_params))
    for i, (min_val, max_val) in enumerate(params_range):
        params[:, i] = np.random.uniform(min_val, max_val, n_samples)
    
    bounds = func_dict['bounds']
    if isinstance(bounds, tuple):
        a_bounds = np.full(n_samples, bounds[0])
        b_bounds = np.full(n_samples, bounds[1])
    else:
        a_bounds = np.array(bounds[0])
        b_bounds = np.array(bounds[1])

    return params, a_bounds, b_bounds