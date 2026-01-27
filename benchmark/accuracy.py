"""
Accuracy benchmark for rapid-roots against sci-py to ensure validity.

Author: Cian Quezon
"""
import time
import sys
from pathlib import Paths
from typing import Dict

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, bisect, newton

from rapid_roots.solvers.core import RootSolvers
from functions import (
    ACCURACY_TEST_FUNCTIONS,
    get_function_by_name
)

def calculate_error_metrics(scipy_results: npt.NDArray,
                            rapid_results: npt.NDArray) -> Dict[str, float]:
    """
    Calculates different error metrics between scipy and rapid-roots.
    """
    valid_results_mask = np.isfinite(scipy_results) & np.isfinite(rapid_results)

    if not np.any(valid_results_mask):
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'max_error': np.nan,
            'max_error_percent': np.nan,
            'mean_error_percent': np.nan,
            'n_valid': 0
        }
    
    scipy_valid_results = scipy_results[valid_results_mask]
    rapid_valid_results = rapid_results[valid_results_mask]

    abs_errors = np.abs(scipy_valid_results - rapid_valid_results)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(abs_errors**2))
    max_error = np.max(abs_errors)

    with np.errstate(divide='ignore', invalid='ignore'):
        
        denominators = np.maximum(np.abs(scipy_valid_results), 1e-20)
        relative_errors = abs_errors / denominators * 100.0

        relative_errors = relative_errors[np.isfinite(relative_errors)]

        if len(relative_errors) > 0:
            max_error_percent = np.max(relative_errors)
            mean_error_percent = np.mean(relative_errors)
        else:
            max_error_percent = np.nan
            mean_error_percent = np.nan

    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'max_error_percent': max_error_percent,
        'mean_error_percent': mean_error_percent,
        'n_valid': np.sum(valid_results_mask)
    }

    