"""
Docstring for benchmark.accuracy._calculations
"""

from typing import Dict

import numpy as np
import numpy.typing as npt


def calculate_error_metrics(
    scipy_results: npt.NDArray, rapid_roots_results: npt.NDArray
) -> Dict[str, float]:
    """
    Calculates error metrics between Scipy and rapid-roots.

    Parameters
    ----------
    scipy_results : ndarray
        Results from SciPy solver
    rapid_results : ndarray
        Results from rapid-roots solver

    Returns
    -------
    dict
        Dictionary containing:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - max_error: Maximum absolute error
        - max_error_percent: Maximum error percentage
        - mean_error_percent: Mean error percentage
        - n_valid: Number of valid comparisons
    """
    valid_mask = np.isfinite(scipy_results) & np.isfinite(rapid_roots_results)

    if not np.any(valid_mask):
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "max_error": np.nan,
            "max_error_percent": np.nan,
            "mean_error_percent": np.nan,
            "n_valid": 0,
            "abs_errors": np.array([]),
            "relative_errors": np.array([]),
        }

    scipy_valid = scipy_results[valid_mask]
    rapid_roots_valid = rapid_roots_results[valid_mask]

    abs_errors = np.abs(scipy_valid - rapid_roots_valid)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(abs_errors**2))
    max_error = np.max(abs_errors)

    with np.errstate(divide="ignore", invalid="ignore"):
        denominators = np.maximum(np.abs(scipy_valid), 1e-10)
        relative_errors = abs_errors / denominators * 100.0

        relative_errors = relative_errors[np.isfinite(relative_errors)]

        if len(relative_errors) > 0:
            max_error_percent = np.max(relative_errors)
            mean_error_percent = np.mean(relative_errors)

        else:
            max_error_percent = np.nan
            mean_error_percent = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "max_error_percent": max_error_percent,
        "mean_error_percent": mean_error_percent,
        "n_valid": np.sum(valid_mask),
        "abs_errors": abs_errors,
        "relative_errors": relative_errors,
    }
