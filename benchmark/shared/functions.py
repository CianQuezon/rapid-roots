"""
Generated 25-Function Benchmark Suite for Root-Finding

Carefully selected functions covering all important numerical properties:
- Polynomials (6): Degrees 2-5, including ill-conditioned cases
- Exponential/Logarithmic (7): Pure and mixed forms
- Trigonometric (6): Standard and hyperbolic
- Pathological (6): Edge cases and challenging problems

Each function has both JIT-compiled (@njit) and pure Python versions
for compatibility with both rapid-roots and SciPy.

Author: Cian Quezon
"""

from numba import njit
import numpy as np
from typing import Callable, Dict, List, Tuple


# ============================================================================
# CATEGORY 1: POLYNOMIALS (6 functions)
# ============================================================================

# -----------------------------------------------------------------------------
# 1. Simple Quadratic
# -----------------------------------------------------------------------------


@njit
def poly_quad_simple(x, a, b, c):
    """ax² + bx + c = 0"""
    return a * x**2 + b * x + c


@njit
def poly_quad_simple_prime(x, a, b, c):
    return 2.0 * a * x + b


def poly_quad_simple_scipy(x, a, b, c):
    return a * x**2 + b * x + c


def poly_quad_simple_prime_scipy(x, a, b, c):
    return 2.0 * a * x + b


# -----------------------------------------------------------------------------
# 3. Biquadratic Quartic
# -----------------------------------------------------------------------------


@njit
def poly_quartic_biquad(x, a, b):
    """x⁴ + ax² + b = 0"""
    return x**4 + a * x**2 + b


@njit
def poly_quartic_biquad_prime(x, a, b):
    return 4.0 * x**3 + 2.0 * a * x


def poly_quartic_biquad_scipy(x, a, b):
    return x**4 + a * x**2 + b


def poly_quartic_biquad_prime_scipy(x, a, b):
    return 4.0 * x**3 + 2.0 * a * x


# -----------------------------------------------------------------------------
# 4. Simple Quintic
# -----------------------------------------------------------------------------


@njit
def poly_quintic_simple(x, a, b):
    """x⁵ + ax + b = 0"""
    return x**5 + a * x + b


@njit
def poly_quintic_simple_prime(x, a, b):
    return 5.0 * x**4 + a


def poly_quintic_simple_scipy(x, a, b):
    return x**5 + a * x + b


def poly_quintic_simple_prime_scipy(x, a, b):
    return 5.0 * x**4 + a


# -----------------------------------------------------------------------------
# 6. Chebyshev-like
# -----------------------------------------------------------------------------


@njit
def poly_chebyshev_2(x, a):
    """T₂(x) - a = 0, where T₂(x) = 2x² - 1"""
    return 2.0 * x**2 - 1.0 - a


@njit
def poly_chebyshev_2_prime(x, a):
    return 4.0 * x


def poly_chebyshev_2_scipy(x, a):
    return 2.0 * x**2 - 1.0 - a


def poly_chebyshev_2_prime_scipy(x, a):
    return 4.0 * x


# ============================================================================
# CATEGORY 2: EXPONENTIAL & LOGARITHMIC (7 functions)
# ============================================================================

# -----------------------------------------------------------------------------
# 7. Simple Exponential
# -----------------------------------------------------------------------------


@njit
def exp_simple(x, a):
    """e^x - a = 0"""
    return np.exp(x) - a


@njit
def exp_simple_prime(x, a):
    return np.exp(x)


def exp_simple_scipy(x, a):
    return np.exp(x) - a


def exp_simple_prime_scipy(x, a):
    return np.exp(x)


# -----------------------------------------------------------------------------
# 8. Exponential minus Linear
# -----------------------------------------------------------------------------


@njit
def exp_linear(x, a, b):
    """e^x - ax - b = 0"""
    return np.exp(x) - a * x - b


@njit
def exp_linear_prime(x, a, b):
    return np.exp(x) - a


def exp_linear_scipy(x, a, b):
    return np.exp(x) - a * x - b


def exp_linear_prime_scipy(x, a, b):
    return np.exp(x) - a


# -----------------------------------------------------------------------------
# 9. Exponential minus Quadratic
# -----------------------------------------------------------------------------


@njit
def exp_quadratic(x, a, b):
    """e^x - ax² - b = 0"""
    return np.exp(x) - a * x**2 - b


@njit
def exp_quadratic_prime(x, a, b):
    return np.exp(x) - 2.0 * a * x


def exp_quadratic_scipy(x, a, b):
    return np.exp(x) - a * x**2 - b


def exp_quadratic_prime_scipy(x, a, b):
    return np.exp(x) - 2.0 * a * x


# -----------------------------------------------------------------------------
# 10. x times Exponential
# -----------------------------------------------------------------------------


@njit
def exp_x_times_x(x, a, b):
    """x*e^x - ax - b = 0"""
    if x > 700.0:
        x_safe = 700.0
    elif x < -700.0:
        x_safe = -700.0
    else:
        x_safe = x
    return x * np.exp(x_safe) - a * x_safe - b


@njit
def exp_x_times_x_prime(x, a, b):
    if x > 700.0:
        x_safe = 700.0
    elif x < -700.0:
        x_safe = -700.0
    else:
        x_safe = x
    return np.exp(x_safe) + x_safe * np.exp(x_safe) - a


def exp_x_times_x_scipy(x, a, b):
    if x > 700.0:
        x_safe = 700.0
    elif x < -700.0:
        x_safe = -700.0
    else:
        x_safe = x
    return x * np.exp(x_safe) - a * x_safe - b


def exp_x_times_x_prime_scipy(x, a, b):
    if x > 700.0:
        x_safe = 700.0
    elif x < -700.0:
        x_safe = -700.0
    else:
        x_safe = x
    return np.exp(x_safe) + x_safe * np.exp(x_safe) - a


# -----------------------------------------------------------------------------
# 11. Simple Logarithm
# -----------------------------------------------------------------------------


@njit
def log_simple(x, a):
    """ln(x) - a = 0"""
    if x <= 0:
        return np.inf
    return np.log(x) - a


@njit
def log_simple_prime(x, a):
    if x <= 0:
        return np.inf
    return 1.0 / x


def log_simple_scipy(x, a):
    if x <= 0:
        return np.inf
    return np.log(x) - a


def log_simple_prime_scipy(x, a):
    if x <= 0:
        return np.inf
    return 1.0 / x


# -----------------------------------------------------------------------------
# 12. Logarithm minus Linear
# -----------------------------------------------------------------------------


@njit
def log_linear(x, a, b):
    """ln(x) - ax - b = 0"""
    if x <= 0:
        return np.inf
    return np.log(x) - a * x - b


@njit
def log_linear_prime(x, a, b):
    if x <= 0:
        return np.inf
    return 1.0 / x - a


def log_linear_scipy(x, a, b):
    if x <= 0:
        return np.inf
    return np.log(x) - a * x - b


def log_linear_prime_scipy(x, a, b):
    if x <= 0:
        return np.inf
    return 1.0 / x - a


# -----------------------------------------------------------------------------
# 13. Lambert W Function (x*e^x = a)
# -----------------------------------------------------------------------------


@njit
def lambert_w_equation(x, a):
    """x*e^x - a = 0"""
    return x * np.exp(x) - a


@njit
def lambert_w_equation_prime(x, a):
    return np.exp(x) + x * np.exp(x)


def lambert_w_equation_scipy(x, a):
    return x * np.exp(x) - a


def lambert_w_equation_prime_scipy(x, a):
    return np.exp(x) + x * np.exp(x)


# ============================================================================
# CATEGORY 3: TRIGONOMETRIC (6 functions)
# ============================================================================

# -----------------------------------------------------------------------------
# 14. Simple Sine
# -----------------------------------------------------------------------------


@njit
def trig_sin_simple(x, a):
    """sin(x) - a = 0"""
    return np.sin(x) - a


@njit
def trig_sin_simple_prime(x, a):
    return np.cos(x)


def trig_sin_simple_scipy(x, a):
    return np.sin(x) - a


def trig_sin_simple_prime_scipy(x, a):
    return np.cos(x)


# -----------------------------------------------------------------------------
# 15. Sine minus Linear
# -----------------------------------------------------------------------------


@njit
def trig_sin_linear(x, a, b):
    """sin(x) - ax - b = 0"""
    return np.sin(x) - a * x - b


@njit
def trig_sin_linear_prime(x, a, b):
    return np.cos(x) - a


def trig_sin_linear_scipy(x, a, b):
    return np.sin(x) - a * x - b


def trig_sin_linear_prime_scipy(x, a, b):
    return np.cos(x) - a


# -----------------------------------------------------------------------------
# 16. Simple Cosine
# -----------------------------------------------------------------------------


@njit
def trig_cos_simple(x, a):
    """cos(x) - a = 0"""
    return np.cos(x) - a


@njit
def trig_cos_simple_prime(x, a):
    return -np.sin(x)


def trig_cos_simple_scipy(x, a):
    return np.cos(x) - a


def trig_cos_simple_prime_scipy(x, a):
    return -np.sin(x)


# -----------------------------------------------------------------------------
# 17. Tangent minus Linear
# -----------------------------------------------------------------------------


@njit
def trig_tan_linear(x, a, b):
    """tan(x) - ax - b = 0"""
    return np.tan(x) - a * x - b


@njit
def trig_tan_linear_prime(x, a, b):
    return 1.0 / np.cos(x) ** 2 - a


def trig_tan_linear_scipy(x, a, b):
    return np.tan(x) - a * x - b


def trig_tan_linear_prime_scipy(x, a, b):
    return 1.0 / np.cos(x) ** 2 - a


# -----------------------------------------------------------------------------
# 18. Hyperbolic Sine
# -----------------------------------------------------------------------------


@njit
def trig_sinh_simple(x, a):
    """sinh(x) - a = 0"""
    return np.sinh(x) - a


@njit
def trig_sinh_simple_prime(x, a):
    return np.cosh(x)


def trig_sinh_simple_scipy(x, a):
    return np.sinh(x) - a


def trig_sinh_simple_prime_scipy(x, a):
    return np.cosh(x)


# -----------------------------------------------------------------------------
# 19. Sinh minus Scaled Cosh
# -----------------------------------------------------------------------------


@njit
def trig_sinh_cosh(x, a, b):
    """sinh(x) - a*cosh(x) - b = 0"""
    return np.sinh(x) - a * np.cosh(x) - b


@njit
def trig_sinh_cosh_prime(x, a, b):
    return np.cosh(x) - a * np.sinh(x)


def trig_sinh_cosh_scipy(x, a, b):
    return np.sinh(x) - a * np.cosh(x) - b


def trig_sinh_cosh_prime_scipy(x, a, b):
    return np.cosh(x) - a * np.sinh(x)


# ============================================================================
# CATEGORY 4: PATHOLOGICAL & CHALLENGING (6 functions)
# ============================================================================


# -----------------------------------------------------------------------------
# 21. Nearly Flat (Gaussian-like)
# -----------------------------------------------------------------------------


@njit
def path_nearly_flat_exp(x, a):
    """e^(-x²) - a (nearly flat for large |x|)"""
    if x > 27.0:
        x_safe = 27.0
    elif x < -27.0:
        x_safe = -27.0
    else:
        x_safe = x
    return np.exp(-(x_safe**2)) - a


@njit
def path_nearly_flat_exp_prime(x, a):
    if x > 27.0:
        x_safe = 27.0
    elif x < -27.0:
        x_safe = -27.0
    else:
        x_safe = x
    return -2.0 * x_safe * np.exp(-(x_safe**2))


def path_nearly_flat_exp_scipy(x, a):
    if x > 27.0:
        x_safe = 27.0
    elif x < -27.0:
        x_safe = -27.0
    else:
        x_safe = x
    return np.exp(-(x_safe**2)) - a


def path_nearly_flat_exp_prime_scipy(x, a):
    if x > 27.0:
        x_safe = 27.0
    elif x < -27.0:
        x_safe = -27.0
    else:
        x_safe = x
    return -2.0 * x_safe * np.exp(-(x_safe**2))


# -----------------------------------------------------------------------------
# 23. Multi-scale Exponential
# -----------------------------------------------------------------------------


@njit
def path_multiscale_exp(x, a, b):
    """e^x + e^(-x) - ax - b (multiple scales)"""
    return np.exp(x) + np.exp(-x) - a * x - b


@njit
def path_multiscale_exp_prime(x, a, b):
    return np.exp(x) - np.exp(-x) - a


def path_multiscale_exp_scipy(x, a, b):
    return np.exp(x) + np.exp(-x) - a * x - b


def path_multiscale_exp_prime_scipy(x, a, b):
    return np.exp(x) - np.exp(-x) - a


# -----------------------------------------------------------------------------
# 24. Exponential times Sine
# -----------------------------------------------------------------------------


@njit
def path_exp_sin(x, a, b):
    """e^x * sin(x) - ax - b (oscillatory with growth)"""
    return np.exp(x) * np.sin(x) - a * x - b


@njit
def path_exp_sin_prime(x, a, b):
    return np.exp(x) * (np.sin(x) + np.cos(x)) - a


def path_exp_sin_scipy(x, a, b):
    return np.exp(x) * np.sin(x) - a * x - b


def path_exp_sin_prime_scipy(x, a, b):
    return np.exp(x) * (np.sin(x) + np.cos(x)) - a


# -----------------------------------------------------------------------------
# 25. Steep Derivative
# -----------------------------------------------------------------------------


@njit
def path_steep_derivative(x, a, b):
    """x¹⁰ - ax - b (very steep near root)"""
    return x**10 - a * x - b


@njit
def path_steep_derivative_prime(x, a, b):
    return 10.0 * x**9 - a


def path_steep_derivative_scipy(x, a, b):
    return x**10 - a * x - b


def path_steep_derivative_prime_scipy(x, a, b):
    return 10.0 * x**9 - a


# ============================================================================
# FUNCTION REGISTRY
# ============================================================================

FUNCTIONS_LIST = [
    # POLYNOMIALS (6)
    {
        "name": "poly_quad_simple",
        "func": poly_quad_simple,
        "func_prime": poly_quad_simple_prime,
        "func_scipy": poly_quad_simple_scipy,
        "func_prime_scipy": poly_quad_simple_prime_scipy,
        "params_range": [(0.5, 2.0), (-10.0, -2.0), (5.0, 20.0)],
        "bounds": (-10.0, 10.0),
        "x0": 0.0,
        "difficulty": "easy",
        "category": "polynomial",
        "description": "Standard quadratic ax² + bx + c",
    },
    {
        "name": "poly_quartic_biquad",
        "func": poly_quartic_biquad,
        "func_prime": poly_quartic_biquad_prime,
        "func_scipy": poly_quartic_biquad_scipy,
        "func_prime_scipy": poly_quartic_biquad_prime_scipy,
        "params_range": [(-7.0, -3.0), (1.0, 6.0)],
        "bounds": (0.0, 2.0),
        "x0": 1.5,
        "difficulty": "medium",
        "category": "polynomial",
        "description": "Biquadratic x⁴ + ax² + b",
    },
    {
        "name": "poly_quintic_simple",
        "func": poly_quintic_simple,
        "func_prime": poly_quintic_simple_prime,
        "func_scipy": poly_quintic_simple_scipy,
        "func_prime_scipy": poly_quintic_simple_prime_scipy,
        "params_range": [(-5.0, 5.0), (-10.0, 10.0)],
        "bounds": (-3.0, 3.0),
        "x0": 1.0,
        "difficulty": "medium",
        "category": "polynomial",
        "description": "Quintic x⁵ + ax + b",
    },
    {
        "name": "poly_chebyshev_2",
        "func": poly_chebyshev_2,
        "func_prime": poly_chebyshev_2_prime,
        "func_scipy": poly_chebyshev_2_scipy,
        "func_prime_scipy": poly_chebyshev_2_prime_scipy,
        "params_range": [(-0.9, 0.9)],
        "bounds": (0.0, 1.5),
        "x0": 0.7,
        "difficulty": "easy",
        "category": "polynomial",
        "description": "Chebyshev T₂(x) = 2x² - 1",
    },
    # EXPONENTIAL & LOGARITHMIC (7)
    {
        "name": "exp_simple",
        "func": exp_simple,
        "func_prime": exp_simple_prime,
        "func_scipy": exp_simple_scipy,
        "func_prime_scipy": exp_simple_prime_scipy,
        "params_range": [(0.5, 10.0)],
        "bounds": (-2.0, 5.0),
        "x0": 1.0,
        "difficulty": "easy",
        "category": "exponential",
        "description": "Simple exponential e^x - a",
    },
    {
        "name": "exp_linear",
        "func": exp_linear,
        "func_prime": exp_linear_prime,
        "func_scipy": exp_linear_scipy,
        "func_prime_scipy": exp_linear_prime_scipy,
        "params_range": [(0.5, 2.0), (1.0, 5.0)],
        "bounds": (-2.0, 5.0),
        "x0": 1.0,
        "difficulty": "easy",
        "category": "exponential",
        "description": "Exponential minus linear e^x - ax - b",
    },
    {
        "name": "exp_quadratic",
        "func": exp_quadratic,
        "func_prime": exp_quadratic_prime,
        "func_scipy": exp_quadratic_scipy,
        "func_prime_scipy": exp_quadratic_prime_scipy,
        "params_range": [(0.1, 0.5), (1.0, 10.0)],
        "bounds": (-2.0, 5.0),
        "x0": 2.0,
        "difficulty": "medium",
        "category": "exponential",
        "description": "Exponential minus quadratic e^x - ax² - b",
    },
    {
        "name": "exp_x_times_x",
        "func": exp_x_times_x,
        "func_prime": exp_x_times_x_prime,
        "func_scipy": exp_x_times_x_scipy,
        "func_prime_scipy": exp_x_times_x_prime_scipy,
        "params_range": [(1.0, 5.0), (1.0, 10.0)],
        "bounds": (-1.0, 2.0),
        "x0": 0.5,
        "difficulty": "medium",
        "category": "exponential",
        "description": "x times exponential x*e^x - ax - b",
    },
    {
        "name": "log_simple",
        "func": log_simple,
        "func_prime": log_simple_prime,
        "func_scipy": log_simple_scipy,
        "func_prime_scipy": log_simple_prime_scipy,
        "params_range": [(0.0, 3.0)],
        "bounds": (0.1, 10.0),
        "x0": 2.0,
        "difficulty": "easy",
        "category": "logarithmic",
        "description": "Simple logarithm ln(x) - a",
    },
    {
        "name": "log_linear",
        "func": log_linear,
        "func_prime": log_linear_prime,
        "func_scipy": log_linear_scipy,
        "func_prime_scipy": log_linear_prime_scipy,
        "params_range": [(0.1, 1.0), (-2.0, 2.0)],
        "bounds": (0.1, 10.0),
        "x0": 2.0,
        "difficulty": "easy",
        "category": "logarithmic",
        "description": "Log minus linear ln(x) - ax - b",
    },
    {
        "name": "lambert_w_equation",
        "func": lambert_w_equation,
        "func_prime": lambert_w_equation_prime,
        "func_scipy": lambert_w_equation_scipy,
        "func_prime_scipy": lambert_w_equation_prime_scipy,
        "params_range": [(1.0, 10.0)],
        "bounds": (-0.5, 3.0),
        "x0": 1.0,
        "difficulty": "medium",
        "category": "mixed",
        "description": "Lambert W function x*e^x - a",
    },
    # TRIGONOMETRIC (6)
    {
        "name": "trig_sin_simple",
        "func": trig_sin_simple,
        "func_prime": trig_sin_simple_prime,
        "func_scipy": trig_sin_simple_scipy,
        "func_prime_scipy": trig_sin_simple_prime_scipy,
        "params_range": [(0.3, 0.7)],
        "bounds": (0.2, 2.0),
        "x0": 1.0,
        "difficulty": "easy",
        "category": "trigonometric",
        "description": "Simple sine sin(x) - a",
    },
    {
        "name": "trig_sin_linear",
        "func": trig_sin_linear,
        "func_prime": trig_sin_linear_prime,
        "func_scipy": trig_sin_linear_scipy,
        "func_prime_scipy": trig_sin_linear_prime_scipy,
        "params_range": [(0.1, 0.5), (-0.5, 0.5)],
        "bounds": (-np.pi, np.pi),
        "x0": 1.0,
        "difficulty": "easy",
        "category": "trigonometric",
        "description": "Sine minus linear sin(x) - ax - b",
    },
    {
        "name": "trig_cos_simple",
        "func": trig_cos_simple,
        "func_prime": trig_cos_simple_prime,
        "func_scipy": trig_cos_simple_scipy,
        "func_prime_scipy": trig_cos_simple_prime_scipy,
        "params_range": [(0.0, 1.0)],
        "bounds": (0.0, np.pi),
        "x0": np.pi / 4,
        "difficulty": "easy",
        "category": "trigonometric",
        "description": "Simple cosine cos(x) - a",
    },
    {
        "name": "trig_tan_linear",
        "func": trig_tan_linear,
        "func_prime": trig_tan_linear_prime,
        "func_scipy": trig_tan_linear_scipy,
        "func_prime_scipy": trig_tan_linear_prime_scipy,
        "params_range": [(0.5, 1.5), (0.0, 1.0)],
        "bounds": (-1.4, 1.4),
        "x0": 0.5,
        "difficulty": "hard",
        "category": "trigonometric",
        "description": "Tangent minus linear tan(x) - ax - b",
    },
    {
        "name": "trig_sinh_simple",
        "func": trig_sinh_simple,
        "func_prime": trig_sinh_simple_prime,
        "func_scipy": trig_sinh_simple_scipy,
        "func_prime_scipy": trig_sinh_simple_prime_scipy,
        "params_range": [(0.5, 3.0)],
        "bounds": (-3.0, 3.0),
        "x0": 1.0,
        "difficulty": "easy",
        "category": "hyperbolic",
        "description": "Hyperbolic sine sinh(x) - a",
    },
    {
        "name": "trig_sinh_cosh",
        "func": trig_sinh_cosh,
        "func_prime": trig_sinh_cosh_prime,
        "func_scipy": trig_sinh_cosh_scipy,
        "func_prime_scipy": trig_sinh_cosh_prime_scipy,
        "params_range": [(0.5, 1.5), (0.0, 2.0)],
        "bounds": (-3.0, 3.0),
        "x0": 1.0,
        "difficulty": "medium",
        "category": "hyperbolic",
        "description": "sinh(x) - a*cosh(x) - b",
    },
    {
        "name": "path_nearly_flat_exp",
        "func": path_nearly_flat_exp,
        "func_prime": path_nearly_flat_exp_prime,
        "func_scipy": path_nearly_flat_exp_scipy,
        "func_prime_scipy": path_nearly_flat_exp_prime_scipy,
        "params_range": [(0.3, 0.9)],
        "bounds": (-3.0, 1.0),
        "x0": -1.0,
        "difficulty": "hard",
        "category": "pathological",
        "description": "Nearly flat Gaussian e^(-x²) - a",
    },
    {
        "name": "path_multiscale_exp",
        "func": path_multiscale_exp,
        "func_prime": path_multiscale_exp_prime,
        "func_scipy": path_multiscale_exp_scipy,
        "func_prime_scipy": path_multiscale_exp_prime_scipy,
        "params_range": [(2.0, 4.0), (0.0, 5.0)],
        "bounds": (-2.0, 2.0),
        "x0": 0.0,
        "difficulty": "medium",
        "category": "pathological",
        "description": "Multi-scale e^x + e^(-x) - ax - b",
    },
    {
        "name": "path_exp_sin",
        "func": path_exp_sin,
        "func_prime": path_exp_sin_prime,
        "func_scipy": path_exp_sin_scipy,
        "func_prime_scipy": path_exp_sin_prime_scipy,
        "params_range": [(0.5, 1.5), (0.0, 2.0)],
        "bounds": (0.0, 3.0),
        "x0": 1.5,
        "difficulty": "hard",
        "category": "pathological",
        "description": "Exponential growth with oscillation e^x*sin(x)",
    },
    {
        "name": "path_steep_derivative",
        "func": path_steep_derivative,
        "func_prime": path_steep_derivative_prime,
        "func_scipy": path_steep_derivative_scipy,
        "func_prime_scipy": path_steep_derivative_prime_scipy,
        "params_range": [(5.0, 15.0), (50.0, 200.0)],
        "bounds": (0.5, 2.0),
        "x0": 1.5,
        "difficulty": "hard",
        "category": "pathological",
        "description": "Very steep derivative x¹⁰ - ax - b",
    },
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_function_by_name(name: str) -> Dict:
    """Get function entry by name."""
    for func in FUNCTIONS_LIST:
        if func["name"] == name:
            return func
    raise ValueError(f"Function '{name}' not found in test suite")


def get_by_category(category: str) -> List[Dict]:
    """Get all functions in a specific category."""
    return [f for f in FUNCTIONS_LIST if f["category"] == category]


def get_by_difficulty(difficulty: str) -> List[Dict]:
    """Get all functions of a specific difficulty."""
    return [f for f in FUNCTIONS_LIST if f["difficulty"] == difficulty]


def get_test_summary() -> Dict:
    """Get summary statistics of the test suite."""
    categories = {}
    difficulties = {}

    for func in FUNCTIONS_LIST:
        cat = func["category"]
        diff = func["difficulty"]

        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

    return {
        "total": len(FUNCTIONS_LIST),
        "categories": categories,
        "difficulties": difficulties,
    }


if __name__ == "__main__":
    # Print summary
    summary = get_test_summary()

    print("\n" + "=" * 70)
    print("ACCURACY TEST FUNCTION SUITE")
    print("=" * 70)
    print(f"\nTotal functions: {summary['total']}")

    print(f"\nBy category:")
    for cat, count in sorted(summary["categories"].items()):
        print(f"  {cat:20}: {count:2}")

    print(f"\nBy difficulty:")
    for diff, count in sorted(summary["difficulties"].items()):
        print(f"  {diff:20}: {count:2}")

    print("\n" + "=" * 70)
    print("Function List:")
    print("=" * 70)

    for i, func in enumerate(FUNCTIONS_LIST, 1):
        print(
            f"{i:2}. {func['name']:25} [{func['difficulty']:6}] {func['description']}"
        )

    print("=" * 70 + "\n")
