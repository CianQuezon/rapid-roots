# Rapid-Roots

[![PyPI version](https://badge.fury.io/py/rapid-roots.svg)](https://badge.fury.io/py/rapid-roots)
[![Python 3.9+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

General parallel vectorised root solver using Numba for large data volumes. 

## Problem

Open source root-finding libraries like Scipy requires users to processes problems sequentially.

```python
# Sequential approach (Slow)
for i in range(1_000_000):
    root = scipy.optimize.brentq(func, a[i], b[i], args=(params[i],))
```

## Solution

rapid-roots processes all problems at once using vectorisation and parallelisation.

```python

# Vectorised rapid-roots approach 
roots, iters, converged = RootSolvers.get_root(
    func, a, b, func_params=func_params, main_solver='brent', use_backup=False
)

```

## Quick Start
```bash
pip install rapid-roots
```

```python
from rapid_roots.solvers import RootSolvers
from numba import njit
import numpy as np

@njit
def f(x, a):
    return x**2 - a

@njit
def f_prime(x, a):
    return 2*x


# Creates function parameter a in the equation.
params = np.full((10000, 1), 4.0)


# Creates a and b bounds for backup bracketed solvers
a = np.zeros(10000)
b = np.full(10000, 10.0)

# Creates x0 initial guess for main Newton solver
x0 = (a + b) / 2

# Vectorised solver implementation solves 10,000 problems at once.
roots, iters, converged = RootSolvers.get_root(
    func=f, a=a,  b=b, x0=x0, func_prime=f_prime, 
    func_params=params, main_solver='newton', 
    use_backup=True 
)

print(f"Solved {converged.sum()} problems") # Solved 10,000 problems
print(f"Mean root: {roots.mean()}")         # Mean root: 2.0
```

## API Reference

### `RootSolvers.get_root()`

Main interface for vectorized root finding.

```python
roots, iterations, converged = RootSolvers.get_root(
    func=func,              # Numba JIT-compiled function
    a=None,                 # Lowerbound brackets for bracketing methods
    b=None,                 # Upperbound brackets for bracketing methods
    x0=None,                # Initial guess for open methods (Newton)
    func_prime=None,        # Derivatives for open method (Newton)
    func_params=None,       # Parameters Array (n_problems, n_params)
    main_solver='brent',    # 'brent', 'bisection', or 'newton'
    tol=1e-12,              # Convergence tolerance
    max_iter=100,           # Maximum iteration
    use_backup=True,        # Enables automatic fallback
    backup_solvers=['newton', 'bisection']  # Fallback methods for unconverged problems
)
```
**Returns:**
- `roots`: ndArray - Found roots for each problem
- `iterations`: ndArray - Number of iterations used
- `converged`: ndArray (bool) - Convergence status for each problem

## Performance
![Throughput Scaling](https://github.com/CianQuezon/rapid-roots/blob/main/benchmark/generated/plots/method_throughput_line_plot.png?raw=true)

<table width="100%" align="center">
  <thead>
    <tr>
      <th align="center">Samples</th>
      <th colspan="3" align="center">Throughput (Solves/sec)</th>
    </tr>
    <tr>
      <th></th>
      <th align="center">Newton</th>
      <th align="center">Brent</th>
      <th align="center">Bisection</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">10K</td>
      <td align="center">22,951</td>
      <td align="center">19,632</td>
      <td align="center">23,219</td>
    </tr>
    <tr>
      <td align="center">100K</td>
      <td align="center">231,065</td>
      <td align="center">194,212</td>
      <td align="center">219,248</td>
    </tr>
    <tr>
      <td align="center">1M</td>
      <td align="center">2,187,503</td>
      <td align="center">1,787,564</td>
      <td align="center">1,566,471</td>
    </tr>
    <tr>
      <td align="center">10M</td>
      <td align="center">15,404,269</td>
      <td align="center">9,486,609</td>
      <td align="center">4,589,518</td>
    </tr>
  </tbody>
</table>

## Accuracy
![Error Distribution Boxplot](https://github.com/CianQuezon/rapid-roots/blob/main/benchmark/generated/plots/error_distribution_boxplot.png?raw=true)

Box plot compares the absolute error distributions against Scipy for Brent, Bisection and Newton implementation across different function categories.

Newton shows the lowest median error with occasional ouliers in challenging categories. Brent is slightly higher in median errors with a wider interquantile range. Meanwhile Bisection displays the largest spread and highest typical error.

Results show errors approach machine precision for all methods against scipy.

## Installation

### Requirements

- Python ≥ 3.8
- Numpy ≥ 1.20.0
- Numba ≥ 0.56.0

### Install from PyPI
```bash
pip install rapid-roots
```

### Install from source 
```bash
git clone https://github.com/CianQuezon/rapid-roots.git
cd rapid-roots
pip install -e .
```


## License

This project is licensed under the MIT License - see the LICENSE file for more details.

## Acknowledgements

- Built with [Numba](https://numba.pydata.org/)
- Validated against [Scipy](https://scipy.org/) reference implementations 
