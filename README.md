# Rapid-Roots

[![PyPI version](https://badge.fury.io/py/rapid-roots.svg)](https://badge.fury.io/py/rapid-roots)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

General parallel vectorised root solver using Numba for large data volumes. 

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
## Performance
![alt text](benchmark/generated/plots/method_throughput_line_plot.png)

<table style="width:100%; text-align:center;">
  <thead>
    <tr>
      <th>Samples</th>
      <th colspan="3">Throughput (Solves/sec)</th>
    </tr>
    <tr>
      <th></th>
      <th>Brent</th>
      <th>Newton</th>
      <th>Bisection</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>10K</td><td>19632</td><td>194212</td><td>Value 5</td></tr>
    <tr><td>100K</td><td>19632</td><td>194212</td><td>Value 5</td></tr>
    <tr><td>1M</td><td>19632</td><td>194212</td><td>Value 5</td></tr>
    <tr><td>10M</td><td>19632</td><td>194212</td><td>Value 5</td></tr>
  </tbody>
</table>


## Accuracy
![alt text](benchmark/generated/plots/error_distribution_boxplot.png)
