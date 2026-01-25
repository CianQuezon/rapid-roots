"""
Shared types and maps for solvers to avoid circular import errors.

Author: Cian Quezon
"""

from rapid_roots.solvers._enums import SolverName
from rapid_roots.solvers._solvers import (
    BisectionSolver,
    BrentSolver,
    NewtonRaphsonSolver,
)

SolverMap = {
    SolverName.NEWTON: NewtonRaphsonSolver,
    SolverName.BRENT: BrentSolver,
    SolverName.BISECTION: BisectionSolver,
}
