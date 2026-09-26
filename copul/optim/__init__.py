r"""
Exact optimisation of dependence measures over checkerboard copulas.

The toolbox replaces the ad-hoc ``cvxpy`` scripts that discretise
:math:`h=\partial_1C` by an *exact* parametrisation: the decision variable is
the mass matrix :math:`P` of a checkerboard copula
(:class:`~copul.checkerboard.biv_check_pi.BivCheckPi` for ``kind="pi"``,
``BivCheckMin``/``BivCheckW`` for ``"min"``/``"w"``), on which

* Spearman's :math:`\rho`, Blest's :math:`\nu`, Spearman's footrule
  :math:`\psi`, Gini's :math:`\gamma` and Blomqvist's :math:`\beta` are affine,
* Chatterjee's :math:`\xi` is a convex quadratic,
* Kendall's :math:`\tau` is an indefinite quadratic

(see :mod:`copul.optim.checkerboard_formulas`).  Every optimum is therefore an
honest copula whose measures are known in closed form -- traced boundaries are
*inner* approximations of exact regions that converge as the grid is refined.

``cvxpy`` is an optional dependency (``pip install copul[optim]``); it is
imported only when a problem is built or solved.

Worked example
--------------
Upper boundary of the :math:`(\xi,\rho)` region on a 24x24 grid, compared with
the exact bound of Ansari & Rockel:

>>> import numpy as np
>>> import copul.optim as co
>>> from copul.schur_order.bounds_from_xi import rho_max_given_xi
>>> prob = co.CheckerboardProblem(n=24)
>>> res = prob.maximize("rho", subject_to={"xi": ("<=", 0.3)})
>>> bool(res["rho"] <= rho_max_given_xi(0.3))
True
>>> trace = co.trace_boundary("xi", "rho", side="upper", n=24, n_points=8)
>>> bool(np.all(trace.ys <= [rho_max_given_xi(x) + 1e-9 for x in trace.xs]))
True

Shape constraints are exact linear conditions on :math:`P`:

>>> res = prob.maximize("nu", subject_to=[("rho", "==", 0.2), "si"])
>>> co.shape_violation(res.P, "si") < 1e-8
True
"""

from copul.optim._backend import installed_solvers, require_cvxpy
from copul.optim.boundary import BoundaryTrace, trace_boundary
from copul.optim.checkerboard_formulas import (
    Bilinear,
    QuadraticForm,
    RowGram,
    checkerboard_copula,
    is_feasible_mass_matrix,
    measure_form,
    measure_values,
)
from copul.optim.problem import (
    CheckerboardProblem,
    NonConvexError,
    OptimResult,
    balance,
    to_cvxpy,
)
from copul.optim.shapes import (
    available_shapes,
    satisfies_shape,
    shape_violation,
)

__all__ = [
    "Bilinear",
    "BoundaryTrace",
    "CheckerboardProblem",
    "NonConvexError",
    "OptimResult",
    "QuadraticForm",
    "RowGram",
    "available_shapes",
    "balance",
    "checkerboard_copula",
    "installed_solvers",
    "is_feasible_mass_matrix",
    "measure_form",
    "measure_values",
    "require_cvxpy",
    "satisfies_shape",
    "shape_violation",
    "to_cvxpy",
    "trace_boundary",
]
