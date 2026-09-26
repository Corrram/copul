r"""
Random search for counterexamples and stress tests of conjectured inequalities.

The loops that research scripts keep re-implementing (draw a random
checkerboard, optionally make it SI, compare two measures, keep the worst
case) are provided by

* :func:`random_checkerboards` -- an iterator over diverse random
  ``BivCheckPi`` / ``BivCheckMin`` / ``BivCheckW`` copulas, optionally SI,
  SD, exchangeable, radially symmetric or filtered by a predicate;
* :func:`check_inequality` -- samples copulas, tracks the maximal violation of
  ``lhs <= rhs`` (or ``>=``, ``==``) and refines the worst checkerboard
  locally;
* :func:`find_counterexample` -- the same, returning a
  :class:`Counterexample` or ``None``.

Measure keys are evaluated with the exact checkerboard formulas of
:mod:`copul.optim.checkerboard_formulas` (fast, and independent of the
per-class methods); callables ``copula -> float`` can be used for anything
else.

Worked example
--------------
Is :math:`\xi\le|\rho|`?  (No -- a quick counterexample.)

>>> from copul.search import find_counterexample, check_inequality
>>> ce = find_counterexample("xi", lambda C: abs(C.spearmans_rho()), n_iter=500, rng=0)
>>> ce is not None
True

The Ansari--Rockel bound :math:`|\rho|\le M(\xi)` holds on SI checkerboards:

>>> from copul.regions import get
>>> reg = get("xi", "rho")
>>> rep = check_inequality(
...     lambda C: abs(C.spearmans_rho()) - float(reg.upper(C.chatterjees_xi())),
...     0.0, "<=", n_iter=100, condition="si", rng=0)
>>> rep.holds
True
"""

from copul.search.counterexamples import (
    Counterexample,
    InequalityReport,
    check_inequality,
    find_counterexample,
)
from copul.search.sampling import (
    random_checkerboards,
    random_mass_matrix,
    si_rearrangement,
)

__all__ = [
    "Counterexample",
    "InequalityReport",
    "check_inequality",
    "find_counterexample",
    "random_checkerboards",
    "random_mass_matrix",
    "si_rearrangement",
]
