Research workflows
==================

This page collects the typical workflows around *exact regions* between
dependence measures: looking up and plotting known regions, tracing
boundaries numerically with exact optimisation over checkerboard copulas, and
searching for counterexamples to conjectured inequalities.

Exact regions
-------------

:mod:`copul.regions` is a registry of proven exact regions

.. math::

   \mathcal{R}(\kappa_1,\kappa_2) = \{(\kappa_1(C), \kappa_2(C)) : C \text{ bivariate copula}\}

(or the analogue for a subclass such as SI copulas), each with closed-form
lower and upper boundaries, key points, boundary-attaining copula families, a
reference and the source file in the repository. Every registered region is
validated numerically in the test suite (boundary copulas attain the boundary,
random checkerboards lie inside, optimal checkerboards approach the boundary
from inside).

.. code-block:: python

   import copul as cp

   cp.regions.available()                 # [('rho', 'nu'), ('xi', 'nu'), ('xi', 'rho')]
   region = cp.get_region("xi", "rho")    # same as copul.regions.get
   float(region.upper(0.3))               # 0.7 = max rho given xi = 0.3
   bool(region.contains(0.3, 0.5))        # True
   C = region.boundary_copula(0.3, side="upper")   # attains the boundary
   ax = region.plot()                     # filled region with key points

   # axes are swapped automatically
   cp.get_region("rho", "xi").lower(0.7)  # 0.3

   # several regions side by side, e.g. for a paper figure
   fig = cp.regions.plot_grid([("xi", "rho"), ("xi", "nu"), ("rho", "nu")])

Regions restricted to a subclass are looked up with ``copula_class``, e.g.
``cp.get_region("xi", "footrule", copula_class="si")``;
``cp.regions.available(None)`` lists all registered ``(x, y, class)`` triples.

Boundary tracing with exact optimisation
----------------------------------------

For a checkerboard copula with mass matrix :math:`P`, Spearman's :math:`\rho`,
Blest's :math:`\nu`, Spearman's footrule :math:`\psi`, Gini's :math:`\gamma`
and Blomqvist's :math:`\beta` are *affine* in :math:`P`, Chatterjee's
:math:`\xi` is a *convex quadratic* and Kendall's :math:`\tau` an indefinite
quadratic (:mod:`copul.optim.checkerboard_formulas`). Extremal problems over
checkerboard copulas are therefore exact LPs/QPs, and every optimum is an
honest copula whose measures are known in closed form: traced boundaries are
*inner* approximations of the exact region which converge as the grid is
refined. ``cvxpy`` is required (``pip install "copul[optim]"``).

.. code-block:: python

   import copul.optim as co
   from copul.schur_order.bounds_from_xi import rho_max_given_xi

   problem = co.CheckerboardProblem(n=24)
   res = problem.maximize("rho", subject_to={"xi": ("<=", 0.3)})
   res["rho"], rho_max_given_xi(0.3)       # inner approximation vs. exact bound
   res.copula                              # optimal BivCheckPi

   # shape constraints are exact linear conditions on P
   res = problem.maximize("nu", subject_to=[("rho", "==", 0.2), "si"])
   co.shape_violation(res.P, "si")         # ~ 0

   # trace a whole boundary (supporting-hyperplane sweep)
   trace = co.trace_boundary("xi", "rho", side="upper", n=24, n_points=12)
   ax = cp.get_region("xi", "rho").plot()
   trace.plot(ax=ax, label="n = 24")

Use ``kind="min"`` or ``kind="w"`` for checkerboards with Fréchet-bound
cells (``BivCheckMin``/``BivCheckW``), and ``method="target"`` in
:func:`~copul.optim.trace_boundary` for a constraint sweep instead of the
:math:`\mu`-sweep. Non-convex objectives (e.g. Kendall's :math:`\tau`) are
handled by a penalty convex-concave procedure.

Counterexample search
---------------------

:mod:`copul.search` automates the loop "draw random copulas, compare two
measures, keep the worst case, refine it locally". Measure keys are evaluated
with the exact checkerboard formulas; any callable ``copula -> float`` works
as well.

.. code-block:: python

   import copul as cp

   # Is xi <= |rho| for all copulas?  No -- a counterexample is found quickly.
   ce = cp.find_counterexample("xi", lambda C: abs(C.spearmans_rho()), n_iter=500, rng=0)
   ce.copula, ce.lhs, ce.rhs, ce.margin

   # Stress-test a proven bound on random SI checkerboards
   region = cp.get_region("xi", "rho")
   report = cp.check_inequality(
       lambda C: abs(C.spearmans_rho()) - float(region.upper(C.chatterjees_xi())),
       0.0, "<=", n_iter=2000, condition="si", rng=0,
   )
   report.holds, report.max_violation, report.argmax

   # random copulas for custom experiments
   for C in cp.random_checkerboards(100, grid=(2, 20), condition="si", rng=1):
       ...

``condition`` accepts ``"si"``, ``"sd"``, ``"exchangeable"``,
``"radially_symmetric"`` or any predicate ``copula -> bool``; ``kind`` selects
``BivCheckPi``/``BivCheckMin``/``BivCheckW``.

From numerical evidence to a proof
----------------------------------

A typical workflow for a new pair of measures :math:`(\kappa_1, \kappa_2)`:

1. trace both boundaries with :func:`~copul.optim.trace_boundary` on a few
   grid sizes and inspect the optimal mass matrices (``trace.results``);
2. guess a boundary copula family from the optimisers, implement it as a
   :class:`~copul.family.core.biv_copula.BivCopula` subclass and compare its
   measures (``measure_curve``) with the traced points;
3. stress-test the conjectured boundary with :func:`~copul.search.check_inequality`;
4. once proven, register the region with :func:`copul.regions.register` so that
   it is plotted and validated like the built-in ones.
