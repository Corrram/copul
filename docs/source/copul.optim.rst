copul.optim package
===================

Exact LP/QP optimisation of dependence measures over checkerboard copulas.
``cvxpy`` is an optional dependency (``pip install "copul[optim]"``); it is
imported only when a problem is built or solved. See
:doc:`research_workflows` for worked examples.

.. automodule:: copul.optim
   :members:

Submodules
----------

The objects above are defined in the following modules; their public members
are documented above.

- ``copul.optim.problem`` -- :class:`~copul.optim.CheckerboardProblem` and results
- ``copul.optim.boundary`` -- :func:`~copul.optim.trace_boundary`
- ``copul.optim.checkerboard_formulas`` -- exact affine/quadratic measure forms in the mass matrix
- ``copul.optim.shapes`` -- exact linear shape constraints (SI, SD, LTD, PQD, symmetry, ...)
- ``copul.optim.ccp`` -- penalty convex-concave procedure for non-convex problems
