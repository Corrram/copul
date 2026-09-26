copul.measures package
======================

Registry of dependence measures, the numerical measures engine and the
``method=`` dispatch shared by all measure methods of bivariate copulas.
See :doc:`measures` for the list of measures, their definitions and the
dispatch semantics.

.. automodule:: copul.measures
   :members:

Submodules
----------

The objects above are defined in the following modules; their public members
are documented above.

- ``copul.measures.registry`` -- :class:`~copul.measures.Measure` records and lookup
- ``copul.measures.engine`` -- :func:`~copul.measures.compute` and the dispatch
- ``copul.measures.curves`` -- parameter sweeps and calibration
- ``copul.measures.numeric`` -- measure formulas on vectorised callables
- ``copul.measures.quadrature`` -- adaptive Gauss--Kronrod and Gauss--Legendre rules
- ``copul.measures.backend`` -- vectorised cdf / h-functions / pdf of copula objects
