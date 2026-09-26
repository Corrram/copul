Dependence measures
===================

Every bivariate copula in :mod:`copul` exposes the dependence measures below
as methods (e.g. ``C.spearmans_rho()``), and all of them can be computed by
key with :func:`copul.measures.compute` (``copul.compute_measures``). The keys
are also used by :func:`~copul.measures.measure_curve`,
:func:`~copul.measures.from_measure`, :mod:`copul.regions`,
:mod:`copul.optim` and :mod:`copul.search`.

.. code-block:: python

   import copul as cp

   C = cp.Clayton(theta=2)
   C.chatterjees_xi()                          # float
   cp.compute_measures(C, ["xi", "rho", "tau"])  # {"xi": ..., "rho": ..., "tau": ...}

Table of measures
-----------------

:math:`M`, :math:`W` and :math:`\Pi` denote the upper Fréchet bound, the lower
Fréchet bound and the independence copula; :math:`\partial_1 C` and
:math:`\partial_2 C` are the partial derivatives (conditional distribution
functions) and :math:`c` is the copula density.

.. list-table::
   :header-rows: 1
   :widths: 14 20 22 34 5 5 5

   * - key
     - measure
     - method
     - definition
     - :math:`M`
     - :math:`W`
     - :math:`\Pi`
   * - ``xi``
     - Chatterjee's :math:`\xi`
     - ``chatterjees_xi``
     - :math:`6\int_0^1\!\int_0^1 (\partial_1 C(u,v))^2\,du\,dv - 2`
     - 1
     - 1
     - 0
   * - ``xi_2``
     - Chatterjee's :math:`\xi` (conditioning on the second variable)
     - ``chatterjees_xi(condition_on_y=True)``
     - :math:`6\int_0^1\!\int_0^1 (\partial_2 C(u,v))^2\,du\,dv - 2`
     - 1
     - 1
     - 0
   * - ``rho``
     - Spearman's :math:`\rho`
     - ``spearmans_rho``
     - :math:`12\int_0^1\!\int_0^1 C(u,v)\,du\,dv - 3`
     - 1
     - -1
     - 0
   * - ``tau``
     - Kendall's :math:`\tau`
     - ``kendalls_tau``
     - :math:`1 - 4\int_0^1\!\int_0^1 \partial_1 C\,\partial_2 C\,du\,dv`
     - 1
     - -1
     - 0
   * - ``nu``
     - Blest's :math:`\nu`
     - ``blests_nu``
     - :math:`24\int_0^1\!\int_0^1 (1-u)\,C(u,v)\,du\,dv - 2`
     - 1
     - -1
     - 0
   * - ``footrule``
     - Spearman's footrule :math:`\psi`
     - ``spearmans_footrule``
     - :math:`6\int_0^1 C(t,t)\,dt - 2`
     - 1
     - -1/2
     - 0
   * - ``gamma``
     - Gini's :math:`\gamma`
     - ``ginis_gamma``
     - :math:`4\int_0^1 \bigl[C(t,t) + C(t,1-t)\bigr]\,dt - 2`
     - 1
     - -1
     - 0
   * - ``beta``
     - Blomqvist's :math:`\beta`
     - ``blomqvists_beta``
     - :math:`4\,C(\tfrac12,\tfrac12) - 1`
     - 1
     - -1
     - 0
   * - ``hoeffdings_d``
     - Hoeffding's :math:`\Phi^2`
     - ``hoeffdings_d``
     - :math:`90\int_0^1\!\int_0^1 (C(u,v)-uv)^2\,du\,dv`
     - 1
     - 1
     - 0
   * - ``sigma``
     - Schweizer--Wolff :math:`\sigma`
     - ``schweizer_wolff_sigma``
     - :math:`12\int_0^1\!\int_0^1 |C(u,v)-uv|\,du\,dv`
     - 1
     - 1
     - 0
   * - ``kappa``
     - uniform distance :math:`\kappa`
     - ``uniform_distance``
     - :math:`4\sup_{(u,v)\in[0,1]^2}|C(u,v)-uv|`
     - 1
     - 1
     - 0
   * - ``lp``
     - :math:`L^p` distance :math:`\delta_p`
     - ``lp_distance``
     - :math:`k(p)\int_0^1\!\int_0^1 |C(u,v)-uv|^p\,du\,dv`, :math:`k(p)=\frac{p+1}{2B(p+1,p+2)}`
     - 1
     - 1
     - 0
   * - ``bkr``
     - Blum--Kiefer--Rosenblatt :math:`B`
     - ``blum_kiefer_rosenblatt``
     - :math:`30\int_{[0,1]^2}(C(u,v)-uv)^2\,dC(u,v)`
     - 1
     - 1
     - 0
   * - ``mutual_information``
     - mutual information :math:`I`
     - ``mutual_information``
     - :math:`\int_0^1\!\int_0^1 c(u,v)\log c(u,v)\,du\,dv`
     - :math:`\infty`
     - :math:`\infty`
     - 0
   * - ``lambda_l``
     - lower tail dependence :math:`\lambda_L`
     - ``lambda_L``
     - :math:`\lim_{t\to0^+} C(t,t)/t`
     - 1
     - 0
     - 0
   * - ``lambda_u``
     - upper tail dependence :math:`\lambda_U`
     - ``lambda_U``
     - :math:`\lim_{t\to1^-} (1-2t+C(t,t))/(1-t)`
     - 1
     - 0
     - 0

Keys accept aliases such as ``"spearman"``, ``"kendall"``, ``"psi"`` or
``"gini"``; :func:`copul.measures.list_measures` and
:func:`copul.measures.get_measure` give the full records (aliases, ranges,
LaTeX symbols and formulas).

Dispatch: the ``method=`` keyword
---------------------------------

All measure methods and :func:`~copul.measures.compute` accept
``method="auto" | "closed" | "numeric" | "symbolic" | "mc"``.

**Copulas with free parameters** (e.g. ``cp.Clayton()``) return SymPy
expressions: the family's closed form, or the generic symbolic integration of
the base class. ``method="numeric"`` and ``"mc"`` raise :class:`ValueError`
because there is nothing to evaluate numerically.

.. code-block:: python

   cp.Clayton().kendalls_tau()               # theta/(theta + 2)
   cp.FarlieGumbelMorgenstern().spearmans_rho()  # theta/3

**Fully specified copulas** return Python floats:

``"auto"`` (default)
   A closed form (a family-specific override) is tried first. If there is
   none, or it raises, returns ``nan``, a non-number or an unevaluated
   ``sympy.Integral``, the numerical engine is used instead (the fallback is
   logged at DEBUG level).
``"closed"``
   Only the closed form; an error if there is none.
``"numeric"``
   Adaptive quadrature on vectorised cdf / :math:`h`-functions / density
   (:func:`copul.measures.numeric_backend`), with one-dimensional formulas
   where available (Kendall's :math:`\tau` of Archimedean copulas,
   :math:`\rho` and :math:`\tau` of extreme-value copulas) and exact
   formulas for checkerboard copulas.
``"symbolic"``
   The SymPy route (the family's symbolic implementation if present,
   otherwise generic symbolic integration); returns SymPy objects.
``"mc"``
   Empirical estimator on ``n_samples`` samples of ``copula.rvs``
   (``random_state`` for reproducibility).

.. code-block:: python

   C = cp.Clayton(theta=2)
   C.spearmans_rho()                    # 0.6822... (auto -> numeric)
   C.spearmans_rho(method="mc", n_samples=10_000, random_state=0)
   cp.compute_measures(C, "rho", full_output=True).method   # 'numeric'

The numerical engine uses the default tolerances ``rtol=1e-8`` and
``atol=1e-10`` on the measure scale (``rtol=``, ``atol=`` in
:func:`~copul.measures.compute`).

Parameter sweeps and calibration
--------------------------------

.. code-block:: python

   curve = cp.Frank().measure_curve(["tau", "rho", "xi"], n=50)
   curve.plot()
   cp.Frank.from_measure("tau", 0.5)    # Frank copula with Kendall's tau 0.5
