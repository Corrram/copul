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

Rank-correlation atlas
----------------------

All 23 population coefficients in the coefficient table of
``pairwise-exact-regions.tex`` are available through the shared measure API.
The tuple ``copul.measures.ATLAS_MEASURES`` selects that complete collection:

.. code-block:: python

   from copul.measures import ATLAS_MEASURES, compute

   C = cp.FarlieGumbelMorgenstern(0.7)
   values = compute(C, ATLAS_MEASURES)
   C.integrated_r2()                         # 0.7**2 / 18
   C.correlation_ratio(condition_on_y=True)  # coefficient of the transpose
   C.hoeffding_d()                          # 0.7**2 / 30

In addition to rho, tau, beta, footrule, gamma, nu, xi, zeta1, sigma,
Phi-square and kappa, the following names are supported:

.. list-table::
   :header-rows: 1
   :widths: 20 28 28 24

   * - Key
     - Coefficient
     - Method
     - Definition
   * - ``eta``
     - Symmetrized Blest
     - ``blests_eta()``
     - :math:`[\nu(C)+\nu(C^\top)]/2`
   * - ``normal_score`` (``rho_n``)
     - Normal-score correlation
     - ``normal_score_correlation()``
     - :math:`E[\Phi_N^{-1}(U)\Phi_N^{-1}(V)]`
   * - ``savage_score``
     - Savage-score correlation
     - ``savage_score_correlation()``
     - :math:`E[\log(1-U)\log(1-V)]-1`
   * - ``omega``
     - Siburg--Stoimenov
     - ``siburg_stoimenov()``
     - :math:`\sqrt{[\xi(C)+\xi(C^\top)]/2}`
   * - ``hoeffding_d``
     - Hoeffding's D
     - ``hoeffding_d()``
     - :math:`30\int(C-uv)^2\,dC`
   * - ``tau_star``
     - Bergsma--Dassios sign covariance
     - ``bergsma_dassios()``
     - :math:`2D/5+4\Phi^2/15`
   * - ``chi``
     - Copula correlation ratio
     - ``correlation_ratio()``
     - :math:`12\operatorname{Var}(E[V\mid U])`
   * - ``maximal_correlation`` (``R``)
     - Gebelein--Renyi maximal correlation
     - ``maximal_correlation()``
     - :math:`\sup_{f,g}\operatorname{corr}(f(U),g(V))`
   * - ``integrated_r2`` (``A``)
     - Azadkia--Roudaki integrated R-squared
     - ``integrated_r2()``
     - :math:`\int(\partial_1C-v)^2/[v(1-v)]\,du\,dv`
   * - ``linfoot`` (``L``)
     - Linfoot informational correlation
     - ``linfoot_correlation()``
     - :math:`\sqrt{1-e^{-2I(C)}}`
   * - ``hellinger`` (``H``)
     - Gaussian-calibrated Hellinger correlation
     - ``hellinger_correlation()``
     - :math:`2B^{-2}\sqrt{B^4+\sqrt{4-3B^4}-2}`, :math:`B=\int\sqrt{c_C}`
   * - ``silvey`` (``delta_s``)
     - Silvey total variation
     - ``silvey_delta()``
     - :math:`[\int|c_C-1|+s_C]/2`

Naming and normalization
~~~~~~~~~~~~~~~~~~~~~~~~

The existing key/method ``hoeffdings_d`` and aliases ``phi2`` and ``d``
continue to return :math:`\Phi^2=90\int(C-uv)^2\,du\,dv`.
Use the new **singular spelling** ``hoeffding_d`` for the paper's
:math:`D=30\int(C-uv)^2\,dC`. The legacy ``bkr`` key also returns that D.
These names retain their previous meanings for compatibility.
Similarly, ``rho_s`` remains an alias for Spearman's rho; use
``savage_score`` for the paper's :math:`\rho_{\mathrm S}`.
Sign covariance is normalized as in the paper, with maximum **2/3**.
The correlation ratio returns explained variance, without a square root.

Numerical and symbolic evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``chi`` and ``integrated_r2`` support ``condition_on_y=True`` for the
transpose. The default conditions on the first coordinate, as does xi.
Checkerboard evaluation uses cellwise formulas, including rectangular
grids and cells with singular Min/W kernels. Generic CDF quadrature for
score correlations and conditional-distribution quadrature for regression
coefficients include singular mass without requiring a density.

For divergences, :math:`c_C` denotes the density of the absolutely
continuous part and :math:`s_C` the singular mass. Linfoot equals 1 as soon
as a singular component is known to be present. Hellinger equals 1 when
:math:`B=0`; Silvey includes the singular mass through the equivalent
formula :math:`\int(1-c_C)_+`. Density divergences require an explicit
density; finite differences cannot reliably separate a singular component.
Both Linfoot and Hellinger equal the absolute correlation parameter for
Gaussian copulas. Density integrals remain subject to quadrature accuracy.

Maximal correlation has exact routes for Gaussian copulas, Fréchet
mixtures, shuffles of Min, checkerboards and Bernstein copulas. For other copulas it uses the
largest singular value of the centered conditional-expectation matrix on
a dyadic partition. This is a **lower approximation**, converging to R as
the partition is refined. ``max_grid=256`` is the default; it must be a
power of two of at least 16. With ``full_output=True``, ``error`` reports
the difference from the preceding resolution, **not an error bound**.
Unresolved fine-scale dependence can remain even when this difference is
small. ``rtol`` and ``atol`` do not control this discretization. Exact
checkerboard and Bernstein formulas do not depend on ``max_grid``.

.. code-block:: python

   estimate = compute(C, "R", max_grid=512, full_output=True)
   print(estimate.value, estimate.error, estimate.info)

Polynomial symbolic formulas are evaluated directly. Unbounded score and
density formulas may return unevaluated SymPy integrals; symbolic density
divergences require known absolute continuity. Maximal correlation has no
generic symbolic implementation. Monte Carlo rank estimators are also
available for eta and the two score correlations. The full atlas is
available for copula objects and the callable numerical API; the sampled
h-grid API supports its separately documented subset.

Closed formulas in the checkerboard modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BivCheckPi`` implements all 23 atlas coefficients as closed cellwise
formulas. ``BivCheckMin``, ``BivCheckW``, ``BivCheckMixed``,
``BivBlockDiagMixed`` and ``ShuffleOfMin`` have the same coverage except
for normal scores on off-diagonal singular cells. Normal scores are closed
when every positive-mass singular cell lies on the global diagonal or
anti-diagonal. Other singular score products use one-dimensional numerical
integration under ``method="auto"``; ``method="closed"`` raises
:class:`NotImplementedError` for those normal-score cases.

.. doctest::

   >>> from copul import BivCheckPi
   >>> from copul.measures import compute
   >>> C = BivCheckPi([[0.4, 0.1], [0.1, 0.4]])
   >>> round(C.maximal_correlation(method="closed"), 6)
   0.6
   >>> result = compute(C, "integrated_r2", full_output=True)
   >>> (result.method, result.error)
   ('closed', 0.0)

The implementations include:

* Polynomial moments over squares and triangles for :math:`\Phi^2`, and
  over singular diagonals for :math:`D`; :math:`\tau^*` combines both.
* Polynomial and logarithmic primitives for :math:`\sigma` and integrated
  :math:`R^2`; cell corners and diagonal stationary points for :math:`\kappa`.
* Inverse-normal cell means for normal scores, and logarithm/dilogarithm
  primitives for Savage scores on Min/W diagonals.
* Finite matrix singular values for maximal correlation. The cell-constant
  and local-function parts of the operator are both included.
* Cell-density sums for Linfoot, Hellinger and Silvey, with singular mass
  included explicitly.

``BivBernsteinCopula`` additionally has closed formulas for ``eta``,
``omega``, ``hoeffdings_d`` (Phi²), ``hoeffding_d`` (D), ``tau_star``,
``chi``, ``integrated_r2``, ``savage_score`` and ``maximal_correlation``,
alongside its existing concordance and xi formulas. Polynomial integrals
use degree-matched Gauss--Legendre rules, which are exact up to floating
point arithmetic; Savage scores use Beta-mixture digamma identities.
The centered polynomial density operator gives a finite SVD for maximal
correlation. The other Bernstein coefficients retain their numerical
fallbacks.

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
