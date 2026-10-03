Copula theory toolkit
=====================

:mod:`copul.theory`, :mod:`copul.sklar` and :mod:`copul.multivariate` turn the
standard results of copula theory into functions that work on *any* copula
object of the package (parametric families, checkerboards, constructions,
empirical copulas).  Everything implemented is published theory; each function
cites its sources, mainly

* R. B. Nelsen, *An Introduction to Copulas*, 2nd ed., Springer 2006;
* F. Durante & C. Sempi, *Principles of Copula Theory*, CRC 2016;
* H. Joe, *Dependence Modeling with Copulas*, CRC 2014.

Most functions are also available as methods of copula objects, e.g.
``C.kendall_distribution(t)``, ``C.dependence_profile()``,
``C.nonexchangeability()`` or ``C.distance(D, "D1")``.

.. code-block:: python

   import copul as cp
   from copul import theory as th

Sklar's theorem
---------------

:class:`copul.sklar.JointDistribution` couples a copula with arbitrary
``scipy.stats`` margins, :math:`H(x,y)=C(F(x),G(y))`: cdf, density, sampling,
conditional distributions and quantiles, copula regression curves, rectangle
probabilities and Hoeffding's covariance formula.
:func:`~copul.sklar.copula_from_joint` extracts the copula of a joint
distribution, :math:`C(u,v)=H(F^{-1}(u),G^{-1}(v))`.

.. code-block:: python

   from scipy import stats

   H = cp.JointDistribution(cp.Clayton(2), [stats.norm(), stats.expon()])
   H.cdf([0.0, 1.0])            # 0.4263
   H.regression(0.5)            # E[Y | X = 0.5]
   H.correlation()              # Pearson correlation via Hoeffding's formula
   X = H.rvs(1000, random_state=0)

Archimedean copulas
-------------------

* :func:`~copul.theory.archimedean.kendall_distribution` --
  :math:`K_C(t)=P(C(U,V)\le t)` for any copula (closed form
  :math:`t-\varphi(t)/\varphi'(t)` for Archimedean ones), with
  :math:`\tau = 3-4\int_0^1 K_C`;
* :func:`~copul.theory.archimedean.check_generator` /
  :func:`~copul.theory.archimedean.generator_properties` -- validity,
  strictness, zero-curve mass, :math:`d`-monotonicity and complete
  monotonicity of the inverse generator, :func:`~copul.theory.archimedean.max_dimension`;
* :func:`~copul.theory.archimedean.is_archimedean` -- associativity and
  :math:`C(t,t)<t`;
* numerical Archimedean copulas from a generator, a Laplace transform or a
  Kendall distribution function.

.. code-block:: python

   cp.Clayton(2).kendall_distribution(0.3)     # 0.4365 = t + t(1 - t^theta)/theta
   th.check_generator(cp.Clayton(2)).completely_monotone   # True
   th.max_dimension(cp.Clayton(-0.3))          # 4 = floor(1 - 1/theta)
   cp.Gaussian(0.5).is_archimedean()           # False

Extreme-value copulas
---------------------

Max-stability checks, Pickands-function checks, the extremal coefficient
:math:`2A(\tfrac12)`, the stable tail dependence function, tail copulas,
Pickands/CFG estimators and the extreme-value attractor
:math:`\lim_n C(u^{1/n},v^{1/n})^n` of any copula.

.. code-block:: python

   th.is_max_stable(cp.GumbelHougaard(2))      # True
   th.extremal_coefficient(cp.Galambos(1.0))   # 1.5
   cp.survival(cp.Clayton(2)).ev_attractor()   # a Galambos-type EV copula

Dependence concepts and orderings
---------------------------------

PQD/NQD, LTD/LTI, RTI/RTD, SI/SD, LCSD/RCSI and TP2/RR2 with exact
characterizations for many families (Gaussian, FGM, Archimedean via generator
criteria, extreme-value copulas, checkerboards) and dense-grid checks
otherwise; :func:`~copul.theory.dependence.dependence_profile` reports all of
them and checks the implication hierarchy.  The concordance order and the
monotonicity of families in their parameter are in :mod:`copul.theory.orders`.

.. code-block:: python

   cp.Frank(3).dependence_profile()            # PQD, LTD, RTI, SI, TP2, ... all exact
   th.is_concordance_ordered(cp.Clayton, [0.5, 1, 2, 4]).increasing   # True

Markov products, distances and complete dependence
--------------------------------------------------

Markov products and powers, left/right invertibility, complete and mutual
complete dependence, idempotents, Markov operators and conditional
expectations (:mod:`copul.theory.markov`); sup, :math:`L^p` and Trutschnig's
:math:`\partial`-distances between copulas and the dependence measure
:math:`\zeta_1` (:mod:`copul.theory.distances`; also a registered measure).

.. code-block:: python

   cp.ShuffleOfMin([2, 1, 3]).is_completely_dependent()   # True
   cp.Clayton(2).distance(cp.Frank(5), "D1")
   cp.Frank(3).trutschnig_zeta()

Bounds, quasi-copulas and diagonals
-----------------------------------

Best-possible pointwise bounds for copulas with a given value
:math:`C(a,b)`, a given diagonal, or a given value of Kendall's :math:`\tau`,
Spearman's :math:`\rho` or Blomqvist's :math:`\beta`
(:func:`~copul.theory.bounds.bounds_given_measure`); quasi-copulas and their
2-increasing defects; diagonal sections, Bertino and diagonal copulas.

.. code-block:: python

   b = th.bounds_given_measure("tau", 0.5)
   b.lower.cdf(0.3, 0.6), b.upper.cdf(0.3, 0.6)
   th.BertinoCopula(cp.Clayton(2).diagonal_section())   # smallest copula with that diagonal

Symmetry
--------

Non-exchangeability :math:`\mu_\infty(C)=3\sup|C-C^\top|\in[0,1]`, radial
asymmetry :math:`\sup|C-\hat C|`, symmetrizations, and tests of
exchangeability and radial symmetry from data.

.. code-block:: python

   K = cp.khoudraji(cp.BivIndependenceCopula(), cp.GumbelHougaard(3), 0.3, 0.9)
   K.nonexchangeability()                      # 0.116
   th.exchangeability_test(K.rvs(500, random_state=1), n_boot=100, random_state=0)

Multivariate copulas
--------------------

:mod:`copul.multivariate` provides :math:`d`-dimensional Gaussian, Student-t and
Archimedean copulas (Clayton, Gumbel, Frank, Joe, AMH, or any generator /
Laplace transform) with vectorized cdf, density and sampling, bivariate margins
as ordinary copul copulas, and multivariate Spearman's :math:`\rho`
(Schmid & Schmidt), Kendall's :math:`\tau` (Nelsen) and Blomqvist's
:math:`\beta` for copulas and data.

.. code-block:: python

   C3 = cp.ClaytonND(2.0, dim=3)
   cp.multivariate.kendalls_tau_nd(C3)         # 0.5
   C3.rvs(1000, random_state=0)
