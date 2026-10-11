Vine copulas and likelihood fitting
============================================

``copul.vine`` implements **simplified, absolutely continuous regular
vine copulas**, including canonical (C-) and drawable (D-) vines, in any
dimension :math:`d\ge2`. The models inherit the numerical ``CopulaND`` API
and combine existing bivariate copula families. No external vine library
is required.

Fit a model
-----------

.. code-block:: python

   import copul as cp

   # An example with lower-tail, symmetric and negative dependence.
   truth = cp.DVineCopula(3, [
       [cp.Clayton(theta=2), cp.Gaussian(rho=-0.5)],
       [cp.Frank(theta=3)],
   ])
   U = truth.rvs(1500, random_state=42)

   result = cp.fit_vine(
       U, vine_type="rvine", pseudo_obs=False,
       families=["Independence", "Gaussian", "Clayton", "Frank", "T"],
       criterion="bic", joint=True,
   )
   print(result.summary())
   model = result.copula
   model.logpdf(U[:5])
   simulated = model.rvs(1000, random_state=7)
   residuals = model.rosenblatt(U)
   recovered = model.inverse_rosenblatt(residuals)

For raw observations, leave ``pseudo_obs=True`` (the default). Columns are
ranked once as :math:`R_{ij}/(n+1)` with average ranks for ties. Conditional
observations in subsequent trees are **not ranked again**. Missing or
infinite observations, constant columns and copula-scale values outside
the open unit cube are rejected. This is a continuous likelihood; average
ranks do not turn it into a likelihood for discrete margins.

There are three entry points:

* ``cp.fit_vine(X, ...)`` returns ``VineFitResult``; the default is sequential
  estimation, with optional ``joint=True`` refinement.
* ``cp.CVineCopula.fit(X, ...)``, ``cp.DVineCopula.fit(X, ...)`` and
  ``cp.VineCopula.fit(X, ...)`` return the model. Diagnostics are available
  through ``model.fit_result_``.
* ``cp.fit(cp.DVineCopula, X, method="sequential", ...)`` returns a
  ``VineFitResult``. Here ``method="mle"`` (the generic fitting default)
  additionally performs joint likelihood refinement.

``result.loglik`` is the full fitted copula log-likelihood;
``result.sequential_loglik`` records its value before joint refinement.
``result.edge_fits`` gives the final conditional likelihood contribution,
parameters and fixed parameters for every edge. ``result.converged`` and
``result.optimizer_result`` expose joint optimizer status. A failed joint
optimizer is not reported as converged; refinement never replaces the
initial model with a lower-likelihood solution. Candidate failures are
recorded in ``result.selection_failures``; failure of all candidates on an
edge raises an error instead of silently substituting independence.

Regular-vine definition
-----------------------

A regular vine on :math:`d` variables consists of trees
:math:`T_1,\ldots,T_{d-1}`. The first tree has nodes :math:`0,\ldots,d-1`;
the nodes of :math:`T_{t+1}` are the edges of :math:`T_t`. Two nodes in
:math:`T_{t+1}` may be joined only if their corresponding edges share a
node in :math:`T_t`: the **proximity condition**.

Every edge has a conditioned pair :math:`(a,b)` and a conditioning set
:math:`D`, written :math:`(a,b\mid D)`. In tree :math:`T_t`,
:math:`|D|=t-1`. Its endpoints have complete variable sets
:math:`D\cup\{a\}` and :math:`D\cup\{b\}`. ``VineStructure`` validates
edge counts, indices, connectivity, absence of cycles, parent nodes,
proximity and the availability of all conditional distributions.

Assign an absolutely continuous bivariate copula :math:`C_{ab;D}` to
each edge. Under the simplifying assumption its parameters do not depend
on the values :math:`u_D`. The density is

.. math::

   c(u_0,\ldots,u_{d-1}) =
   \prod_{t=1}^{d-1}\prod_{(a,b\mid D)\in T_t}
   c_{ab;D}\bigl(F_{a\mid D}(u_a\mid u_D),
                 F_{b\mid D}(u_b\mid u_D)\bigr).

These factors are evaluated at **conditional** probabilities, not at the
original observations. With :math:`F_{a\mid\varnothing}=u_a`, the recursive
identities are

.. math::

   F_{a\mid D\cup\{b\}} =
   \partial_2 C_{ab;D}(F_{a\mid D},F_{b\mid D}),\qquad
   F_{b\mid D\cup\{a\}} =
   \partial_1 C_{ab;D}(F_{a\mid D},F_{b\mid D}).

Thus ``h1`` means differentiation with respect to the first argument and
returns the conditional distribution of the *second* variable. ``h2``
conditions on the second variable. Edge orientation matters for
asymmetric pairs.

Why this is a copula
--------------------

The conditional-distribution identities follow by differentiating the
conditional joint distribution supplied by Sklar's theorem. Applying the
chain rule recursively along an admissible variable order gives the
product above as the Jacobian determinant of the Rosenblatt transform.
Each conditional distribution is nondecreasing and maps its support onto
the unit interval (up to null sets). The inverse conditional transforms
therefore map independent uniforms into a probability distribution with
that density. Integrating the conditional densities successively gives
total mass one. Pair copulas have uniform margins, so the univariate
margins of the construction are uniform as well. The graph and proximity
conditions ensure these recursions use consistent parent conditionals.

The implementation checks these identities independently against Gaussian
and Student-t multivariate densities, their conditional distributions,
the Rosenblatt Jacobian, quadrature normalization and simulated margins.
For a Gaussian vine, an edge parameter is the partial correlation
:math:`\rho_{ab\cdot D}`, not the marginal correlation. For a vine
representing a multivariate Student-t copula with degrees of freedom
:math:`\nu`, that edge has degrees of freedom :math:`\nu+|D|`. Keeping the
same :math:`\nu` on every tree generally specifies a different vine.

C-vines, D-vines and explicit regular vines
--------------------------------------------------

For a C-vine with order :math:`p`, tree :math:`t+1` has edges
:math:`(p_t,p_j\mid p_0,\ldots,p_{t-1})`, :math:`j>t`. Its roots follow
``order``. A D-vine's first tree is the path ``order``; tree :math:`k`
contains :math:`(p_i,p_{i+k}\mid p_{i+1},\ldots,p_{i+k-1})`.

``pair_copulas`` is a list of lists: one per tree, with lengths
:math:`d-1,d-2,\ldots,1`. For C-vines the pairs follow increasing :math:`j`;
for D-vines increasing :math:`i`. Variable labels are zero based. Omitting
the pairs supplies independence on every edge.

.. code-block:: python

   C = cp.CVineCopula(3, [
       [cp.Clayton(theta=2), cp.GumbelHougaard(theta=1.5)],
       [cp.Gaussian(rho=-0.3)],
   ], order=[2, 0, 1])
   # T1: (2,0), (2,1); T2: (0,1 | 2).

   structure = cp.VineStructure([
       [(0, 1, ()), (1, 2, ()), (1, 3, ()), (3, 4, ())],
       [(0, 2, (1,)), (2, 3, (1,)), (1, 4, (3,))],
       [(0, 3, (1, 2)), (2, 4, (1, 3))],
       [(0, 4, (1, 2, 3))],
   ])
   R = cp.RVineCopula(structure)  # independence pairs initially
   X = R.rvs(300, random_state=8)
   fitted = cp.fit_vine(X, structure=structure, families=["Gaussian"],
                       pseudo_obs=False)

``RVineCopula`` is an alias for ``VineCopula``. Explicit general vines use
edge lists rather than a matrix convention. A valid forward sampling order
is derived automatically, or can be supplied to ``VineStructure`` and is
then validated. C-vines and D-vines use their specified order for sampling.
All transforms return columns in their **original variable order**:
column ``order[k]`` conditions on variables ``order[:k]``.

Structure, families and parameter estimation
--------------------------------------------

With ``vine_type="rvine"`` (the default), each tree is a maximum spanning
tree of absolute empirical Kendall taus of the current conditional
observations, restricted by proximity. Ties are broken deterministically.
C-vines greedily choose the root maximizing the sum of absolute conditional
taus. D-vines start from the strongest marginal pair and greedily extend
either end of a path. These are heuristics, not globally optimal structure
searches. Fix a C/D order with ``order=...`` or an R-vine with
``structure=...`` to bypass that search.

Each selected edge is fitted by maximum pair likelihood and compared by
``criterion="aic"``, ``"bic"`` or ``"loglik"``. The defaults are
independence, Gaussian, Clayton, Frank, Gumbel and Student-t. Explicit
candidates may use any existing bivariate family with an absolutely
continuous density and analytic or symbolic conditional distributions.
Singular pairs and finite-difference-only ingredients are rejected.
Parameters of a supplied partially specified family stay fixed.

.. code-block:: python

   candidates = [
       "Independence", cp.Gaussian, cp.StudentT(nu=4),
       cp.PairCopulaFamily(cp.Clayton, rotation=90),
       cp.PairCopulaFamily(cp.Clayton, rotation=180),
       cp.PairCopulaFamily(cp.Clayton, rotation=270),
   ]
   result = cp.fit_vine(U, vine_type="dvine", order=[0, 1, 2],
                        families=candidates, truncation=1,
                        pseudo_obs=False)

``PairCopulaFamily`` uses the common vine reflection convention: 90
reflects the first coordinate, 180 both, and 270 the second. For asymmetric
base pairs this differs from geometric rotations in ``cp.rotate``. Supply
reflected candidates explicitly; they are not automatically added.

Sequential estimation fits each tree using previously fitted conditional
distributions. It is not joint maximum likelihood. ``joint=True`` uses
these estimates as starts, holds the chosen structure and families fixed,
and recomputes every conditional probability at each evaluation of the
full likelihood. This is numerical local optimization, with no guarantee
of a global maximum. ``truncation=L`` replaces all pairs after tree
:math:`L` by independence. ``truncation=0`` gives the product copula.

For :math:`k` estimated continuous parameters and :math:`n` observations,

.. math::

   \mathrm{AIC}=2k-2\ell,\qquad
   \mathrm{BIC}=k\log n-2\ell.

Fixed parameters and independent pairs contribute zero to :math:`k`.
These scores do not account for the discrete search over structures and
families. Pair Hessians are not valid joint standard errors after
sequential estimation, ranking or selection; the vine API does not claim
such uncertainty estimates. A bootstrap must repeat the entire relevant
fitting/selection procedure.

Numerical scope and limitations
-------------------------------

The density is the pair product evaluated numerically. Sampling uses
inverse conditional transforms, with analytic pair inverses when available
and numerical inversion otherwise. Probabilities used in density and inverse
calculations are clipped to :math:`[10^{-12},1-10^{-12}]` for stability;
this can affect extreme
tails, particularly for strong dependence or deep vines. Density values
on the boundary use that interior convention, not a claimed boundary limit.

``model.cdf(u, n_points=4096, random_state=0)`` performs scrambled Sobol
conditional integration. ``n_points`` is rounded up to a power of two.
In dimension two the pair's own CDF is used. In higher dimensions it
successively samples from conditional laws truncated at the CDF bounds,
and averages the product of the conditional truncation probabilities.
This change of variables avoids integrating a potentially unbounded
density directly. Grounding and univariate margins are exact; general
values are approximations. Increase the point count and compare seeds to
assess error. There is no deterministic error bound, and numerical CDF
errors can affect derived box probabilities and CDF-based measures.

Unconditional margins beyond the first tree generally require integration;
a higher-tree pair is a conditional copula, not an unconditional margin.
The inherited multivariate marginal/measure machinery uses numerical
evaluation where needed and may be expensive. The construction does not
implement non-simplified vines with parameters varying in conditioning
values, discrete/mixed likelihoods, or singular pair copulas.

References
----------

* Aas, K., Czado, C., Frigessi, A. and Bakken, H. (2009).
  Pair-copula constructions of multiple dependence.
  *Insurance: Mathematics and Economics* 44, 182--198.
  `doi:10.1016/j.insmatheco.2007.02.001 <https://doi.org/10.1016/j.insmatheco.2007.02.001>`_.
* Dißmann, J., Brechmann, E. C., Czado, C. and Kurowicka, D. (2013).
  Selecting and estimating regular vine copulae and application to financial
  returns. *Computational Statistics & Data Analysis* 59, 52--69.
  `Preprint <https://arxiv.org/abs/1202.2002>`_.
* Hobæk Haff, I. (2013). Parameter estimation for pair-copula constructions.
  *Bernoulli* 19, 462--491.
  `doi:10.3150/12-BEJ413 <https://doi.org/10.3150/12-BEJ413>`_.
* Rosenblatt, M. (1952). Remarks on a multivariate transformation.
  *Annals of Mathematical Statistics* 23, 470--472.

See :doc:`copul.vine` for the API reference.
