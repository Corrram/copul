Copula mixtures and parameter uncertainty
============================================

The fitting APIs support likelihood estimation for finite copula mixtures
and bootstrap inference for the parameters of a fixed bivariate family.
Both work with the existing numerical copula API. For multivariate
pair-copula fitting, see :doc:`vine_copulas`.

Fit heterogeneous dependence with a mixture
--------------------------------------------------

A convex combination of copulas is itself a copula. For absolutely
continuous components,

.. math::

   C(u,v)=\sum_{j=1}^m\pi_j C_j(u,v;\theta_j),\qquad
   c(u,v)=\sum_{j=1}^m\pi_j c_j(u,v;\theta_j),\qquad
   \pi_j\ge0,\quad\sum_j\pi_j=1.

The component label is latent. Mixtures can represent populations with
different dependence patterns, including opposing positive and negative
associations which a single fitted correlation can obscure.

.. code-block:: python

   import copul as cp

   truth = cp.mixture(
       [cp.Gaussian(rho=0.8), cp.Gaussian(rho=-0.7)], [0.6, 0.4]
   )
   U = truth.rvs(1200, random_state=38)
   result = cp.fit_mixture(
       U, [cp.Gaussian, cp.Gaussian], pseudo_obs=False,
       start=[{"rho": 0.6}, {"rho": -0.5}],
       n_init=3, random_state=42,
   )
   print(result.summary())
   result.component_params
   result.weights
   probabilities = result.predict_proba(U[:5])
   simulated = result.rvs(1000, random_state=7)

For raw observations, leave ``pseudo_obs=True``. Columns are ranked once
as :math:`R_{ij}/(n+1)` before fitting. Ranking within each component would
change the objective and is not performed. ``predict_proba`` accepts
copula-scale observations: transform new raw observations with the same
original marginal models, rather than ranking each prediction batch.

Generalized EM
---------------

The observed log-likelihood is

.. math::

   \ell(\pi,\theta)=\sum_{i=1}^n
   \log\left[\sum_{j=1}^m\pi_j c_j(U_{i1},U_{i2};\theta_j)\right].

For the current parameters, the E-step computes responsibilities

.. math::

   r_{ij}=\frac{\pi_j c_j(U_i;\theta_j)}
   {\sum_{k=1}^m\pi_k c_k(U_i;\theta_k)}.

The M-step updates :math:`\pi_j=n^{-1}\sum_i r_{ij}` and fits each
component to the weighted log-likelihood
:math:`Q_j(\theta_j)=\sum_i r_{ij}\log c_j(U_i;\theta_j)`.
Zero-responsibility observations are excluded before taking log-densities,
so zero density there does not produce an invalid :math:`0\log0` product.
Numerical component updates are accepted only if they do not lower
:math:`Q_j`; the complete observed likelihood is also checked before the
iterate is accepted. ``result.loglik_path`` records accepted values.
Likelihoods and responsibilities use log-sum-exp for numerical stability.

``max_iter`` and ``tol`` control convergence separately for each start.
After the first start, free parameters are perturbed on their unconstrained
scales and estimated mixture weights are randomized. ``random_state``
makes this reproducible. The highest-likelihood run is returned; it may
have reached its iteration limit, so inspect ``result.converged``,
``result.message`` and ``result.starts``. EM and multiple starts do not
guarantee a global maximum. An identical start for repeated families can
remain symmetric, which is why distinct initial parameters or multiple
starts matter.

Fixed parameters, weights and reflections
-------------------------------------------

Fully specified component copulas stay fixed, so one can estimate just
their mixing weights. Partially specified families retain their fixed
parameters. Set ``fit_weights=False`` to hold normalized starting weights
fixed. Starting weights must be strictly positive; components whose
estimated weight tends to zero are retained with a diagnostic message.

.. code-block:: python

   fitted_weights = cp.fit_mixture(
       U, [cp.Gaussian(rho=0.8), cp.Gaussian(rho=-0.7)],
       pseudo_obs=False, n_init=1,
   )

   tail_data = cp.Clayton(theta=2).rvs(600, random_state=8)
   reflected = cp.fit_mixture(
       tail_data,
       [cp.Clayton, cp.PairCopulaFamily(cp.Clayton, rotation=180)],
       weights=[0.7, 0.3], n_init=2, random_state=4,
   )

The reflection convention is the same as for vine fitting: 90 reflects
the first coordinate, 180 both, and 270 the second. Singular components
are excluded because this fitter uses ordinary continuous densities.

``result.k`` is the sum of the free component parameter counts, plus
:math:`m-1` if weights are estimated. ``aic`` and ``bic`` use this nominal
count. Mixtures are nonregular at zero weights, coincident components and
some parameter boundaries. Repeated families also allow label switching.
These scores are useful descriptions, but ordinary chi-square
likelihood-ratio reference laws and naive Wald intervals are not justified
by this implementation. It does not automatically select the number of
components or provide mixture-parameter confidence intervals.

Bootstrap fitted copula parameters
-------------------------------------

``bootstrap_fit`` estimates sampling uncertainty for a **fixed bivariate
family**. It resamples paired rows (nonparametric bootstrap) or samples
from the fitted copula (parametric bootstrap), then repeats the fitting
procedure. Fixed parameters are preserved.

.. code-block:: python

   X = cp.StudentT(rho=0.5, nu=4).rvs(500, random_state=12)
   fitted = cp.fit(cp.StudentT(nu=4), X)
   boot = fitted.bootstrap(X, n_boot=200, random_state=17)
   boot.conf_int()
   boot.to_frame()          # estimates, standard errors, bias and intervals
   boot.cov                # covariance of the free parameter estimates

   parametric = cp.bootstrap_fit(
       cp.Gaussian, X, method="parametric", n_boot=200,
       ci_method="basic", random_state=19,
   )

For pseudo-likelihood fits, the columns are ranked **again in every
bootstrap replicate**. This propagates estimation of the marginal
distributions. The original ``FitResult`` records the family's fixed/free
parameters and whether ranks were estimated. ``fitted.bootstrap`` reuses
those conventions; changing them requires a new original fit.

If the data are uniforms from *known* marginal distributions, pass
``pseudo_obs=False`` when fitting, or directly to ``bootstrap_fit`` with
a family specification. Replicates then use the uniforms directly.
Passing False for previously estimated ranks conditions on those estimated
margins and omits their uncertainty.

For bootstrap estimates :math:`\hat\theta^{*(1)},\ldots,\hat\theta^{*(B)}`,
the covariance is their sample covariance with denominator :math:`B-1`,
and estimated bias is :math:`\overline{\theta^*}-\hat\theta`. If
:math:`q_p` denotes an empirical bootstrap quantile, the pointwise
:math:`1-\alpha` intervals are

.. math::

   I_{\mathrm{percentile}}=[q_{\alpha/2},q_{1-\alpha/2}],\qquad
   I_{\mathrm{basic}}=[2\hat\theta-q_{1-\alpha/2},
                       2\hat\theta-q_{\alpha/2}].

Choose ``ci_method="percentile"`` (default) or ``"basic"``. Basic intervals
can leave the parameter domain and are not silently clipped. These are
pointwise first-order intervals, not simultaneous confidence regions or
corrections for boundary/nonidentifiability problems. Use hundreds or more
replicates for final inference and check stability across seeds.

A failed or nonconverged refit raises by default. With explicit
``errors="omit"``, failed replicates remain NaN rows, their messages appear
in ``boot.failures``, and a warning reports omissions. ``boot.n_success``
counts usable replicates; at least two are required. Inference conditional
on successful refits can be biased. The procedure does not silently retry
until the requested number succeeds.

This bootstrap does not repeat family selection, vine structure selection,
mixture component selection or marginal-model selection. For dependent
time-series observations, ordinary row resampling is also insufficient;
no block bootstrap is claimed.

References
------------

* Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977).
  Maximum likelihood from incomplete data via the EM algorithm.
  *JRSS B* 39, 1--22.
  `doi:10.1111/j.2517-6161.1977.tb01600.x <https://doi.org/10.1111/j.2517-6161.1977.tb01600.x>`_.
* Efron, B. and Tibshirani, R. J. (1993).
  *An Introduction to the Bootstrap*. Chapman & Hall, ch. 13.
* Genest, C., Ghoudi, K. and Rivest, L.-P. (1995).
  A semiparametric estimation procedure of dependence parameters in
  multivariate families of distributions. *Biometrika* 82, 543--552.

API details are in :doc:`copul.stats`.
