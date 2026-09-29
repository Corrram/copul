r"""
Statistical inference for bivariate copulas.

Nonparametric
-------------
:func:`pseudo_obs`
    Rank transform to pseudo-observations :math:`R_{ij}/(n+1)`.
:class:`EmpiricalCopula`
    Empirical copula (vectorized cdf, empirical checkerboard and Bernstein
    copulas, plots) with sample versions of all dependence measures of
    :mod:`copul.measures` (:data:`SAMPLE_ESTIMATORS`).
:func:`estimate`
    Tidy table of estimates with asymptotic or bootstrap confidence
    intervals.
:func:`independence_test`
    Rank tests of independence (Kendall, Spearman, Chatterjee, Hoeffding,
    Cramér--von Mises).

Parametric
----------
:func:`fit`
    Maximum pseudo-likelihood and inversion of tau/rho/xi/beta for any
    parametric family, returning a :class:`FitResult`.
:func:`select`
    AIC/BIC ranking of several families.
:func:`gof_test`
    Goodness-of-fit test of Genest, Rémillard & Beaudoin (2009) with
    parametric bootstrap.

Examples
--------
>>> import copul as cp
>>> from copul import stats as cs
>>> X = cp.Clayton(theta=2).rvs(1000, random_state=0)
>>> cs.estimate(X, ["tau", "rho", "xi"], ci="asymptotic")   # doctest: +SKIP
>>> res = cs.fit(cp.Clayton, X)                             # doctest: +SKIP
>>> cs.select(X, ["Clayton", "Frank", "Gaussian"])          # doctest: +SKIP
>>> cs.gof_test(res, X, n_boot=100, random_state=0)         # doctest: +SKIP
"""

from copul.stats._adapters import sample
from copul.stats.empirical import EmpiricalCopula, checkerboard_mass
from copul.stats.estimators import (
    SAMPLE_ESTIMATORS,
    cramer_von_mises_independence,
    sample_beta,
    sample_bkr,
    sample_footrule,
    sample_gamma,
    sample_hoeffdings_d,
    sample_kappa,
    sample_lambda_l,
    sample_lambda_u,
    sample_lp,
    sample_measure,
    sample_mutual_information,
    sample_nu,
    sample_rho,
    sample_sigma,
    sample_tau,
    sample_xi,
    sample_xi_2,
)
from copul.stats.fitting import (
    DEFAULT_FAMILIES,
    FitResult,
    SingularFamilyError,
    fit,
    loglik,
    select,
)
from copul.stats.gof import GofResult, gof_statistic, gof_test
from copul.stats.inference import (
    ASYMPTOTIC_MEASURES,
    IndependenceTestResult,
    asymptotic_variance,
    bootstrap,
    estimate,
    independence_test,
)
from copul.stats.pseudo_obs import pseudo_obs, ranks

__all__ = [
    "ASYMPTOTIC_MEASURES",
    "DEFAULT_FAMILIES",
    "SAMPLE_ESTIMATORS",
    "EmpiricalCopula",
    "FitResult",
    "GofResult",
    "IndependenceTestResult",
    "SingularFamilyError",
    "asymptotic_variance",
    "bootstrap",
    "checkerboard_mass",
    "cramer_von_mises_independence",
    "estimate",
    "fit",
    "gof_statistic",
    "gof_test",
    "independence_test",
    "loglik",
    "pseudo_obs",
    "ranks",
    "sample",
    "sample_beta",
    "sample_bkr",
    "sample_footrule",
    "sample_gamma",
    "sample_hoeffdings_d",
    "sample_kappa",
    "sample_lambda_l",
    "sample_lambda_u",
    "sample_lp",
    "sample_measure",
    "sample_mutual_information",
    "sample_nu",
    "sample_rho",
    "sample_sigma",
    "sample_tau",
    "sample_xi",
    "sample_xi_2",
    "select",
]
