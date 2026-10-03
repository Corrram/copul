r"""
Multivariate (:math:`d`-dimensional) copulas and dependence measures.

All classes derive from :class:`CopulaND` and share a vectorized numerical
API (``cdf``, ``pdf``, ``logpdf``, ``survival_function``, ``rvs``,
``h_volume``, ``margin``, ``survival_copula``, ``is_copula``) plus the
multivariate measures of :mod:`copul.multivariate.measures`.

==============================  =====================================================
object                          description
==============================  =====================================================
:class:`IndependenceND`         product copula :math:`\Pi_d`
:class:`UpperFrechetND`         comonotonicity copula :math:`M_d`
:class:`GaussianND`             Gaussian copula with correlation matrix :math:`R`
:class:`StudentTND`             Student-t copula :math:`(R,\nu)`
:class:`ArchimedeanCopulaND`    Archimedean copulas from a d-monotone generator
                                (Clayton, Gumbel, Frank, Joe, AMH, SymPy generators)
:class:`MixtureND`              convex combinations of d-copulas
:class:`FunctionalCopulaND`     d-copula from vectorized callables
:func:`as_copula_nd`            view any copul copula object as a :class:`CopulaND`
:func:`margin`                  margins :math:`C(u_I,\mathbf 1)` of any d-copula
:func:`is_copula_nd`            grid check of grounding, margins, d-increasingness
:func:`is_d_monotone`           d-monotonicity of Archimedean generators
:func:`spearmans_rho_nd`        Schmid--Schmidt :math:`\rho_1,\rho_2,\rho_3`
:func:`kendalls_tau_nd`         Nelsen's multivariate :math:`\tau_d`
:func:`blomqvists_beta_nd`      multivariate Blomqvist :math:`\beta_d`
==============================  =====================================================

Examples
--------
>>> import numpy as np
>>> from copul.multivariate import ClaytonND, GaussianND, kendalls_tau_nd
>>> C = ClaytonND(theta=2.0, dim=3)
>>> X = C.rvs(2000, random_state=0)
>>> round(C.kendalls_tau(), 6)            # exact (Kendall distribution)
0.5
>>> abs(kendalls_tau_nd(X) - 0.5) < 0.05  # sample version
True
>>> G = GaussianND(np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1]]))
>>> round(G.blomqvists_beta(), 4)        # 2 * sum(arcsin R_ij) / (3 pi)
0.2631
"""

from copul.multivariate.archimedean import (
    AliMikhailHaqND,
    ArchimedeanCopulaND,
    ClaytonND,
    FrankND,
    GumbelND,
    JoeND,
    is_d_monotone,
)
from copul.multivariate.base import (
    BivariateCopulaND,
    BivariateMarginCopula,
    CopulaND,
    FunctionalCopulaND,
    MarginalCopulaND,
    SurvivalCopulaND,
    frechet_hoeffding_bounds,
)
from copul.multivariate.basic import (
    IndependenceND,
    MixtureND,
    UpperFrechetND,
    as_copula_nd,
    margin,
)
from copul.multivariate.elliptical import (
    EllipticalCopulaND,
    GaussianND,
    StudentTND,
    nearest_correlation,
)
from copul.multivariate.measures import (
    blomqvists_beta_nd,
    kendalls_tau_nd,
    sample_blomqvists_beta_nd,
    sample_kendalls_tau_nd,
    sample_spearmans_rho_nd,
    spearmans_rho_h,
    spearmans_rho_nd,
)
from copul.multivariate.validation import is_copula_nd

__all__ = [
    "AliMikhailHaqND",
    "ArchimedeanCopulaND",
    "BivariateCopulaND",
    "BivariateMarginCopula",
    "ClaytonND",
    "CopulaND",
    "EllipticalCopulaND",
    "FrankND",
    "FunctionalCopulaND",
    "GaussianND",
    "GumbelND",
    "IndependenceND",
    "JoeND",
    "MarginalCopulaND",
    "MixtureND",
    "StudentTND",
    "SurvivalCopulaND",
    "UpperFrechetND",
    "as_copula_nd",
    "blomqvists_beta_nd",
    "frechet_hoeffding_bounds",
    "is_copula_nd",
    "is_d_monotone",
    "kendalls_tau_nd",
    "margin",
    "nearest_correlation",
    "sample_blomqvists_beta_nd",
    "sample_kendalls_tau_nd",
    "sample_spearmans_rho_nd",
    "spearmans_rho_h",
    "spearmans_rho_nd",
]
