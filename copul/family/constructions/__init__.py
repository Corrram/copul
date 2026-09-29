r"""
Constructions of new bivariate copulas from given ones.

Every construction returns a :class:`~copul.family.constructions.NumericBivCopula`
(a :class:`~copul.family.core.biv_copula.BivCopula`) with vectorised
``cdf``/``cond_distr_1``/``cond_distr_2``/``pdf``, exact sampling
(``rvs(n, random_state)``) and all dependence measures.  Where a measure of
the construction is an exact function of measures of the components, the
closed relation is used (``method="closed"``); everything else goes through
the numerical engine (``method="numeric"``).

============================  ============================================================
function                      copula
============================  ============================================================
``rotate(C, angle)``          rotation of the scatter plot by 90/180/270 degrees
``reflect(C, axis)``          :math:`(1-U,V)`, :math:`(U,1-V)`, survival, (anti)diagonal
``transpose(C)``              :math:`C(v,u)`
``survival(C)``               :math:`u+v-1+C(1-u,1-v)`
``mixture(Cs, weights)``      :math:`\sum_i w_i C_i`
``khoudraji(C1, C2, a, b)``   :math:`C_1(u^{1-a},v^{1-b})\,C_2(u^a,v^b)`
``ordinal_sum(parts)``        Nelsen's ordinal sum with :math:`M` outside the squares
``gluing(C1, C2, theta)``     Siburg–Stoimenov gluing at :math:`\theta`
``markov_product(A, B)``      Markov (:math:`*`-) product (from :mod:`copul.star_product`)
============================  ============================================================

Examples
--------
>>> import copul as cp
>>> from copul.family.constructions import khoudraji, mixture, rotate
>>> C = rotate(cp.GumbelHougaard(2), 180)          # survival Gumbel
>>> round(C.lambda_L(), 12) == round(2 - 2 ** 0.5, 12)
True
>>> K = khoudraji(cp.BivIndependenceCopula(), cp.Clayton(3), 0.4, 0.9)
>>> K.rvs(5, random_state=0).shape
(5, 2)
>>> M = mixture([cp.Clayton(2), rotate(cp.Clayton(2), 180)], [0.5, 0.5])
>>> round(M.lambda_L(), 12) == round(M.lambda_U(), 12)
True
"""

from copul.family.constructions._base import NumericBivCopula
from copul.family.constructions.gluing import GluingCopula, gluing
from copul.family.constructions.khoudraji import KhoudrajiCopula, khoudraji
from copul.family.constructions.mixture import MixtureCopula, mixture
from copul.family.constructions.ordinal_sum import OrdinalSumCopula, ordinal_sum
from copul.family.constructions.rotation import (
    TransformedCopula,
    reflect,
    rotate,
    survival,
    transpose,
)
from copul.star_product import markov_product

__all__ = [
    "GluingCopula",
    "KhoudrajiCopula",
    "MixtureCopula",
    "NumericBivCopula",
    "OrdinalSumCopula",
    "TransformedCopula",
    "gluing",
    "khoudraji",
    "markov_product",
    "mixture",
    "ordinal_sum",
    "reflect",
    "rotate",
    "survival",
    "transpose",
]
