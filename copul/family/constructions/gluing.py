r"""
Gluing of two bivariate copulas (Siburg & Stoimenov 2008).

For :math:`\theta\in(0,1)` the copulas :math:`C_1` and :math:`C_2` are
squeezed into the strips :math:`[0,\theta]\times[0,1]` and
:math:`[\theta,1]\times[0,1]`:

.. math::

   C(u,v) =
   \begin{cases}
     \theta\, C_1\!\bigl(u/\theta,\, v\bigr), & u\le\theta,\\[0.5ex]
     \theta v + (1-\theta)\, C_2\!\Bigl(\dfrac{u-\theta}{1-\theta},\, v\Bigr), & u>\theta.
   \end{cases}

Sampling: with probability :math:`\theta` draw :math:`(X,Y)\sim C_1` and
return :math:`(\theta X, Y)`, otherwise :math:`(X,Y)\sim C_2` and
:math:`(\theta+(1-\theta)X, Y)`.

Measure relations (from the integral representations; verified numerically)
---------------------------------------------------------------------------
.. math::

   \rho(C) &= \theta^2\rho(C_1) + (1-\theta)^2\rho(C_2),\qquad
   \tau(C)  = \theta^2\tau(C_1) + (1-\theta)^2\tau(C_2).

Gluing in the second coordinate (``axis="v"``) is the transpose of the
gluing of the transposes.

References
----------
Siburg, K. F. & Stoimenov, P. A. (2008). Gluing copulas.
*Communications in Statistics – Theory and Methods* 37(19), 3124–3134.
"""

from __future__ import annotations

import numpy as np

from copul.family.constructions._base import (
    NumericBivCopula,
    component_callables,
    component_is_ac,
    component_pdf,
    component_rvs,
    ensure_numeric_copula,
)

__all__ = ["GluingCopula", "gluing"]


class GluingCopula(NumericBivCopula):
    r"""Gluing of :math:`C_1` (on :math:`u\le\theta`) and :math:`C_2` (on :math:`u>\theta`).

    Parameters
    ----------
    C1, C2 : BivCopula
        Fully specified bivariate copulas.
    theta : float
        Gluing point in :math:`(0, 1)`.
    """

    def __init__(self, C1, C2, theta: float):
        theta = float(theta)
        if not 0.0 < theta < 1.0:
            raise ValueError(f"theta must lie in (0, 1), got {theta}")
        self.C1 = ensure_numeric_copula(C1, "C1")
        self.C2 = ensure_numeric_copula(C2, "C2")
        self.theta = theta
        self._f1 = component_callables(C1)
        self._f2 = component_callables(C2)
        self._ac = component_is_ac(C1) and component_is_ac(C2)
        super().__init__()

    def __repr__(self):
        return f"gluing({self.C1!r}, {self.C2!r}, theta={self.theta:.6g})"

    __str__ = __repr__

    @property
    def is_absolutely_continuous(self) -> bool:
        return self._ac

    def _split(self, u):
        th = self.theta
        low = u <= th
        x1 = np.clip(u / th, 0.0, 1.0)
        x2 = np.clip((u - th) / (1.0 - th), 0.0, 1.0)
        return low, x1, x2

    def _cdf(self, u, v):
        th = self.theta
        low, x1, x2 = self._split(u)
        return np.where(low, th * self._f1[0](x1, v), th * v + (1 - th) * self._f2[0](x2, v))

    def _h1(self, u, v):
        low, x1, x2 = self._split(u)
        return np.where(low, self._f1[1](x1, v), self._f2[1](x2, v))

    def _h2(self, u, v):
        th = self.theta
        low, x1, x2 = self._split(u)
        return np.where(low, th * self._f1[2](x1, v), th + (1 - th) * self._f2[2](x2, v))

    def _pdf(self, u, v):
        low, x1, x2 = self._split(u)
        return np.where(low, component_pdf(self.C1)(x1, v), component_pdf(self.C2)(x2, v))

    def _rvs(self, n, rng):
        th = self.theta
        k = rng.binomial(n, th)
        s1 = component_rvs(self.C1, k, rng)
        s2 = component_rvs(self.C2, n - k, rng)
        out = np.concatenate(
            [
                np.column_stack([th * s1[:, 0], s1[:, 1]]),
                np.column_stack([th + (1 - th) * s2[:, 0], s2[:, 1]]),
            ]
        )
        return out[rng.permutation(n)]

    # -- measure relations -------------------------------------------------------
    def spearmans_rho(self, *args, **kwargs):
        r""":math:`\theta^2\rho(C_1) + (1-\theta)^2\rho(C_2)`."""
        th = self.theta
        return th**2 * float(self.C1.spearmans_rho()) + (1 - th) ** 2 * float(
            self.C2.spearmans_rho()
        )

    def kendalls_tau(self, *args, **kwargs):
        r""":math:`\theta^2\tau(C_1) + (1-\theta)^2\tau(C_2)`."""
        th = self.theta
        return th**2 * float(self.C1.kendalls_tau()) + (1 - th) ** 2 * float(self.C2.kendalls_tau())


def gluing(C1, C2, theta: float, axis: str = "u"):
    r"""Glue :math:`C_1` and :math:`C_2` at ``theta`` (Siburg & Stoimenov 2008).

    Parameters
    ----------
    C1, C2 : BivCopula
        Fully specified bivariate copulas.
    theta : float
        Gluing point in :math:`(0,1)`.
    axis : {"u", "v"}
        Coordinate that is split (``"u"``: see :class:`GluingCopula`;
        ``"v"``: :math:`\theta C_1(u, v/\theta)` for :math:`v\le\theta` and
        :math:`\theta u + (1-\theta)C_2(u, (v-\theta)/(1-\theta))` otherwise).
    """
    if axis == "u":
        return GluingCopula(C1, C2, theta)
    if axis == "v":
        from copul.family.constructions.rotation import transpose

        return transpose(GluingCopula(transpose(C1), transpose(C2), theta))
    raise ValueError(f"axis must be 'u' or 'v', got {axis!r}")
