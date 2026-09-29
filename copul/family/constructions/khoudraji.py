r"""
Khoudraji's device for asymmetric copulas.

.. math::

   C(u,v) = C_1\bigl(u^{1-a}, v^{1-b}\bigr)\, C_2\bigl(u^{a}, v^{b}\bigr),
   \qquad a, b \in [0,1],

(Khoudraji 1995; Genest, Ghoudi & Rivest 1998).  With :math:`C_1 = \Pi`
this is the familiar one-copula version
:math:`C(u,v) = u^{1-a}v^{1-b}\,C_2(u^a, v^b)`, which is non-exchangeable
for :math:`a\neq b` even if :math:`C_2` is exchangeable.

Exact sampling (Liebscher 2008): if :math:`(X_1,Y_1)\sim C_1` and
:math:`(X_2,Y_2)\sim C_2` are independent, then

.. math::

   U = \max\bigl(X_1^{1/(1-a)}, X_2^{1/a}\bigr),\qquad
   V = \max\bigl(Y_1^{1/(1-b)}, Y_2^{1/b}\bigr)

has distribution :math:`C` (a power :math:`x^{1/0}` is read as :math:`0`).

Special cases: :math:`a=b=0` gives :math:`C_1`, :math:`a=b=1` gives
:math:`C_2`.  Blomqvist's :math:`\beta` is exact (from the cdf); all other
measures are evaluated numerically.

References
----------
Khoudraji, A. (1995). *Contributions à l'étude des copules et à la
modélisation des valeurs extrêmes bivariées*. PhD thesis, Université Laval.

Genest, C., Ghoudi, K. & Rivest, L.-P. (1998). Discussion of "Understanding
relationships using copulas" by Frees & Valdez. *North American Actuarial
Journal* 2(3), 143–149.

Liebscher, E. (2008). Construction of asymmetric multivariate copulas.
*Journal of Multivariate Analysis* 99, 2234–2250.
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

__all__ = ["KhoudrajiCopula", "khoudraji"]


def _pow(x, p):
    """``x ** p`` with ``0 ** 0 = 1`` and safe handling of ``x = 0``."""
    with np.errstate(all="ignore"):
        return np.where(p == 0, 1.0, np.power(x, p))


def _root(x, a):
    r""":math:`x^{1/a}` with :math:`x^{1/0} = 0` for :math:`x<1`."""
    if a <= 0:
        return np.zeros_like(x)
    return np.power(x, 1.0 / a)


class KhoudrajiCopula(NumericBivCopula):
    r"""Khoudraji copula :math:`C_1(u^{1-a}, v^{1-b})\,C_2(u^a, v^b)`.

    Parameters
    ----------
    C1, C2 : BivCopula
        Fully specified bivariate copulas.
    a, b : float
        Exponents in :math:`[0, 1]`.

    Notes
    -----
    With :math:`x_1 = u^{1-a}, y_1 = v^{1-b}, x_2 = u^a, y_2 = v^b`,

    .. math::

       \partial_1 C = (1-a)u^{-a}\,\partial_1 C_1(x_1,y_1)\,C_2(x_2,y_2)
                      + a u^{a-1}\,C_1(x_1,y_1)\,\partial_1 C_2(x_2,y_2),

    and the density (both components absolutely continuous) is

    .. math::

       c = (1-a)(1-b)u^{-a}v^{-b}c_1C_2
           + (1-a)b\,u^{-a}v^{b-1}\partial_1C_1\,\partial_2C_2
           + a(1-b)\,u^{a-1}v^{-b}\partial_2C_1\,\partial_1C_2
           + ab\,u^{a-1}v^{b-1}C_1c_2 .

    Examples
    --------
    >>> import copul as cp
    >>> from copul.family.constructions import khoudraji
    >>> K = khoudraji(cp.BivIndependenceCopula(), cp.GumbelHougaard(3), 0.3, 0.9)
    >>> K.is_symmetric
    False
    """

    def __init__(self, C1, C2, a: float, b: float):
        a, b = float(a), float(b)
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError(f"a and b must lie in [0, 1], got a={a}, b={b}")
        self.C1 = ensure_numeric_copula(C1, "C1")
        self.C2 = ensure_numeric_copula(C2, "C2")
        self.a, self.b = a, b
        self._f1 = component_callables(C1)
        self._f2 = component_callables(C2)
        self._ac = component_is_ac(C1) and component_is_ac(C2)
        super().__init__()

    def __repr__(self):
        return f"khoudraji({self.C1!r}, {self.C2!r}, a={self.a:.6g}, b={self.b:.6g})"

    __str__ = __repr__

    @property
    def is_absolutely_continuous(self) -> bool:
        return self._ac

    def _args(self, u, v):
        a, b = self.a, self.b
        return _pow(u, 1 - a), _pow(v, 1 - b), _pow(u, a), _pow(v, b)

    def _cdf(self, u, v):
        x1, y1, x2, y2 = self._args(u, v)
        return self._f1[0](x1, y1) * self._f2[0](x2, y2)

    def _h1(self, u, v):
        a = self.a
        x1, y1, x2, y2 = self._args(u, v)
        uu = np.maximum(u, 1e-300)
        out = 0.0
        if a < 1:
            out = out + (1 - a) * uu**-a * self._f1[1](x1, y1) * self._f2[0](x2, y2)
        if a > 0:
            out = out + a * uu ** (a - 1) * self._f1[0](x1, y1) * self._f2[1](x2, y2)
        return out

    def _h2(self, u, v):
        b = self.b
        x1, y1, x2, y2 = self._args(u, v)
        vv = np.maximum(v, 1e-300)
        out = 0.0
        if b < 1:
            out = out + (1 - b) * vv**-b * self._f1[2](x1, y1) * self._f2[0](x2, y2)
        if b > 0:
            out = out + b * vv ** (b - 1) * self._f1[0](x1, y1) * self._f2[2](x2, y2)
        return out

    def _pdf(self, u, v):
        a, b = self.a, self.b
        x1, y1, x2, y2 = self._args(u, v)
        uu, vv = np.maximum(u, 1e-300), np.maximum(v, 1e-300)
        C1, h11, h12 = (f(x1, y1) for f in self._f1)
        C2, h21, h22 = (f(x2, y2) for f in self._f2)
        out = 0.0
        if a < 1 and b < 1:
            c1 = component_pdf(self.C1)(x1, y1)
            out = out + (1 - a) * (1 - b) * uu**-a * vv**-b * c1 * C2
        if a < 1 and b > 0:
            out = out + (1 - a) * b * uu**-a * vv ** (b - 1) * h11 * h22
        if a > 0 and b < 1:
            out = out + a * (1 - b) * uu ** (a - 1) * vv**-b * h12 * h21
        if a > 0 and b > 0:
            c2 = component_pdf(self.C2)(x2, y2)
            out = out + a * b * uu ** (a - 1) * vv ** (b - 1) * C1 * c2
        return out

    def _rvs(self, n, rng):
        s1 = component_rvs(self.C1, n, rng)
        s2 = component_rvs(self.C2, n, rng)
        a, b = self.a, self.b
        u = np.maximum(_root(s1[:, 0], 1 - a), _root(s2[:, 0], a))
        v = np.maximum(_root(s1[:, 1], 1 - b), _root(s2[:, 1], b))
        return np.column_stack([u, v])


def khoudraji(C1, C2, a: float, b: float) -> KhoudrajiCopula:
    r"""Khoudraji copula :math:`C_1(u^{1-a}, v^{1-b})\,C_2(u^a, v^b)`.

    Pass :class:`~copul.family.frechet.biv_independence_copula.BivIndependenceCopula`
    as ``C1`` for the classical asymmetrisation
    :math:`u^{1-a}v^{1-b}C_2(u^a, v^b)`.  See :class:`KhoudrajiCopula`.
    """
    return KhoudrajiCopula(C1, C2, a, b)
