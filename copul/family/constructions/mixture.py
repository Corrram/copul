r"""
Finite mixtures (convex combinations) of bivariate copulas.

.. math::

   C(u,v) = \sum_{i=1}^k w_i\, C_i(u,v), \qquad w_i\ge 0,\ \sum_i w_i = 1 .

The set of copulas is convex, so :math:`C` is a copula; it is sampled
exactly by drawing the component index with probabilities :math:`w_i`.
Conditional distributions and the density (if every component is
absolutely continuous) are the same convex combinations.

Measure relations
-----------------
Every measure that is an affine functional of :math:`C` with value
:math:`\mu(C) = a\,L(C) + b` and :math:`\mu(\Pi)` well defined is
*linear* in the mixture weights:

* Spearman's :math:`\rho = 12\int\!\!\int C - 3`,
* Blest's :math:`\nu = 24\int\!\!\int (1-u)C - 2`,
* Spearman's footrule :math:`\phi = 6\int_0^1 C(t,t)\,dt - 2`,
* Gini's :math:`\gamma = 4\int_0^1 [C(t,t)+C(t,1-t)]\,dt - 2`,
* Blomqvist's :math:`\beta = 4C(\tfrac12,\tfrac12) - 1`,
* the tail coefficients :math:`\lambda_L, \lambda_U` (limits of linear
  functionals),

so :math:`\mu(C) = \sum_i w_i\,\mu(C_i)`.  Kendall's :math:`\tau`,
Chatterjee's :math:`\xi` and the distances to independence are not linear
and are evaluated numerically.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
Sec. 5.1 (Exercises 5.3, 5.8: convex combinations).
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

__all__ = ["MixtureCopula", "mixture"]


class MixtureCopula(NumericBivCopula):
    r"""Convex combination :math:`\sum_i w_i C_i` of bivariate copulas.

    Parameters
    ----------
    copulas : sequence of BivCopula
        Fully specified bivariate copulas :math:`C_1,\dots,C_k`.
    weights : sequence of float, optional
        Non-negative weights (normalised to sum 1); equal weights by default.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.family.constructions import mixture
    >>> C = mixture([cp.Clayton(2), cp.UpperFrechet()], [0.7, 0.3])
    >>> round(C.lambda_L(), 12) == round(0.7 * 2 ** -0.5 + 0.3, 12)
    True
    """

    def __init__(self, copulas, weights=None):
        copulas = list(copulas)
        if not copulas:
            raise ValueError("mixture needs at least one copula")
        for i, C in enumerate(copulas):
            ensure_numeric_copula(C, f"copulas[{i}]")
        if weights is None:
            w = np.full(len(copulas), 1.0 / len(copulas))
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.size != len(copulas):
                raise ValueError("weights and copulas must have the same length")
            if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
                raise ValueError("weights must be non-negative and not all zero")
            w = w / w.sum()
        self.copulas = copulas
        self.weights = w
        self._active = [i for i in range(len(copulas)) if w[i] > 0]
        self._fns = {i: component_callables(copulas[i]) for i in self._active}
        self._ac = all(component_is_ac(copulas[i]) for i in self._active)
        super().__init__()

    def __repr__(self):
        parts = ", ".join(f"{w:.6g}*{C!r}" for w, C in zip(self.weights, self.copulas))
        return f"mixture({parts})"

    __str__ = __repr__

    @property
    def is_absolutely_continuous(self) -> bool:
        return self._ac

    def _combine(self, which, u, v):
        out = 0.0
        for i in self._active:
            out = out + self.weights[i] * self._fns[i][which](u, v)
        return out

    def _cdf(self, u, v):
        return self._combine(0, u, v)

    def _h1(self, u, v):
        return self._combine(1, u, v)

    def _h2(self, u, v):
        return self._combine(2, u, v)

    def _pdf(self, u, v):
        out = 0.0
        for i in self._active:
            out = out + self.weights[i] * component_pdf(self.copulas[i])(u, v)
        return out

    def _rvs(self, n, rng):
        counts = rng.multinomial(n, self.weights)
        parts = [component_rvs(C, k, rng) for C, k in zip(self.copulas, counts) if k > 0]
        out = np.concatenate(parts, axis=0)
        return out[rng.permutation(n)]

    # -- linear measures ---------------------------------------------------------
    def _linear(self, name):
        return float(
            sum(self.weights[i] * float(getattr(self.copulas[i], name)()) for i in self._active)
        )

    def spearmans_rho(self, *args, **kwargs):
        r""":math:`\rho(C) = \sum_i w_i \rho(C_i)`."""
        return self._linear("spearmans_rho")

    def blests_nu(self, *args, **kwargs):
        r""":math:`\nu(C) = \sum_i w_i \nu(C_i)`."""
        return self._linear("blests_nu")

    def spearmans_footrule(self, *args, **kwargs):
        r""":math:`\phi(C) = \sum_i w_i \phi(C_i)`."""
        return self._linear("spearmans_footrule")

    def ginis_gamma(self, *args, **kwargs):
        r""":math:`\gamma(C) = \sum_i w_i \gamma(C_i)`."""
        return self._linear("ginis_gamma")

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L(C) = \sum_i w_i \lambda_L(C_i)`."""
        return self._linear("lambda_L")

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U(C) = \sum_i w_i \lambda_U(C_i)`."""
        return self._linear("lambda_U")


def mixture(copulas, weights=None) -> MixtureCopula:
    r"""Convex combination :math:`\sum_i w_i C_i` of bivariate copulas.

    See :class:`MixtureCopula`.
    """
    return MixtureCopula(copulas, weights)
