r"""
Ordinal sums of bivariate copulas (Nelsen 2006, Def. 3.2.2).

Given non-overlapping intervals :math:`J_i = [a_i, b_i]\subseteq[0,1]`
with lengths :math:`\ell_i = b_i - a_i > 0` and copulas :math:`C_i`,

.. math::

   C(u,v) =
   \begin{cases}
     a_i + \ell_i\, C_i\!\Bigl(\dfrac{u-a_i}{\ell_i}, \dfrac{v-a_i}{\ell_i}\Bigr),
       & (u,v)\in J_i^2,\\[1ex]
     \min(u,v), & \text{otherwise}.
   \end{cases}

The mass of :math:`C` is the scaled mass of :math:`C_i` on the squares
:math:`J_i^2` and uniform mass on the diagonal outside :math:`\bigcup_i J_i`.

Measure relations (derived from the definitions, verified in the tests)
-----------------------------------------------------------------------
.. math::

   \rho(C) &= 1 - \sum_i \ell_i^3\,\bigl(1-\rho(C_i)\bigr), \\
   \tau(C) &= 1 - \sum_i \ell_i^2\,\bigl(1-\tau(C_i)\bigr), \\
   \phi(C) &= 1 - \sum_i \ell_i^2\,\bigl(1-\phi(C_i)\bigr),

(all follow from the integral representations of the measures, since
:math:`C - M` and :math:`\partial_1 C\,\partial_2 C` vanish off the squares
:math:`J_i^2`).  :math:`\lambda_L(C) = \lambda_L(C_1)` if :math:`a_1 = 0` and
:math:`1` otherwise; :math:`\lambda_U(C) = \lambda_U(C_k)` if
:math:`b_k = 1` and :math:`1` otherwise (intervals sorted).

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
Sec. 3.2.2.
"""

from __future__ import annotations

import itertools

import numpy as np

from copul.family.constructions._base import (
    NumericBivCopula,
    component_callables,
    component_is_ac,
    component_pdf,
    component_rvs,
    ensure_numeric_copula,
)

__all__ = ["OrdinalSumCopula", "ordinal_sum"]


class OrdinalSumCopula(NumericBivCopula):
    r"""Ordinal sum of copulas :math:`C_i` with respect to intervals :math:`[a_i,b_i]`.

    Parameters
    ----------
    components : sequence of (a, b, C)
        Non-overlapping intervals :math:`0\le a<b\le 1` with copulas.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.family.constructions import ordinal_sum
    >>> C = ordinal_sum([(0.0, 0.5, cp.BivIndependenceCopula()),
    ...                  (0.5, 1.0, cp.BivIndependenceCopula())])
    >>> round(C.spearmans_rho(), 12)   # 1 - 2 * 0.5**3
    0.75
    """

    def __init__(self, components):
        comps = []
        for k, item in enumerate(components):
            a, b, C = item
            a, b = float(a), float(b)
            if not (0.0 <= a < b <= 1.0):
                raise ValueError(f"interval {k} must satisfy 0 <= a < b <= 1, got [{a}, {b}]")
            comps.append((a, b, ensure_numeric_copula(C, f"components[{k}]")))
        if not comps:
            raise ValueError("ordinal_sum needs at least one component")
        comps.sort(key=lambda t: t[0])
        for (_, b0, _), (a1, _, _) in itertools.pairwise(comps):
            if a1 < b0 - 1e-15:
                raise ValueError("intervals of an ordinal sum must not overlap")
        self.components = comps
        self._fns = [component_callables(C) for _, _, C in comps]
        covered = sum(b - a for a, b, _ in comps)
        self._ac = abs(covered - 1.0) < 1e-12 and all(component_is_ac(C) for _, _, C in comps)
        super().__init__()

    def __repr__(self):
        parts = ", ".join(f"([{a:.6g}, {b:.6g}], {C!r})" for a, b, C in self.components)
        return f"ordinal_sum({parts})"

    __str__ = __repr__

    @property
    def is_absolutely_continuous(self) -> bool:
        return self._ac

    def _loop(self, u, v, which, outside):
        out = np.array(outside(u, v), dtype=float, copy=True)
        for (a, b, _), fns in zip(self.components, self._fns):
            m = (u >= a) & (u <= b) & (v >= a) & (v <= b)
            if not np.any(m):
                continue
            ell = b - a
            x = np.clip((u[m] - a) / ell, 0.0, 1.0)
            y = np.clip((v[m] - a) / ell, 0.0, 1.0)
            val = fns[which](x, y)
            out[m] = a + ell * val if which == 0 else val
        return out

    def _cdf(self, u, v):
        return self._loop(u, v, 0, np.minimum)

    def _h1(self, u, v):
        return self._loop(u, v, 1, lambda s, t: (s < t).astype(float))

    def _h2(self, u, v):
        return self._loop(u, v, 2, lambda s, t: (t < s).astype(float))

    def _pdf(self, u, v):
        out = np.zeros(np.shape(u))
        for a, b, C in self.components:
            m = (u >= a) & (u <= b) & (v >= a) & (v <= b)
            if not np.any(m):
                continue
            ell = b - a
            x = np.clip((u[m] - a) / ell, 0.0, 1.0)
            y = np.clip((v[m] - a) / ell, 0.0, 1.0)
            out[m] = component_pdf(C)(x, y) / ell
        return out

    def _rvs(self, n, rng):
        lengths = np.array([b - a for a, b, _ in self.components])
        rest = max(0.0, 1.0 - lengths.sum())
        probs = np.append(lengths, rest)
        probs = probs / probs.sum()
        counts = rng.multinomial(n, probs)
        parts = []
        for (a, b, C), k in zip(self.components, counts[:-1]):
            if k > 0:
                s = component_rvs(C, k, rng)
                parts.append(a + (b - a) * s)
        k = counts[-1]
        if k > 0:
            # uniform on the complement of the intervals, on the diagonal
            gaps, lo = [], 0.0
            for a, b, _ in self.components:
                if a > lo:
                    gaps.append((lo, a))
                lo = b
            if lo < 1.0:
                gaps.append((lo, 1.0))
            glen = np.array([g1 - g0 for g0, g1 in gaps])
            idx = rng.choice(len(gaps), size=k, p=glen / glen.sum())
            t = np.array([g[0] for g in gaps])[idx] + glen[idx] * rng.random(k)
            parts.append(np.column_stack([t, t]))
        out = np.concatenate(parts, axis=0)
        return out[rng.permutation(n)]

    # -- measure relations ----------------------------------------------------------
    def _sum(self, power, name):
        return sum(
            (b - a) ** power * (1.0 - float(getattr(C, name)())) for a, b, C in self.components
        )

    def spearmans_rho(self, *args, **kwargs):
        r""":math:`1 - \sum_i \ell_i^3(1-\rho(C_i))`."""
        return 1.0 - self._sum(3, "spearmans_rho")

    def kendalls_tau(self, *args, **kwargs):
        r""":math:`1 - \sum_i \ell_i^2(1-\tau(C_i))`."""
        return 1.0 - self._sum(2, "kendalls_tau")

    def spearmans_footrule(self, *args, **kwargs):
        r""":math:`1 - \sum_i \ell_i^2(1-\phi(C_i))`."""
        return 1.0 - self._sum(2, "spearmans_footrule")

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L(C_1)` if the first interval starts at 0, else 1."""
        a, _, C = self.components[0]
        return float(C.lambda_L()) if a == 0.0 else 1.0

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U(C_k)` if the last interval ends at 1, else 1."""
        _, b, C = self.components[-1]
        return float(C.lambda_U()) if b == 1.0 else 1.0


def ordinal_sum(components) -> OrdinalSumCopula:
    r"""Ordinal sum of ``[(a_1, b_1, C_1), ..., (a_k, b_k, C_k)]`` (M outside).

    See :class:`OrdinalSumCopula`.
    """
    return OrdinalSumCopula(components)
