r"""
Overflow-free evaluation of Archimedean copulas with exponential-type generators.

Nelsen's families 4.2.19 and 4.2.20 have generators
:math:`\varphi(t)=e^{g(t)}-e^{g(1)}` with :math:`g(t)=\theta/t` resp.
:math:`g(t)=t^{-\theta}`, so that

.. math::

   C(u,v) = g^{-1}(L),\qquad
   L = \log\bigl(e^{g(u)}+e^{g(v)}-e^{g(1)}\bigr).

For small :math:`u` the exponentials overflow in double precision.  Writing
:math:`m=\max(g(u),g(v))`,
:math:`L = m+\log(e^{g(u)-m}+e^{g(v)-m}-e^{g(1)-m})` is evaluated without
overflow, and

.. math::

   \partial_1 C = \frac{g'(u)}{g'(C)}\,e^{g(u)-L},\qquad
   c(u,v) = -g'(u)g'(v)\,e^{g(u)+g(v)-2L}
            \Bigl(\frac1{g'(C)}+\frac{g''(C)}{g'(C)^3}\Bigr).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class ExpGenerator:
    """Stable cdf, conditional distributions and density for
    :math:`\\varphi(t)=e^{g(t)}-e^{g(1)}`."""

    def __init__(
        self,
        g: Callable[[np.ndarray], np.ndarray],
        dg: Callable[[np.ndarray], np.ndarray],
        d2g: Callable[[np.ndarray], np.ndarray],
        g_inv: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.g, self.dg, self.d2g, self.g_inv = g, dg, d2g, g_inv
        self.g1 = float(g(np.array(1.0)))

    def _prep(self, u, v):
        u, v = np.broadcast_arrays(
            np.clip(np.asarray(u, dtype=float), 0.0, 1.0),
            np.clip(np.asarray(v, dtype=float), 0.0, 1.0),
        )
        inner = (u > 0) & (v > 0)
        uu = np.where(inner, u, 0.5)
        vv = np.where(inner, v, 0.5)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gu, gv = self.g(uu), self.g(vv)
            m = np.maximum(gu, gv)
            L = m + np.log(np.exp(gu - m) + np.exp(gv - m) - np.exp(self.g1 - m))
            C = np.clip(self.g_inv(L), 0.0, np.minimum(uu, vv))
        return u, v, inner, uu, vv, gu, gv, L, C

    def cdf(self, u, v):
        u, v, inner, *_rest, C = self._prep(u, v)
        return np.where(inner, C, 0.0)

    def _h(self, first: bool, u, v):
        u, v, inner, uu, vv, gu, gv, L, C = self._prep(u, v)
        x, gx = (uu, gu) if first else (vv, gv)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            h = self.dg(x) / self.dg(C) * np.exp(gx - L)
        h = np.clip(np.nan_to_num(h, nan=0.0), 0.0, 1.0)
        other = v if first else u
        x_full = u if first else v
        # boundary values: C(0, v) = 0 -> dC/du(0, v) = 1 for these families
        # (lambda_L = 1); dC/du(u, 0) = 0; dC/du(u, 1) = 1
        out = np.where(inner, h, 0.0)
        out = np.where((x_full == 0) & (other > 0), 1.0, out)
        out = np.where(other >= 1.0, 1.0, out)
        return out

    def h1(self, u, v):
        return self._h(True, u, v)

    def h2(self, u, v):
        return self._h(False, u, v)

    def pdf(self, u, v):
        u, v, inner, uu, vv, gu, gv, L, C = self._prep(u, v)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            dC = self.dg(C)
            c = (
                -self.dg(uu)
                * self.dg(vv)
                * np.exp(gu + gv - 2.0 * L)
                * (1.0 / dC + self.d2g(C) / dC**3)
            )
        c = np.nan_to_num(c, nan=0.0, posinf=0.0)
        return np.where(inner, np.maximum(c, 0.0), 0.0)

    def callables(self) -> dict:
        return {"cdf": self.cdf, "h1": self.h1, "h2": self.h2, "pdf": self.pdf}


def nelsen19(theta: float) -> ExpGenerator:
    r""":math:`g(t)=\theta/t`."""
    th = float(theta)
    return ExpGenerator(
        lambda t: th / t,
        lambda t: -th / t**2,
        lambda t: 2.0 * th / t**3,
        lambda L: th / L,
    )


def nelsen20(theta: float) -> ExpGenerator:
    r""":math:`g(t)=t^{-\theta}`."""
    th = float(theta)
    return ExpGenerator(
        lambda t: t ** (-th),
        lambda t: -th * t ** (-th - 1.0),
        lambda t: th * (th + 1.0) * t ** (-th - 2.0),
        lambda L: L ** (-1.0 / th),
    )
