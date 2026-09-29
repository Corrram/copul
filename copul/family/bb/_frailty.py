r"""
Random variates of the frailty (mixing) distributions of Archimedean copulas.

An Archimedean copula :math:`C(u,v) = \psi(\psi^{-1}(u)+\psi^{-1}(v))` whose
generator inverse :math:`\psi` is the Laplace transform of a positive random
variable :math:`V` is sampled exactly by the Marshall–Olkin algorithm:
draw :math:`V`, independent :math:`E_1, E_2 \sim \mathrm{Exp}(1)` and set
:math:`U_i = \psi(E_i / V)` (Marshall & Olkin 1988).  For the BB families
the frailty is a *compound* of two classical ones (Joe 2014, Sec. 3.2):
if :math:`\psi = \psi_1(-\log\psi_2)` then :math:`V \mid M \sim` (the
distribution with Laplace transform :math:`\psi_2^{M}`) and :math:`M` has
Laplace transform :math:`\psi_1`.

All samplers return :math:`\log V` (frailties can be astronomically small
or large).

References
----------
Marshall, A. W. & Olkin, I. (1988). Families of multivariate distributions.
*JASA* 83, 834–841.

Kanter, M. (1975). Stable densities under change of scale and total
variation inequalities. *Annals of Probability* 3, 697–707.

Hofert, M. (2011). Efficiently sampling nested Archimedean copulas.
*Computational Statistics & Data Analysis* 55, 57–70.

Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "log_gamma_rv",
    "log_positive_stable_rv",
    "log_tilted_stable_rv",
    "sibuya_rv",
]


def log_gamma_rv(shape, rng: np.random.Generator) -> np.ndarray:
    r""":math:`\log G` for :math:`G\sim\mathrm{Gamma}(\text{shape}, 1)` (elementwise shapes).

    Small shapes use :math:`G \overset{d}{=} G' U^{1/k}` with
    :math:`G'\sim\mathrm{Gamma}(k+1)` to avoid underflow.
    """
    shape = np.asarray(shape, dtype=float)
    small = shape < 1.0
    k = np.where(small, shape + 1.0, shape)
    g = rng.gamma(k)
    out = np.log(g)
    if np.any(small):
        u = rng.random(np.shape(shape))
        with np.errstate(divide="ignore"):
            out = np.where(small, out + np.log(u) / np.where(small, shape, 1.0), out)
    return out


def log_positive_stable_rv(alpha: float, size, rng: np.random.Generator) -> np.ndarray:
    r""":math:`\log S` for the positive stable law with Laplace transform :math:`e^{-s^\alpha}`.

    Kanter's (1975) representation, :math:`\alpha\in(0,1]`:

    .. math::

       S = \frac{\sin(\alpha U)}{(\sin U)^{1/\alpha}}
           \Bigl(\frac{\sin((1-\alpha)U)}{W}\Bigr)^{(1-\alpha)/\alpha},
       \quad U\sim\mathcal U(0,\pi),\ W\sim\mathrm{Exp}(1).
    """
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    if alpha == 1.0:
        return np.zeros(size)
    u = np.pi * rng.random(size)
    w = rng.exponential(size=size)
    with np.errstate(divide="ignore"):
        return (
            np.log(np.sin(alpha * u))
            - np.log(np.sin(u)) / alpha
            + (1.0 - alpha) / alpha * (np.log(np.sin((1.0 - alpha) * u)) - np.log(w))
        )


def sibuya_rv(alpha: float, size, rng: np.random.Generator) -> np.ndarray:
    r"""Sibuya(:math:`\alpha`) variates (as floats), Laplace transform :math:`1-(1-e^{-s})^\alpha`.

    :math:`N` is geometric on :math:`\{1,2,\dots\}` with a
    :math:`\mathrm{Beta}(\alpha, 1-\alpha)` success probability, since
    :math:`E[P(1-P)^{k-1}] = \alpha\Gamma(k-\alpha)/(\Gamma(1-\alpha)k!)`.
    """
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    if alpha == 1.0:
        return np.ones(size)
    p = rng.beta(alpha, 1.0 - alpha, size=size)
    u = rng.random(size)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.floor(np.log(u) / np.log1p(-p)) + 1.0
    return np.where(p >= 1.0, 1.0, np.maximum(n, 1.0))


def log_tilted_stable_rv(alpha: float, tilt: float, size, rng: np.random.Generator):
    r""":math:`\log V`, Laplace transform :math:`\exp(-((a+s)^\alpha - a^\alpha))`.

    Exponentially tilted positive stable law (tilt :math:`a\ge 0`).  :math:`V`
    is the sum of :math:`m=\lceil a^\alpha\rceil` independent copies with
    tilt :math:`a m^{-1/\alpha}`, scaled by :math:`m^{-1/\alpha}`; each copy
    is drawn by rejection from the untilted law (acceptance probability
    :math:`\exp(-a^\alpha/m)\ge e^{-1}`).
    """
    alpha, tilt = float(alpha), float(tilt)
    if alpha == 1.0:
        return np.zeros(size)
    m = max(1, int(np.ceil(tilt**alpha)))
    a1 = tilt * m ** (-1.0 / alpha)
    n = int(np.prod(size))
    total = np.zeros(n)
    for _ in range(m):
        out = np.empty(n)
        todo = np.arange(n)
        while todo.size:
            ls = log_positive_stable_rv(alpha, todo.size, rng)
            acc = np.log(rng.random(todo.size)) <= -a1 * np.exp(ls)
            out[todo[acc]] = np.exp(ls[acc])
            todo = todo[~acc]
        total += out
    return (np.log(total) - np.log(m) / alpha).reshape(size)
