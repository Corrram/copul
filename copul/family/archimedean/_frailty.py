r"""
Frailty distributions for the Marshall--Olkin sampling algorithm.

An Archimedean copula whose inverse generator :math:`\psi` is the Laplace
transform of a positive random variable :math:`V` can be sampled exactly by

.. math::

   U_i = \psi(E_i / V), \qquad E_1, E_2 \overset{iid}{\sim} \mathrm{Exp}(1),

see Marshall & Olkin (1988) and Hofert (2008).  This module provides
vectorized samplers of the standard frailty distributions:

========================= ======================================= ==========
family                    :math:`\psi(t)`                         frailty
========================= ======================================= ==========
Clayton, :math:`\theta>0` :math:`(1+t)^{-1/\theta}`               Gamma(1/θ)
Gumbel--Hougaard          :math:`\exp(-t^{1/\theta})`             positive stable
Frank, :math:`\theta>0`   :math:`-\log(1-(1-e^{-\theta})e^{-t})/\theta` logarithmic
Joe                       :math:`1-(1-e^{-t})^{1/\theta}`         Sibuya
Ali--Mikhail--Haq         :math:`(1-\theta)/(e^t-\theta)`         geometric
========================= ======================================= ==========

All samplers take a NumPy ``Generator`` (or legacy ``RandomState``) and
never touch global random state.

References
----------
Marshall, A. W. & Olkin, I. (1988). Families of multivariate distributions.
*JASA* 83, 834--841.
Kanter, M. (1975). Stable densities under change of scale and total
variation inequalities. *Ann. Probab.* 3, 697--707.
Hofert, M. (2008). Sampling Archimedean copulas. *CSDA* 52, 5163--5174.
Hofert, M. (2011). Efficiently sampling nested Archimedean copulas. *CSDA*
55, 57--70.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = [
    "gamma_frailty",
    "geometric_frailty",
    "logarithmic_frailty",
    "marshall_olkin",
    "positive_stable",
    "sibuya",
]


def marshall_olkin(psi, frailty, rng) -> np.ndarray:
    """Samples ``(psi(E_1/V), psi(E_2/V))`` for frailties ``V`` (shape ``(n,)``)."""
    frailty = np.asarray(frailty, dtype=float)
    e = rng.standard_exponential((frailty.size, 2))
    with np.errstate(all="ignore"):
        out = psi(e / frailty[:, None])
    return np.clip(out, 0.0, 1.0)


def gamma_frailty(n, rng, shape: float) -> np.ndarray:
    """``Gamma(shape, 1)`` variates (Laplace transform :math:`(1+t)^{-shape}`)."""
    return rng.gamma(shape, 1.0, int(n))


def positive_stable(n, rng, alpha: float) -> np.ndarray:
    r"""Positive :math:`\alpha`-stable variates with Laplace transform :math:`e^{-t^\alpha}`.

    Kanter's (1975) representation: with :math:`\Theta\sim U(0,\pi)` and
    :math:`W\sim\mathrm{Exp}(1)`,
    :math:`S=\frac{\sin(\alpha\Theta)}{\sin(\Theta)^{1/\alpha}}
    \bigl(\frac{\sin((1-\alpha)\Theta)}{W}\bigr)^{(1-\alpha)/\alpha}`.
    """
    n = int(n)
    if alpha >= 1.0:
        return np.ones(n)
    theta = np.pi * rng.random(n)
    w = rng.standard_exponential(n)
    sin_t = np.maximum(np.sin(theta), np.finfo(float).tiny)
    with np.errstate(all="ignore"):
        return (
            np.sin(alpha * theta)
            / sin_t ** (1.0 / alpha)
            * (np.sin((1.0 - alpha) * theta) / w) ** ((1.0 - alpha) / alpha)
        )


def logarithmic_frailty(n, rng, p: float) -> np.ndarray:
    r"""Logarithmic (log-series) variates, :math:`P(V=k)=-p^k/(k\log(1-p))`."""
    return rng.logseries(p, int(n)).astype(float)


def geometric_frailty(n, rng, p: float) -> np.ndarray:
    r"""Geometric variates on :math:`\{1,2,\dots\}` with success probability ``p``."""
    return rng.geometric(p, int(n)).astype(float)


def sibuya(n, rng, alpha: float) -> np.ndarray:
    r"""Sibuya(:math:`\alpha`) variates, Laplace transform :math:`1-(1-e^{-t})^\alpha`.

    Exact inversion of the distribution function: the survival function is
    :math:`\bar F(k)=\Gamma(k+1-\alpha)/(\Gamma(k+1)\Gamma(1-\alpha))` and by
    Gautschi's inequality the ``t``-quantile lies within two integers below
    :math:`K=((1-t)\Gamma(1-\alpha))^{-1/\alpha}`, so at most three
    candidates have to be checked.  For :math:`K>10^{12}` the (heavy-tailed)
    variate is returned as :math:`K` itself (relative error below
    :math:`10^{-12}`).
    """
    n = int(n)
    if alpha >= 1.0:
        return np.ones(n)
    t = 1.0 - rng.random(n)  # in (0, 1]
    lg = gammaln(1.0 - alpha)
    with np.errstate(over="ignore"):
        big_k = np.exp(-(np.log(t) + lg) / alpha)

    def log_surv(k):
        return gammaln(k + 1.0 - alpha) - gammaln(k + 1.0) - lg

    log_t = np.log(t)
    c0 = np.maximum(np.ceil(np.minimum(big_k, 1e12)) - 2.0, 1.0)
    out = np.where(
        log_surv(c0) <= log_t,
        c0,
        np.where(log_surv(c0 + 1.0) <= log_t, c0 + 1.0, c0 + 2.0),
    )
    return np.where(big_k > 1e12, big_k, out)
