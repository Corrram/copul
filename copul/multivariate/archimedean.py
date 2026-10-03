r"""
:math:`d`-dimensional Archimedean copulas.

An Archimedean copula is

.. math::

   C(u_1,\dots,u_d) = \psi\bigl(\varphi(u_1)+\dots+\varphi(u_d)\bigr),

with a continuous, non-increasing *generator* :math:`\psi:[0,\infty)\to[0,1]`,
:math:`\psi(0)=1`, :math:`\psi(\infty)=0`, and :math:`\varphi=\psi^{-1}`
(on :math:`(0,1]`).  It is a copula iff :math:`\psi` is *d-monotone*:
differentiable up to order :math:`d-2` with :math:`(-1)^k\psi^{(k)}\ge0` for
:math:`k\le d-2` and :math:`(-1)^{d-2}\psi^{(d-2)}` non-increasing and convex
(McNeil & Nešlehová, 2009, §2).  Completely monotone generators --
Laplace transforms of positive random variables :math:`V` (Kimberling,
1974) -- are valid in every dimension and are sampled exactly by the
Marshall & Olkin (1988) algorithm :math:`U_i=\psi(E_i/V)`.

Densities, conditional distributions and the Kendall distribution function
all need the derivatives :math:`\psi^{(k)}`:

.. math::

   c(u) = \psi^{(d)}\Bigl(\sum_i\varphi(u_i)\Bigr)\prod_i\varphi'(u_i),\qquad
   K_C(t) = P\bigl(C(U)\le t\bigr)
   = \sum_{k=0}^{d-1}\frac{(-\varphi(t))^k}{k!}\,\psi^{(k)}\bigl(\varphi(t)\bigr)

(Barbe, Genest, Ghoudi & Rémillard, 1996; McNeil & Nešlehová, 2009, §4).  They are implemented on the logarithmic scale, with the closed
forms of Hofert, Mächler & McNeil (2012) for the standard families:

=============  =====================================  ============================  ==========================
family         :math:`\psi(t)`                        parameter / valid dims        frailty :math:`V`
=============  =====================================  ============================  ==========================
Clayton        :math:`(1+\theta t)_+^{-1/\theta}`      :math:`\theta\ge-1/(d-1)`     Gamma (:math:`\theta>0`)
Gumbel         :math:`\exp(-t^{1/\theta})`             :math:`\theta\ge1`            positive stable
Frank          :math:`-\log(1-(1-e^{-\theta})e^{-t})/\theta`  :math:`\theta>0` (:math:`\theta\ne0` if d=2)  logarithmic
Joe            :math:`1-(1-e^{-t})^{1/\theta}`         :math:`\theta\ge1`            Sibuya
AMH            :math:`(1-\theta)/(e^t-\theta)`         :math:`0\le\theta<1` (:math:`[-1,1)` if d=2)  geometric
=============  =====================================  ============================  ==========================

* Clayton: :math:`\psi^{(k)}(t) = (-1)^k\prod_{j=0}^{k-1}(1+j\theta)\,(1+\theta t)^{-1/\theta-k}`;
  for :math:`-1/(d-1)\le\theta<0` the generator is d-monotone but not
  completely monotone (McNeil & Nešlehová, 2009, §4);
* Gumbel: :math:`(-1)^k\psi^{(k)}(t) = \psi(t)\,t^{-k}\sum_{j=1}^k a_{kj}(\alpha)t^{\alpha j}`,
  :math:`a_{kj}(\alpha) = (-1)^{k-j}\sum_{l=j}^k\alpha^l s(k,l)S(l,j)`,
  :math:`\alpha=1/\theta` (Stirling numbers of the first and second kind);
* Frank: :math:`(-1)^k\psi^{(k)}(t) = \theta^{-1}\mathrm{Li}_{1-k}\bigl((1-e^{-\theta})e^{-t}\bigr)`;
* Joe: :math:`(-1)^k\psi^{(k)}(t)=\alpha\frac{e^{-t}}{(1-e^{-t})^{1-\alpha}}\sum_{j=0}^{k-1}
  S(k,j+1)\frac{\Gamma(j+1-\alpha)}{\Gamma(1-\alpha)}\bigl(\tfrac{e^{-t}}{1-e^{-t}}\bigr)^j`;
* AMH: :math:`(-1)^k\psi^{(k)}(t) = \frac{1-\theta}\theta\mathrm{Li}_{-k}(\theta e^{-t})`,

with :math:`\mathrm{Li}_{-n}(z)=\sum_{j=0}^n j!\,S(n+1,j+1)\bigl(\tfrac z{1-z}\bigr)^{j+1}`.
Other generators can be given as SymPy expressions (derivatives by
``sympy.diff``) or taken from copul's bivariate Archimedean families
(``cp.Nelsen2(theta=3)``, ...); they are sampled with the stochastic
representation :math:`U=\psi(R\,S)` of McNeil & Nešlehová (2009, §3),
:math:`S` uniform on the unit simplex and the radial part :math:`R` drawn by
numerical inversion of the inverse Williamson transform

.. math::

   F_R(x) = 1-\sum_{k=0}^{d-1}\frac{(-x)^k\,\psi^{(k)}(x)}{k!}.

References
----------
* Barbe, P., Genest, C., Ghoudi, K. and Rémillard, B. (1996). On Kendall's
  process. *J. Multivariate Anal.* 58, 197--229.
* Genest, C. (1987). Frank's family of bivariate distributions.
  *Biometrika* 74, 549--555.
* Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric
  estimation procedure of dependence parameters in multivariate families of
  distributions. *Biometrika* 82, 543--552.
* Genest, C. and MacKay, J. (1986). The joy of copulas: bivariate
  distributions with uniform marginals. *Amer. Statist.* 40, 280--283.
* Genest, C. and Rivest, L.-P. (1993). Statistical inference procedures for
  bivariate Archimedean copulas. *JASA* 88, 1034--1043.
* Hofert, M., Mächler, M. and McNeil, A. J. (2012). Likelihood inference for
  Archimedean copulas in high dimensions under known margins.
  *J. Multivariate Anal.* 110, 133--150.
* Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press, §4.2--4.7.
* Kojadinovic, I. and Yan, J. (2010). Comparison of three semiparametric
  methods for estimating dependence parameters in copula models.
  *Insurance Math. Econom.* 47, 52--63.
* Kimberling, C. H. (1974). A probabilistic interpretation of complete
  monotonicity. *Aequationes Math.* 10, 152--164.
* Marshall, A. W. and Olkin, I. (1988). Families of multivariate
  distributions. *JASA* 83, 834--841.
* McNeil, A. J. and Nešlehová, J. (2009). Multivariate Archimedean copulas,
  d-monotone functions and l1-norm symmetric distributions. *Ann. Statist.*
  37, 3059--3097.
* Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., §4.6.
* Rosenblatt, M. (1952). Remarks on a multivariate transformation.
  *Ann. Math. Statist.* 23, 470--472.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import sympy as sp
from scipy import integrate, optimize
from scipy.special import logsumexp

from copul.family.archimedean import _frailty
from copul.family.constructions._base import NumericBivCopula
from copul.multivariate.base import CopulaND

__all__ = [
    "AliMikhailHaqND",
    "ArchimedeanCopulaND",
    "ClaytonND",
    "FrankND",
    "GumbelND",
    "JoeND",
    "is_d_monotone",
]


# ---------------------------------------------------------------------------
# combinatorics
# ---------------------------------------------------------------------------


@functools.cache
def _stirling2(n: int) -> tuple[tuple[int, ...], ...]:
    """Rows ``S(m, k)`` (second kind) for ``m <= n``."""
    S = [[0] * (n + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for m in range(1, n + 1):
        for k in range(1, m + 1):
            S[m][k] = k * S[m - 1][k] + S[m - 1][k - 1]
    return tuple(tuple(r) for r in S)


@functools.cache
def _stirling1(n: int) -> tuple[tuple[int, ...], ...]:
    """Rows ``s(m, k)`` (signed, first kind) for ``m <= n``."""
    s = [[0] * (n + 1) for _ in range(n + 1)]
    s[0][0] = 1
    for m in range(1, n + 1):
        for k in range(1, m + 1):
            s[m][k] = s[m - 1][k - 1] - (m - 1) * s[m - 1][k]
    return tuple(tuple(r) for r in s)


@functools.cache
def _log_polylog_coeffs(n: int) -> np.ndarray:
    r"""``log(j! S(n+1, j+1))``, ``j = 0..n`` (coefficients of :math:`\mathrm{Li}_{-n}`)."""
    S = _stirling2(n + 1)
    return np.array([math.lgamma(j + 1) + math.log(S[n + 1][j + 1]) for j in range(n + 1)])


def _log_abs_polylog_neg(n: int, x: np.ndarray) -> np.ndarray:
    r""":math:`\log|\mathrm{Li}_{-n}(z)|` in terms of :math:`x = z/(1-z)`."""
    c = _log_polylog_coeffs(n)
    x = np.asarray(x, dtype=float)
    pos = x > 0
    out = np.full(x.shape, -np.inf)
    if np.any(pos):
        lx = np.log(x[pos])
        out[pos] = logsumexp(c[None, :] + np.outer(lx, np.arange(1, n + 2)), axis=1)
    neg = x < 0
    if np.any(neg):
        xn = x[neg]
        val = sum(np.exp(c[j]) * xn ** (j + 1) for j in range(n + 1))
        with np.errstate(divide="ignore"):
            out[neg] = np.log(np.abs(val))
    return out


@functools.lru_cache(maxsize=256)
def _gumbel_log_coeffs(k: int, alpha: float) -> np.ndarray:
    r"""``log a_{kj}(alpha)``, ``j = 1..k`` (Hofert, Mächler & McNeil, 2012, §2).

    Evaluated with 60-digit arithmetic (the alternating sum cancels).
    """
    import mpmath

    s1 = _stirling1(k)
    s2 = _stirling2(k)
    with mpmath.workdps(60 + 2 * k):
        a = mpmath.mpf(alpha)
        out = []
        for j in range(1, k + 1):
            tot = mpmath.mpf(0)
            for m in range(j, k + 1):
                tot += a**m * s1[k][m] * s2[m][j]
            tot *= (-1) ** (k - j)
            out.append(float(mpmath.log(tot)) if tot > 0 else -np.inf)
    return np.array(out)


@functools.lru_cache(maxsize=256)
def _joe_log_coeffs(k: int, alpha: float) -> np.ndarray:
    r"""``log(S(k, j+1) Gamma(j+1-alpha)/Gamma(1-alpha))``, ``j = 0..k-1``."""
    S = _stirling2(k)
    out = []
    for j in range(k):
        prod = [i - alpha for i in range(1, j + 1)]
        if any(p <= 0 for p in prod):
            out.append(-np.inf)
        else:
            out.append(math.log(S[k][j + 1]) + float(np.sum(np.log(prod))))
    return np.array(out)


def _log_expm1(x):
    r""":math:`\log(e^x-1)` for :math:`x>0` (stable for small and large ``x``)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        big = x > 30.0
        return np.where(big, x + np.log1p(-np.exp(-np.where(big, x, 30.0))), np.log(np.expm1(x)))


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


class _Generator:
    """Archimedean generator ``psi`` with its derivatives on the log scale."""

    name = "generic"
    completely_monotone: bool | None = None

    def __init__(self, theta: float | None = None) -> None:
        self.theta = theta

    # to implement --------------------------------------------------------------
    def psi(self, s):  # pragma: no cover - abstract
        raise NotImplementedError

    def phi(self, u):  # pragma: no cover - abstract
        raise NotImplementedError

    def log_abs_dpsi(self, k: int, s):  # pragma: no cover - abstract
        raise NotImplementedError

    # defaults -----------------------------------------------------------------
    def log_mdphi(self, u):
        r""":math:`\log(-\varphi'(u)) = -\log|\psi'(\varphi(u))|`."""
        return -self.log_abs_dpsi(1, self.phi(u))

    def frailty(self, n: int, rng: np.random.Generator):
        """Frailties ``V`` with Laplace transform ``psi`` (or ``None``)."""
        return None

    def max_dim(self) -> float:
        """Largest dimension for which ``psi`` is d-monotone (``inf`` if c.m.)."""
        return np.inf

    def singular(self, d: int) -> bool:
        return False

    def biv_copula(self):
        """The corresponding copul bivariate family (or ``None``)."""
        return None

    def params(self) -> dict:
        return {} if self.theta is None else {"theta": self.theta}

    def __repr__(self) -> str:
        return f"{self.name}(theta={self.theta:g})" if self.theta is not None else self.name


class _IndependenceGenerator(_Generator):
    name = "independence"
    completely_monotone = True

    def psi(self, s):
        return np.exp(-np.asarray(s, float))

    def phi(self, u):
        with np.errstate(divide="ignore"):
            return -np.log(np.asarray(u, float))

    def log_abs_dpsi(self, k, s):
        return -np.asarray(s, float)

    def log_mdphi(self, u):
        with np.errstate(divide="ignore"):
            return -np.log(np.asarray(u, float))

    def frailty(self, n, rng):
        return np.ones(int(n))


class _Clayton(_Generator):
    name = "clayton"

    def __init__(self, theta):
        theta = float(theta)
        if theta < -1.0:
            raise ValueError("Clayton: theta must be >= -1.")
        super().__init__(theta)
        self.completely_monotone = theta >= 0
        self._ind = _IndependenceGenerator() if theta == 0 else None

    def psi(self, s):
        if self._ind:
            return self._ind.psi(s)
        th = self.theta
        base = 1.0 + th * np.asarray(s, float)
        with np.errstate(all="ignore"):
            return np.where(base > 0, np.exp(-np.log(np.where(base > 0, base, 1.0)) / th), 0.0)

    def phi(self, u):
        if self._ind:
            return self._ind.phi(u)
        th = self.theta
        with np.errstate(all="ignore"):
            return np.expm1(-th * np.log(np.asarray(u, float))) / th

    def log_mdphi(self, u):
        if self._ind:
            return self._ind.log_mdphi(u)
        with np.errstate(divide="ignore"):
            return -(self.theta + 1.0) * np.log(np.asarray(u, float))

    def log_abs_dpsi(self, k, s):
        if self._ind:
            return self._ind.log_abs_dpsi(k, s)
        th = self.theta
        s = np.asarray(s, float)
        fac = [1.0 + j * th for j in range(k)]
        if any(f < 0 for f in fac):
            return np.full(s.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            lf = float(np.sum(np.log(fac))) if fac else 0.0
            lb = np.log1p(th * s)
            out = lf - (1.0 / th + k) * lb
        return np.where(1.0 + th * s > 0, out, -np.inf)

    def frailty(self, n, rng):
        if self._ind:
            return self._ind.frailty(n, rng)
        if self.theta < 0:
            return None
        return self.theta * _frailty.gamma_frailty(n, rng, 1.0 / self.theta)

    def max_dim(self):
        return np.inf if self.theta >= 0 else math.floor(1.0 - 1.0 / self.theta + 1e-12)

    def singular(self, d):
        return self.theta < 0 and abs(1.0 + (d - 1) * self.theta) < 1e-12

    def tau(self):
        return self.theta / (self.theta + 2.0)

    @staticmethod
    def tau_inverse(tau):
        return 2.0 * tau / (1.0 - tau)

    def biv_copula(self):
        from copul.family.archimedean import Clayton

        return Clayton(self.theta)


class _Gumbel(_Generator):
    name = "gumbel"
    completely_monotone = True

    def __init__(self, theta):
        theta = float(theta)
        if theta < 1.0:
            raise ValueError("Gumbel: theta must be >= 1.")
        super().__init__(theta)
        self.alpha = 1.0 / theta

    def psi(self, s):
        return np.exp(-(np.asarray(s, float) ** self.alpha))

    def phi(self, u):
        with np.errstate(divide="ignore"):
            return (-np.log(np.asarray(u, float))) ** self.theta

    def log_mdphi(self, u):
        u = np.asarray(u, float)
        th = self.theta
        with np.errstate(divide="ignore", invalid="ignore"):
            mlu = -np.log(u)
            return np.log(th) + (th - 1.0) * np.log(mlu) - np.log(u)

    def log_abs_dpsi(self, k, s):
        s = np.asarray(s, float)
        a = self.alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            if k == 0:
                return -(s**a)
            ls = np.log(s)
            lc = _gumbel_log_coeffs(int(k), float(a))
            j = np.arange(1, k + 1)
            terms = lc[None, :] + np.outer(ls.ravel(), a * j)
            lsum = logsumexp(terms, axis=1).reshape(s.shape)
            return -(s**a) - k * ls + lsum

    def frailty(self, n, rng):
        return _frailty.positive_stable(n, rng, self.alpha)

    def tau(self):
        return 1.0 - 1.0 / self.theta

    @staticmethod
    def tau_inverse(tau):
        return 1.0 / (1.0 - tau)

    def biv_copula(self):
        from copul.family.archimedean import GumbelHougaard

        return GumbelHougaard(self.theta)


class _Frank(_Generator):
    name = "frank"

    def __init__(self, theta):
        theta = float(theta)
        super().__init__(theta)
        self.completely_monotone = theta >= 0
        self._ind = _IndependenceGenerator() if theta == 0 else None
        self._c = -np.expm1(-theta)  # 1 - exp(-theta)

    def psi(self, s):
        if self._ind:
            return self._ind.psi(s)
        z = self._c * np.exp(-np.asarray(s, float))
        return -np.log1p(-z) / self.theta

    def phi(self, u):
        if self._ind:
            return self._ind.phi(u)
        th = self.theta
        u = np.asarray(u, float)
        with np.errstate(all="ignore"):
            return -np.log(np.expm1(-th * u) / np.expm1(-th))

    def log_mdphi(self, u):
        if self._ind:
            return self._ind.log_mdphi(u)
        th = self.theta
        u = np.asarray(u, float)
        with np.errstate(all="ignore"):
            if th > 0:
                return np.log(th) - _log_expm1(th * u)
            return np.log(-th) - np.log(-np.expm1(th * u))

    def log_abs_dpsi(self, k, s):
        if self._ind:
            return self._ind.log_abs_dpsi(k, s)
        s = np.asarray(s, float)
        with np.errstate(all="ignore"):
            if k == 0:
                return np.log(self.psi(s))
            z = self._c * np.exp(-s)
            x = z / (1.0 - z)
            return _log_abs_polylog_neg(k - 1, x) - np.log(abs(self.theta))

    def frailty(self, n, rng):
        if self._ind:
            return self._ind.frailty(n, rng)
        if self.theta <= 0:
            return None
        return _frailty.logarithmic_frailty(n, rng, self._c)

    def max_dim(self):
        return np.inf if self.theta >= 0 else 2

    def tau(self):
        r""":math:`1-\frac4\theta(1-D_1(\theta))`, :math:`D_1` the Debye function
        (Genest, 1987)."""
        th = self.theta
        if th == 0:
            return 0.0
        d1 = integrate.quad(lambda t: t / np.expm1(t) if t != 0 else 1.0, 0.0, th)[0] / th
        return 1.0 - 4.0 / th * (1.0 - d1)

    def biv_copula(self):
        from copul.family.archimedean import Frank

        return Frank(self.theta)


class _Joe(_Generator):
    name = "joe"
    completely_monotone = True

    def __init__(self, theta):
        theta = float(theta)
        if theta < 1.0:
            raise ValueError("Joe: theta must be >= 1.")
        super().__init__(theta)
        self.alpha = 1.0 / theta

    def psi(self, s):
        s = np.asarray(s, float)
        with np.errstate(divide="ignore"):
            return -np.expm1(self.alpha * np.log1p(-np.exp(-s)))

    def phi(self, u):
        u = np.asarray(u, float)
        with np.errstate(divide="ignore"):
            return -np.log(-np.expm1(self.theta * np.log1p(-u)))

    def log_mdphi(self, u):
        u = np.asarray(u, float)
        th = self.theta
        with np.errstate(all="ignore"):
            l1u = np.log1p(-u)
            return np.log(th) + (th - 1.0) * l1u - np.log(-np.expm1(th * l1u))

    def log_abs_dpsi(self, k, s):
        s = np.asarray(s, float)
        a = self.alpha
        with np.errstate(all="ignore"):
            if k == 0:
                return np.log(self.psi(s))
            lx = -_log_expm1(s)  # log(e^{-s}/(1-e^{-s}))
            lc = _joe_log_coeffs(int(k), float(a))
            terms = lc[None, :] + np.outer(lx.ravel(), np.arange(k))
            lsum = logsumexp(terms, axis=1).reshape(s.shape)
            return np.log(a) - s + (a - 1.0) * np.log1p(-np.exp(-s)) + lsum

    def frailty(self, n, rng):
        return _frailty.sibuya(n, rng, self.alpha)

    def biv_copula(self):
        from copul.family.archimedean import Joe

        return Joe(self.theta)


class _AMH(_Generator):
    name = "amh"

    def __init__(self, theta):
        theta = float(theta)
        if not -1.0 <= theta < 1.0:
            raise ValueError("Ali-Mikhail-Haq: theta must lie in [-1, 1).")
        super().__init__(theta)
        self.completely_monotone = theta >= 0

    def psi(self, s):
        s = np.asarray(s, float)
        th = self.theta
        with np.errstate(over="ignore"):
            return (1.0 - th) * np.exp(-s) / (1.0 - th * np.exp(-s))

    def phi(self, u):
        u = np.asarray(u, float)
        with np.errstate(divide="ignore"):
            return np.log1p(-self.theta * (1.0 - u)) - np.log(u)

    def log_mdphi(self, u):
        u = np.asarray(u, float)
        th = self.theta
        with np.errstate(divide="ignore"):
            return np.log1p(-th) - np.log(u) - np.log1p(-th * (1.0 - u))

    def log_abs_dpsi(self, k, s):
        s = np.asarray(s, float)
        th = self.theta
        k = int(k)
        S = _stirling2(k + 1)
        e = np.exp(-s)
        w = th * e / (1.0 - th * e)  # theta e^{-s} / (1 - theta e^{-s})
        with np.errstate(all="ignore"):
            base = np.log1p(-th) - s - np.log1p(-th * e)
            if th >= 0:
                lw = np.log(w) if th > 0 else np.full(s.shape, -np.inf)
                terms = np.stack(
                    [math.lgamma(j + 1) + math.log(S[k + 1][j + 1]) + j * lw for j in range(k + 1)]
                )
                terms[0] = math.log(S[k + 1][1])  # j = 0 term (avoid 0 * -inf)
                return base + logsumexp(terms, axis=0)
            val = sum(math.factorial(j) * S[k + 1][j + 1] * w**j for j in range(k + 1))
            return base + np.log(np.abs(val))

    def frailty(self, n, rng):
        if self.theta < 0:
            return None
        return _frailty.geometric_frailty(n, rng, 1.0 - self.theta)

    def max_dim(self):
        return np.inf if self.theta >= 0 else 2

    def tau(self):
        r""":math:`\frac{3\theta-2}{3\theta}-\frac{2(1-\theta)^2\log(1-\theta)}{3\theta^2}`
        (Nelsen, 2006, ch. 5)."""
        th = self.theta
        if abs(th) < 1e-6:
            return 2.0 * th / 9.0
        return (3 * th - 2) / (3 * th) - 2 * (1 - th) ** 2 * np.log1p(-th) / (3 * th**2)

    def biv_copula(self):
        from copul.family.archimedean import AliMikhailHaq

        return AliMikhailHaq(self.theta)


class _SymbolicGenerator(_Generator):
    """Generator given by a SymPy expression (derivatives by ``sympy.diff``)."""

    name = "symbolic"

    def __init__(self, psi_expr, symbol=None, phi_expr=None, phi_symbol=None, name=None):
        super().__init__(None)
        expr = sp.sympify(psi_expr)
        if symbol is None:
            free = sorted(expr.free_symbols, key=str)
            if len(free) != 1:
                raise ValueError(
                    "the generator expression must have exactly one free symbol "
                    f"(substitute parameter values first), got {free}."
                )
            symbol = free[0]
        self.expr = expr
        self.symbol = symbol
        self._derivs: dict[int, Any] = {}
        self._phi_fn = None
        if phi_expr is not None:
            pexpr = sp.sympify(phi_expr)
            if phi_symbol is None:
                free = sorted(pexpr.free_symbols, key=str)
                if len(free) != 1:
                    raise ValueError("phi must have exactly one free symbol.")
                phi_symbol = free[0]
            from copul.numerics import to_numpy_callable

            self._phi_fn = to_numpy_callable(pexpr, [phi_symbol], ae=True)
        if name:
            self.name = name

    def _deriv(self, k: int):
        f = self._derivs.get(k)
        if f is None:
            from copul.numerics import to_numpy_callable

            e = self.expr
            for _ in range(k):
                e = sp.diff(e, self.symbol)
            f = to_numpy_callable(e, [self.symbol], ae=True)
            self._derivs[k] = f
        return f

    def signed_dpsi(self, k: int, s):
        s = np.asarray(s, float)
        with np.errstate(all="ignore"):
            out = np.asarray(self._deriv(k)(s), dtype=float)
        return np.broadcast_to(out, s.shape).astype(float)

    def psi(self, s):
        s = np.asarray(s, float)
        out = self.signed_dpsi(0, np.where(np.isfinite(s), s, 0.0))
        return np.where(np.isinf(s), 0.0, out)

    def log_abs_dpsi(self, k, s):
        s = np.asarray(s, float)
        with np.errstate(divide="ignore"):
            out = np.log(np.abs(self.signed_dpsi(k, np.where(np.isfinite(s), s, 1.0))))
        return np.where(np.isinf(s), -np.inf, out)

    def phi(self, u):
        u = np.asarray(u, float)
        if self._phi_fn is not None:
            with np.errstate(all="ignore"):
                return np.broadcast_to(np.asarray(self._phi_fn(u), float), u.shape).astype(float)
        return self._phi_numeric(u)

    def _phi_numeric(self, u):
        """Invert ``psi`` by bisection on a logarithmic grid."""
        u = np.asarray(u, float)
        shape = u.shape
        u = u.ravel()
        lo = np.zeros_like(u)
        hi = np.ones_like(u)
        for _ in range(200):
            grow = self.psi(hi) > u
            if not np.any(grow):
                break
            hi = np.where(grow, hi * 4.0, hi)
        for _ in range(110):
            mid = 0.5 * (lo + hi)
            above = self.psi(mid) > u
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        out = 0.5 * (lo + hi)
        out = np.where(u >= 1.0, 0.0, out)
        out = np.where(u <= 0.0, np.inf, out)
        return out.reshape(shape)


_FAMILIES = {
    "clayton": _Clayton,
    "gumbel": _Gumbel,
    "gumbel_hougaard": _Gumbel,
    "gumbelhougaard": _Gumbel,
    "frank": _Frank,
    "joe": _Joe,
    "amh": _AMH,
    "ali_mikhail_haq": _AMH,
    "alimikhailhaq": _AMH,
}

_COPUL_CLASSES = {
    "Clayton": "clayton",
    "BivClayton": "clayton",
    "Nelsen1": "clayton",
    "GumbelHougaard": "gumbel",
    "GumbelHougaardEV": "gumbel",
    "Nelsen4": "gumbel",
    "Frank": "frank",
    "Nelsen5": "frank",
    "Joe": "joe",
    "Nelsen6": "joe",
    "AliMikhailHaq": "amh",
    "Nelsen3": "amh",
}


def _resolve_generator(generator, theta=None, phi=None) -> _Generator:
    if isinstance(generator, _Generator):
        return generator
    if isinstance(generator, ArchimedeanCopulaND):
        return generator.generator
    if isinstance(generator, str) and generator.lower().replace("-", "_") in _FAMILIES:
        if theta is None:
            raise ValueError(f"family {generator!r} needs a parameter theta=.")
        return _FAMILIES[generator.lower().replace("-", "_")](theta)
    if isinstance(generator, (str, sp.Basic)):
        return _SymbolicGenerator(generator, phi_expr=phi)
    # a copul bivariate Archimedean family instance
    cname = type(generator).__name__
    if cname == "BivIndependenceCopula":
        return _IndependenceGenerator()
    if hasattr(generator, "theta"):
        th = generator.theta
        if isinstance(th, sp.Basic) and not th.is_number:
            raise ValueError(f"{cname} has a free parameter theta; fix it first.")
        if cname in _COPUL_CLASSES:
            return _FAMILIES[_COPUL_CLASSES[cname]](float(th))
        if hasattr(generator, "_raw_inv_generator"):
            psi_expr = generator._raw_inv_generator
            phi_expr = getattr(generator, "_raw_generator", None)
            return _SymbolicGenerator(
                psi_expr,
                symbol=generator.y,
                phi_expr=phi_expr,
                phi_symbol=getattr(generator, "t", None),
                name=f"{cname}(theta={float(th):g})",
            )
    raise TypeError(f"cannot build an Archimedean generator from {generator!r}.")


# ---------------------------------------------------------------------------
# d-monotonicity
# ---------------------------------------------------------------------------


def is_d_monotone(
    generator: Any,
    d: int,
    theta: float | None = None,
    grid: np.ndarray | None = None,
    tol: float = 1e-10,
) -> bool:
    r"""Whether a generator :math:`\psi` is d-monotone (McNeil & Nešlehová, 2009).

    For the standard families the known parameter ranges are used
    (Clayton: :math:`\theta\ge -1/(d-1)`; Frank and AMH with negative
    parameter: :math:`d=2`; completely monotone generators: all :math:`d`).
    Other (SymPy) generators are checked numerically on a logarithmic grid of
    :math:`t\in(0,\infty)`: :math:`\psi(0)=1`,
    :math:`(-1)^k\psi^{(k)}(t)\ge -\mathrm{tol}` for :math:`k\le d-2`, and
    :math:`f=(-1)^{d-2}\psi^{(d-2)}` non-increasing and convex (first
    differences :math:`\le 0`, slopes non-decreasing), the definition of
    d-monotonicity (McNeil & Nešlehová, 2009, §2).  This detects kinks
    of non-smooth generators (e.g. :math:`(1-t)_+` is 2- but not
    3-monotone).

    Parameters
    ----------
    generator : str, SymPy expression, copul Archimedean family or ArchimedeanCopulaND
    d : int
        Dimension.
    theta : float, optional
        Family parameter (for family names).
    grid : array_like, optional
        Increasing points :math:`t>0` for the numerical check (default: 600
        points from :math:`10^{-6}` to :math:`10^{3}`).
    tol : float
        Relative tolerance of the numerical check.
    """
    gen = _resolve_generator(generator, theta)
    d = int(d)
    if not isinstance(gen, _SymbolicGenerator):
        return d <= gen.max_dim()
    t = np.geomspace(1e-6, 1e3, 600) if grid is None else np.sort(np.asarray(grid, float))
    if abs(float(gen.psi(np.array([0.0]))[0]) - 1.0) > 1e-8:
        return False

    def bad(vals):
        vals = vals[np.isfinite(vals)]
        return vals.size > 0 and vals.min() < -tol * max(1.0, float(np.abs(vals).max()))

    for k in range(max(d - 1, 1)):
        if bad((-1.0) ** k * gen.signed_dpsi(k, t)):
            return False
    f = (-1.0) ** (d - 2) * gen.signed_dpsi(max(d - 2, 0), t)
    ok = np.isfinite(f)
    f, tt = f[ok], t[ok]
    if f.size >= 3:
        dt = np.diff(tt)
        slopes = np.diff(f) / dt
        # rounding error of the difference quotients
        noise = 8.0 * np.finfo(float).eps * np.maximum(np.abs(f[:-1]), np.abs(f[1:])) / dt
        scale = tol * max(1.0, float(np.max(np.abs(slopes))))
        if np.any(slopes > noise + scale):
            return False
        if np.any(np.diff(slopes) < -(noise[:-1] + noise[1:] + scale)):
            return False
    return True


# ---------------------------------------------------------------------------
# copula classes
# ---------------------------------------------------------------------------


class ArchimedeanCopulaND(CopulaND):
    r""":math:`d`-dimensional Archimedean copula :math:`\psi(\sum_i\varphi(u_i))`.

    Parameters
    ----------
    generator : str, SymPy expression or copul bivariate Archimedean family
        * a family name -- ``"clayton"``, ``"gumbel"``, ``"frank"``,
          ``"joe"``, ``"amh"`` -- together with ``theta``;
        * a SymPy expression (or string) of :math:`\psi` in one variable,
          e.g. ``"(1 + t)**(-1/2)"`` (all parameters substituted);
        * a fully specified copul bivariate Archimedean family such as
          ``cp.Clayton(theta=2)`` or ``cp.Nelsen12(theta=2)`` (its
          generator inverse is used).
    dim : int
        Dimension :math:`d\ge 2`.
    theta : float, optional
        Family parameter.
    phi : SymPy expression, optional
        Closed form of :math:`\varphi=\psi^{-1}` for SymPy generators
        (otherwise :math:`\psi` is inverted numerically).
    check : bool
        Raise a ``ValueError`` if :math:`\psi` is not d-monotone
        (see :func:`is_d_monotone`).

    Examples
    --------
    >>> from copul.multivariate import ArchimedeanCopulaND
    >>> C = ArchimedeanCopulaND("clayton", dim=3, theta=2.0)
    >>> round(C.cdf([0.5, 0.5, 0.5]), 6)       # (3 * 2**2 - 2)**(-1/2)
    0.316228
    >>> type(C.margin(0, 1)).__name__           # bivariate copul Clayton
    'BivClayton'
    >>> round(C.kendalls_tau(), 10)             # tau_3 = theta / (theta + 2)
    0.5
    """

    exchangeable = True
    radially_symmetric = False

    def __init__(
        self,
        generator: Any,
        dim: int = 3,
        theta: float | None = None,
        *,
        phi: Any = None,
        check: bool = True,
    ) -> None:
        super().__init__(dim)
        self.generator = _resolve_generator(generator, theta, phi)
        if check and not is_d_monotone(self.generator, self.dim):
            raise ValueError(
                f"the generator {self.generator!r} is not {self.dim}-monotone, so it does not "
                f"generate a {self.dim}-dimensional copula (McNeil & Nešlehová, 2009)."
            )

    # -- generator access ---------------------------------------------------------
    @property
    def family(self) -> str:
        """Family name (``"clayton"``, ..., or ``"symbolic"``)."""
        return self.generator.name

    @property
    def theta(self) -> float | None:
        """Family parameter."""
        return self.generator.theta

    def psi(self, s: Any) -> np.ndarray:
        r"""Generator :math:`\psi(s)`."""
        return self.generator.psi(s)

    def phi(self, u: Any) -> np.ndarray:
        r"""Inverse generator :math:`\varphi(u)=\psi^{-1}(u)`."""
        return self.generator.phi(u)

    def psi_derivative(self, k: int, s: Any) -> np.ndarray:
        r""":math:`\psi^{(k)}(s)` (with sign :math:`(-1)^k`)."""
        return (-1.0) ** k * np.exp(self.generator.log_abs_dpsi(int(k), np.asarray(s, float)))

    def __repr__(self) -> str:
        return f"ArchimedeanCopulaND({self.generator!r}, dim={self.dim})"

    @property
    def is_absolutely_continuous(self) -> bool:
        return not self.generator.singular(self.dim)

    # -- evaluation -------------------------------------------------------------------
    def _phi_sum(self, U):
        with np.errstate(all="ignore"):
            return np.sum(self.generator.phi(U), axis=1)

    def _cdf(self, U):
        return self.generator.psi(self._phi_sum(U))

    def _logpdf(self, U):
        g = self.generator
        with np.errstate(all="ignore"):
            s = self._phi_sum(U)
            return g.log_abs_dpsi(self.dim, s) + np.sum(g.log_mdphi(U), axis=1)

    def _rvs(self, n, rng):
        g = self.generator
        V = g.frailty(n, rng) if g.completely_monotone else None
        E = rng.standard_exponential((n, self.dim))
        if V is not None:
            with np.errstate(all="ignore"):
                return g.psi(E / np.asarray(V, float)[:, None])
        R = self._sample_radial(n, rng)
        S = E / E.sum(axis=1, keepdims=True)
        with np.errstate(all="ignore"):
            return g.psi(R[:, None] * S)

    # -- stochastic representation -------------------------------------------------------
    def radial_cdf(self, x: Any) -> np.ndarray:
        r"""Distribution function :math:`F_R` of the radial part.

        :math:`F_R(x) = 1-\sum_{k=0}^{d-1}(-x)^k\psi^{(k)}(x)/k!`, the inverse
        Williamson :math:`d`-transform of :math:`\psi` (McNeil & Nešlehová,
        2009, §3); :math:`U=\psi(RS)` with :math:`S` uniform on the unit
        simplex has copula :math:`C`.
        """
        x = np.asarray(x, float)
        with np.errstate(all="ignore"):
            lx = np.log(x)
            terms = np.stack(
                [
                    k * lx + self.generator.log_abs_dpsi(k, x) - math.lgamma(k + 1)
                    if k > 0
                    else self.generator.log_abs_dpsi(0, x)
                    for k in range(self.dim)
                ]
            )
            terms = np.where(np.isnan(terms), -np.inf, terms)
            out = 1.0 - np.exp(logsumexp(terms, axis=0))
        out = np.where(x <= 0, 0.0, out)
        return np.clip(out, 0.0, 1.0)

    def _sample_radial(self, n, rng):
        p = rng.random(n)
        lo = np.full(n, -40.0)
        hi = np.full(n, 40.0)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = self.radial_cdf(np.exp(mid)) < p
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return np.exp(0.5 * (lo + hi))

    def kendall_distribution(self, t: Any) -> np.ndarray:
        r"""Kendall distribution function :math:`K_C(t)=P(C(U)\le t)`.

        .. math::

           K_C(t) = \sum_{k=0}^{d-1}\frac{(-\varphi(t))^k}{k!}\,
           \psi^{(k)}\bigl(\varphi(t)\bigr) = 1 - F_R\bigl(\varphi(t)\bigr)

        (Genest & Rivest, 1993, for :math:`d=2`; Barbe et al., 1996;
        McNeil & Nešlehová, 2009, §4).
        """
        t = np.asarray(t, float)
        tc = np.clip(t, 0.0, 1.0)
        with np.errstate(all="ignore"):
            out = 1.0 - self.radial_cdf(self.generator.phi(tc))
        out = np.where(tc >= 1.0, 1.0, out)
        return np.where(t < 0, 0.0, out)

    def rosenblatt(self, U: Any) -> np.ndarray:
        r"""Rosenblatt (1952) transform.

        :math:`C_{k|1..k-1}(u_k\mid u_{<k}) =
        \psi^{(k-1)}(s_k)/\psi^{(k-1)}(s_{k-1})` with
        :math:`s_k=\sum_{i\le k}\varphi(u_i)` (Hofert, Mächler & McNeil, 2012);
        maps a sample of :math:`C` to independent uniforms.
        """
        U = np.atleast_2d(np.clip(np.asarray(U, float), 0.0, 1.0))
        g = self.generator
        with np.errstate(all="ignore"):
            S = np.cumsum(g.phi(U), axis=1)
            out = np.empty_like(U)
            out[:, 0] = U[:, 0]
            for k in range(1, self.dim):
                out[:, k] = np.exp(g.log_abs_dpsi(k, S[:, k]) - g.log_abs_dpsi(k, S[:, k - 1]))
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

    # -- margins and measures -------------------------------------------------------------
    def _margin(self, idx):
        if len(idx) == 2:
            biv = self.generator.biv_copula()
            if biv is not None:
                return biv
            return _ArchimedeanBivariate(self.generator)
        return ArchimedeanCopulaND(self.generator, len(idx), check=False)

    def _tau_from_kendall(self, d: int) -> float:
        C = self if d == self.dim else ArchimedeanCopulaND(self.generator, d, check=False)
        integral, _ = integrate.quad(
            lambda t: float(C.kendall_distribution(np.array([t]))[0]), 0.0, 1.0, limit=200
        )
        mean_c = 1.0 - integral
        return (2.0**d * mean_c - 1.0) / (2.0 ** (d - 1) - 1.0)

    def pairwise_kendalls_tau(self) -> float:
        r"""Kendall's :math:`\tau` of the (identical) bivariate margins.

        Closed forms where known (Clayton :math:`\theta/(\theta+2)`, Gumbel
        :math:`1-1/\theta`), otherwise
        :math:`\tau=1+4\int_0^1\varphi(t)/\varphi'(t)\,dt` (Genest & MacKay,
        1986) by quadrature.
        """
        tau = getattr(self.generator, "tau", None)
        if callable(tau):
            return float(tau())
        return self._tau_from_kendall(2)

    def kendalls_tau_matrix(self) -> np.ndarray:
        t = self.pairwise_kendalls_tau()
        return np.where(np.eye(self.dim, dtype=bool), 1.0, t)

    def _exact_measure(self, key: str):
        if key == "tau":
            if self.dim == 2:
                return self.pairwise_kendalls_tau()
            return self._tau_from_kendall(self.dim)
        return None

    # -- estimation ------------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        data: Any,
        family: str,
        method: str = "mle",
        pseudo_obs: bool = True,
        bounds: tuple[float, float] | None = None,
    ) -> ArchimedeanCopulaND:
        r"""Estimate the parameter of an Archimedean family from data.

        Parameters
        ----------
        data : array_like of shape (n, d)
        family : str
            ``"clayton"``, ``"gumbel"``, ``"frank"``, ``"joe"`` or ``"amh"``.
        method : {"mle", "itau"}
            * ``"mle"``: maximum pseudo-likelihood
              :math:`\arg\max_\theta\sum_i\log c_\theta(\hat U_i)` (Genest,
              Ghoudi & Rivest, 1995; Hofert, Mächler & McNeil, 2012);
            * ``"itau"``: inversion of the average pairwise sample Kendall's
              tau (Genest & Rivest, 1993; Kojadinovic & Yan, 2010).
        pseudo_obs : bool
            Rank-transform the data (default).
        bounds : (float, float), optional
            Parameter search interval.
        """
        from copul.multivariate.elliptical import _pairwise_tau, _prepare_uniform

        U = _prepare_uniform(data, pseudo_obs)
        d = U.shape[1]
        fam = family.lower().replace("-", "_")
        if fam not in _FAMILIES:
            raise ValueError(
                f"unknown family {family!r}; choose from clayton, gumbel, frank, joe, amh."
            )
        gcls = _FAMILIES[fam]
        lo, hi = bounds if bounds is not None else _default_bounds(gcls, d)
        T = _pairwise_tau(U)
        tau_bar = float(T[np.triu_indices(d, 1)].mean())

        def tau_of(th):
            g = gcls(th)
            t = getattr(g, "tau", None)
            if callable(t):
                return float(t())
            return float(g.biv_copula().kendalls_tau())

        if method == "itau":
            inv = getattr(gcls, "tau_inverse", None)
            if callable(inv):
                th = float(np.clip(inv(tau_bar), lo, hi))
            else:
                f_lo, f_hi = tau_of(lo) - tau_bar, tau_of(hi) - tau_bar
                if f_lo >= 0:
                    th = lo
                elif f_hi <= 0:
                    th = hi
                else:
                    th = optimize.brentq(lambda x: tau_of(x) - tau_bar, lo, hi, xtol=1e-10)
            return cls(fam, d, theta=th)
        if method != "mle":
            raise ValueError("method must be 'mle' or 'itau'.")

        def nll(th):
            val = -float(np.sum(cls(fam, d, theta=th, check=False)._logpdf_clean(U)))
            return val if np.isfinite(val) else 1e300

        res = optimize.minimize_scalar(
            nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8}
        )
        return cls(fam, d, theta=float(res.x))


def _default_bounds(gcls, d: int) -> tuple[float, float]:
    if gcls is _Clayton:
        return (max(-1.0 / (d - 1), -1.0) + 1e-6, 50.0)
    if gcls in (_Gumbel, _Joe):
        return (1.0, 50.0)
    if gcls is _Frank:
        return (-60.0, 60.0) if d == 2 else (1e-6, 60.0)
    return (-1.0, 1.0 - 1e-9) if d == 2 else (0.0, 1.0 - 1e-9)


class _ArchimedeanBivariate(NumericBivCopula):
    r"""Bivariate Archimedean copula of a generator without a copul family
    (closed-form cdf, h-functions and density from the generator)."""

    def __init__(self, generator: _Generator) -> None:
        self._g = generator
        super().__init__()

    @property
    def is_absolutely_continuous(self) -> bool:
        return not self._g.singular(2)

    @property
    def is_symmetric(self) -> bool:
        return True

    def _cdf(self, u, v):
        with np.errstate(all="ignore"):
            return self._g.psi(self._g.phi(u) + self._g.phi(v))

    def _h(self, a, b):
        g = self._g
        with np.errstate(all="ignore"):
            s = g.phi(a) + g.phi(b)
            return np.exp(g.log_abs_dpsi(1, s) + g.log_mdphi(a))

    def _h1(self, u, v):
        return self._h(u, v)

    def _h2(self, u, v):
        return self._h(v, u)

    def _pdf(self, u, v):
        g = self._g
        with np.errstate(all="ignore"):
            s = g.phi(u) + g.phi(v)
            return np.exp(g.log_abs_dpsi(2, s) + g.log_mdphi(u) + g.log_mdphi(v))

    def _rvs(self, n, rng):
        return ArchimedeanCopulaND(self._g, 2, check=False)._rvs(n, rng)

    def __repr__(self) -> str:
        return f"ArchimedeanBivariate({self._g!r})"

    __str__ = __repr__


def ClaytonND(theta: float, dim: int = 3) -> ArchimedeanCopulaND:
    r"""Clayton copula :math:`\bigl(\sum_i u_i^{-\theta}-d+1\bigr)_+^{-1/\theta}`,
    :math:`\theta\ge-1/(d-1)` (McNeil & Nešlehová, 2009)."""
    return ArchimedeanCopulaND("clayton", dim, theta=theta)


def GumbelND(theta: float, dim: int = 3) -> ArchimedeanCopulaND:
    r"""Gumbel--Hougaard copula :math:`\exp\bigl(-(\sum_i(-\log u_i)^\theta)^{1/\theta}\bigr)`,
    :math:`\theta\ge1`."""
    return ArchimedeanCopulaND("gumbel", dim, theta=theta)


def FrankND(theta: float, dim: int = 3) -> ArchimedeanCopulaND:
    r"""Frank copula, :math:`\theta>0` (:math:`\theta\ne0` for :math:`d=2`)."""
    return ArchimedeanCopulaND("frank", dim, theta=theta)


def JoeND(theta: float, dim: int = 3) -> ArchimedeanCopulaND:
    r"""Joe copula :math:`1-\bigl(1-\prod_i(1-(1-u_i)^\theta)\bigr)^{1/\theta}`, :math:`\theta\ge1`."""
    return ArchimedeanCopulaND("joe", dim, theta=theta)


def AliMikhailHaqND(theta: float, dim: int = 3) -> ArchimedeanCopulaND:
    r"""Ali--Mikhail--Haq copula, :math:`0\le\theta<1` (:math:`-1\le\theta<1` for :math:`d=2`)."""
    return ArchimedeanCopulaND("amh", dim, theta=theta)
