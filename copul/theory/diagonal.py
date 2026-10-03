r"""
Diagonal sections of bivariate copulas and copulas with a given diagonal.

The **diagonal section** of a copula :math:`C` is :math:`\delta_C(t)=C(t,t)`
and its **opposite diagonal section** is :math:`\omega_C(t) = C(t,1-t)`.
Every diagonal section satisfies (Nelsen 2006, Sect. 3.2.6)

(D1) :math:`\delta(1) = 1` (and :math:`\delta(0)=0`);

(D2) :math:`\delta(t)\le t` for all :math:`t\in[0,1]`;

(D3) :math:`0\le\delta(t_2)-\delta(t_1)\le 2(t_2-t_1)` for
     :math:`0\le t_1\le t_2\le 1`,

which imply :math:`\max(2t-1,0)\le\delta(t)\le t`.  A function with
(D1)--(D3) is called a **diagonal**, and every diagonal is the diagonal
section of a copula -- for instance of the **Bertino copula**
(Bertino 1977; Fredricks & Nelsen 1997)

.. math::

   B_\delta(u,v) = \min(u,v) - \min_{t\in[u\wedge v,\,u\vee v]}\bigl(t-\delta(t)\bigr),

which is the *smallest* copula with diagonal section :math:`\delta`
(:math:`B_\delta\le C` for every copula :math:`C` with
:math:`\delta_C=\delta`), and of the symmetric **diagonal copula** of
Fredricks & Nelsen (1997)

.. math::

   K_\delta(u,v) = \min\Bigl(u, v, \tfrac12\bigl(\delta(u)+\delta(v)\bigr)\Bigr),

which is the *largest symmetric* copula with diagonal section
:math:`\delta` (:math:`C\le K_\delta` for every symmetric copula :math:`C`
with :math:`\delta_C=\delta`; for non-symmetric :math:`C` see
:func:`copul.theory.bounds.bounds_given_diagonal`).

Diagonal sections determine several dependence quantities (Nelsen 2006,
Sects. 5.1 and 5.4):

* Spearman's footrule :math:`\phi = 6\int_0^1\delta(t)\,dt - 2`;
* Gini's gamma :math:`\gamma = 4\int_0^1[\delta(t)+\omega(t)]\,dt - 2`;
* Blomqvist's beta :math:`\beta = 4\delta(\tfrac12) - 1`;
* the tail dependence coefficients
  :math:`\lambda_L = \delta'(0^+)` and :math:`\lambda_U = 2 - \delta'(1^-)`
  (when the one-sided derivatives exist).

References
----------
Bertino, S. (1977). Sulla dissomiglianza tra mutabili cicliche. *Metron*
35, 53–88.

Fredricks, G. A. & Nelsen, R. B. (1997). Copulas constructed from diagonal
sections. In Beneš, V. & Štěpán, J. (eds.), *Distributions with Given
Marginals and Moment Problems*, Kluwer, Dordrecht, 129–136.

Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
Sects. 3.2.6, 5.1, 5.4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from copul.measures.numeric import lambda_l_from_cdf, lambda_u_from_cdf
from copul.measures.quadrature import integrate_1d
from copul.theory.quasi import InversionSamplingCopula, cdf_function

__all__ = [
    "BertinoCopula",
    "Diagonal",
    "DiagonalCheck",
    "DiagonalCopula",
    "as_diagonal",
    "bertino_copula",
    "blomqvist_from_diagonal",
    "check_diagonal",
    "copula_with_diagonal",
    "diagonal_copula",
    "diagonal_section",
    "footrule_from_diagonal",
    "gini_from_diagonals",
    "is_diagonal",
    "opposite_diagonal",
    "tail_dependence_from_diagonal",
]


# ---------------------------------------------------------------------------
# diagonal objects
# ---------------------------------------------------------------------------


def _as_vectorized(f: Callable) -> Callable[[np.ndarray], np.ndarray]:
    def g(t):
        t = np.asarray(t, dtype=float)
        with np.errstate(all="ignore"):
            out = np.asarray(f(t), dtype=float)
        return np.broadcast_to(out, t.shape).astype(float, copy=True)

    return g


class Diagonal:
    r"""A diagonal :math:`\delta:[0,1]\to[0,1]` (vectorized callable).

    Parameters
    ----------
    func : callable
        Vectorized :math:`\delta(t)`.
    derivative : callable, optional
        Vectorized :math:`\delta'(t)` (central finite differences otherwise).
    name : str, optional

    Notes
    -----
    Construction does not validate the axioms (D1)--(D3); use
    :meth:`check` / :func:`is_diagonal`.

    Examples
    --------
    >>> from copul.theory.diagonal import Diagonal
    >>> delta = Diagonal(lambda t: t**2, name="t^2")   # diagonal of Pi
    >>> delta.is_valid()
    True
    >>> round(delta.spearmans_footrule(), 12)
    0.0
    """

    def __init__(self, func: Callable, derivative: Callable | None = None, name: str | None = None):
        if isinstance(func, Diagonal):
            derivative = derivative or func._derivative
            name = name or func.name
            func = func._func
        self._func = _as_vectorized(func)
        self._derivative = None if derivative is None else _as_vectorized(derivative)
        self.name = name or "Diagonal"

    def __repr__(self) -> str:
        return self.name

    __str__ = __repr__

    def __call__(self, t):
        """:math:`\\delta(t)`; a float for scalar input (clipped to [0, 1])."""
        t = np.asarray(t, dtype=float)
        out = self._func(np.clip(t, 0.0, 1.0))
        return float(out) if out.ndim == 0 else out

    def vectorized(self, t) -> np.ndarray:
        """:math:`\\delta(t)` as an ``ndarray``."""
        return self._func(np.clip(np.asarray(t, dtype=float), 0.0, 1.0))

    def derivative(self, t):
        r""":math:`\delta'(t)` (one-sided at the endpoints)."""
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        if self._derivative is not None:
            out = self._derivative(t)
        else:
            h = 1e-6
            lo = np.clip(t - h, 0.0, 1.0)
            hi = np.clip(t + h, 0.0, 1.0)
            out = (self._func(hi) - self._func(lo)) / (hi - lo)
        return float(out) if np.ndim(out) == 0 else out

    def hat(self, t) -> np.ndarray:
        r""":math:`\hat\delta(t) = t - \delta(t)` (1-Lipschitz and nonnegative)."""
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        return t - self._func(t)

    # -- axioms -------------------------------------------------------------
    def check(self, m: int = 2000, tol: float = 1e-9) -> DiagonalCheck:
        """Grid check of (D1)--(D3), see :func:`check_diagonal`."""
        return check_diagonal(self, m=m, tol=tol)

    def is_valid(self, m: int = 2000, tol: float = 1e-9) -> bool:
        """Whether (D1)--(D3) hold on a grid."""
        return self.check(m=m, tol=tol).is_diagonal

    # -- dependence quantities ---------------------------------------------------
    def _as_cdf(self):
        f = self._func
        return lambda u, v: f(np.asarray(u, dtype=float))

    def spearmans_footrule(self) -> float:
        r""":math:`\phi = 6\int_0^1\delta(t)\,dt - 2`."""
        return footrule_from_diagonal(self)

    def blomqvists_beta(self) -> float:
        r""":math:`\beta = 4\delta(\tfrac12) - 1`."""
        return blomqvist_from_diagonal(self)

    def lambda_L(self) -> float:
        r""":math:`\lambda_L = \delta'(0^+) = \lim_{t\downarrow 0}\delta(t)/t`."""
        return tail_dependence_from_diagonal(self)[0]

    def lambda_U(self) -> float:
        r""":math:`\lambda_U = 2 - \delta'(1^-)`."""
        return tail_dependence_from_diagonal(self)[1]

    # -- copulas with this diagonal ---------------------------------------------
    def bertino(self) -> BertinoCopula:
        """The Bertino copula :math:`B_\\delta`."""
        return BertinoCopula(self)

    def diagonal_copula(self) -> DiagonalCopula:
        """The Fredricks–Nelsen diagonal copula :math:`K_\\delta`."""
        return DiagonalCopula(self)

    def plot(self, ax=None, n: int = 401, **kwargs):
        """Plot :math:`\\delta` with the bounds :math:`\\max(2t-1,0)` and :math:`t`."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        t = np.linspace(0.0, 1.0, n)
        ax.plot(t, self.vectorized(t), label=self.name, **kwargs)
        ax.plot(t, t, "k--", lw=0.8, label="t (M)")
        ax.plot(t, np.maximum(2 * t - 1, 0), "k:", lw=0.8, label="max(2t-1, 0) (W)")
        ax.set_xlabel("t")
        ax.set_ylabel("delta(t)")
        ax.set_aspect("equal")
        ax.legend()
        return ax


def diagonal_section(C: Any) -> Diagonal:
    r"""Diagonal section :math:`\delta_C(t) = C(t,t)` of a copula (vectorized).

    Parameters
    ----------
    C : copula, NumericQuasiCopula or callable ``f(u, v)``

    Returns
    -------
    Diagonal

    Examples
    --------
    >>> import copul as cp
    >>> from copul.theory.diagonal import diagonal_section
    >>> delta = diagonal_section(cp.Clayton(2))
    >>> round(delta(0.5), 12) == round(cp.Clayton(2).cdf(0.5, 0.5), 12)
    True
    """
    f = cdf_function(C)
    return Diagonal(lambda t: f(t, t), name=f"diagonal_section({C!r})")


def opposite_diagonal(C: Any) -> Callable[[Any], Any]:
    r"""Opposite diagonal section :math:`\omega_C(t) = C(t, 1-t)` (vectorized).

    Parameters
    ----------
    C : copula, NumericQuasiCopula or callable ``f(u, v)``

    Returns
    -------
    callable
        ``omega(t)``; a float for scalar input.
    """
    f = cdf_function(C)

    def omega(t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        out = f(t, 1.0 - t)
        return float(out) if out.ndim == 0 else out

    return omega


def as_diagonal(delta: Any) -> Diagonal:
    """A :class:`Diagonal` from a ``Diagonal``, a callable or a copula (its diagonal)."""
    if isinstance(delta, Diagonal):
        return delta
    if hasattr(delta, "cdf") and hasattr(delta, "dim"):
        return diagonal_section(delta)
    if callable(delta):
        return Diagonal(delta)
    raise TypeError(f"cannot interpret {type(delta).__name__} as a diagonal")


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------


@dataclass
class DiagonalCheck:
    r"""Result of :func:`check_diagonal`.

    Attributes
    ----------
    is_diagonal : bool
        (D1)--(D3) hold up to the tolerance.
    endpoints_ok : bool
        :math:`\delta(0)=0` and :math:`\delta(1)=1`.
    below_identity_ok : bool
        :math:`\delta(t)\le t`.
    increasing_ok : bool
        :math:`\delta` nondecreasing.
    lipschitz_ok : bool
        :math:`\delta(t_2)-\delta(t_1)\le 2(t_2-t_1)`.
    above_lower_ok : bool
        :math:`\delta(t)\ge\max(2t-1,0)` (implied by the others).
    max_violation : float
        Largest violation of any condition.
    grid : int
    """

    is_diagonal: bool
    endpoints_ok: bool
    below_identity_ok: bool
    increasing_ok: bool
    lipschitz_ok: bool
    above_lower_ok: bool
    max_violation: float
    grid: int

    def __bool__(self) -> bool:
        return self.is_diagonal


def check_diagonal(delta: Any, m: int = 2000, tol: float = 1e-9) -> DiagonalCheck:
    r"""Check (D1)--(D3) for a candidate diagonal on the grid :math:`\{k/m\}`.

    .. math::

       \delta(1)=1,\qquad \delta(t)\le t,\qquad
       0\le\delta(t_2)-\delta(t_1)\le 2(t_2-t_1)\quad(t_1\le t_2).

    Parameters
    ----------
    delta : Diagonal, callable or copula
    m : int
        Number of grid intervals.
    tol : float
        Absolute tolerance.

    Returns
    -------
    DiagonalCheck
    """
    d = as_diagonal(delta)
    t = np.linspace(0.0, 1.0, int(m) + 1)
    y = d.vectorized(t)
    h = 1.0 / int(m)
    dy = np.diff(y)
    viol = {
        "endpoints": max(abs(y[0]), abs(y[-1] - 1.0)),
        "below": max(0.0, float(np.max(y - t))),
        "increasing": max(0.0, float(-dy.min())),
        "lipschitz": max(0.0, float(dy.max() - 2 * h)),
        "lower": max(0.0, float(np.max(np.maximum(2 * t - 1, 0) - y))),
    }
    ok = {k: bool(v <= tol) for k, v in viol.items()}
    if not np.all(np.isfinite(y)):
        ok = dict.fromkeys(ok, False)
    return DiagonalCheck(
        is_diagonal=ok["endpoints"] and ok["below"] and ok["increasing"] and ok["lipschitz"],
        endpoints_ok=ok["endpoints"],
        below_identity_ok=ok["below"],
        increasing_ok=ok["increasing"],
        lipschitz_ok=ok["lipschitz"],
        above_lower_ok=ok["lower"],
        max_violation=float(max(viol.values())),
        grid=int(m),
    )


def is_diagonal(delta: Any, m: int = 2000, tol: float = 1e-9) -> bool:
    """Whether ``delta`` satisfies the diagonal axioms (D1)--(D3) on a grid.

    Examples
    --------
    >>> from copul.theory.diagonal import is_diagonal
    >>> is_diagonal(lambda t: t**2), is_diagonal(lambda t: t**3)
    (True, False)
    """
    return check_diagonal(delta, m=m, tol=tol).is_diagonal


# ---------------------------------------------------------------------------
# interval minima of continuous functions (for B_delta and A_delta)
# ---------------------------------------------------------------------------

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


class IntervalMin:
    r"""Vectorized :math:`\min_{t\in[x,y]} f(t)` for a continuous :math:`f` on :math:`[0,1]`.

    A sparse table over the grid :math:`\{k/n\}` locates the grid minimum
    in :math:`O(1)`; it is refined by one parabolic interpolation step
    through the grid minimum and its two neighbours (clipped to
    :math:`[x,y]`), optionally followed by golden-section steps around the
    parabola's vertex, and compared with the exact values :math:`f(x)`,
    :math:`f(y)` (three evaluations of :math:`f` per query).  The result is never larger than the grid minimum; for
    functions that are smooth near the minimizer the error is of order
    :math:`n^{-4}`, at a kink it is at most :math:`L/(2n)` for an
    :math:`L`-Lipschitz :math:`f`.

    Parameters
    ----------
    f : callable
        Vectorized function on :math:`[0,1]`.
    n : int
        Number of grid intervals (default :math:`2^{14}`).
    iters : int
        Golden-section steps after the parabolic step (default 0).
    """

    def __init__(self, f: Callable[[np.ndarray], np.ndarray], n: int = 16384, iters: int = 0):
        self.f = f
        self.n = int(n)
        self.iters = int(iters)
        self.t = np.linspace(0.0, 1.0, self.n + 1)
        vals = np.asarray(f(self.t), dtype=float)
        self._val = [vals]
        self._idx = [np.arange(vals.size)]
        j = 1
        while (1 << j) <= vals.size:
            pv, pi = self._val[-1], self._idx[-1]
            half = 1 << (j - 1)
            a, b = pv[:-half], pv[half:]
            take_b = b < a
            self._val.append(np.where(take_b, b, a))
            self._idx.append(np.where(take_b, pi[half:], pi[:-half]))
            j += 1

    def __call__(self, x, y) -> np.ndarray:
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        shape = x.shape
        x = np.clip(x.ravel(), 0.0, 1.0)
        y = np.clip(y.ravel(), 0.0, 1.0)
        x, y = np.minimum(x, y), np.maximum(x, y)
        f = self.f
        fx, fy = f(x), f(y)
        best = np.minimum(fx, fy)
        n = self.n
        i0 = np.ceil(x * n - 1e-9).astype(int)
        i1 = np.floor(y * n + 1e-9).astype(int)
        valid = i0 <= i1
        if np.any(valid):
            a0, a1 = i0[valid], i1[valid]
            length = a1 - a0 + 1
            lev = np.floor(np.log2(length)).astype(int)
            v_left = np.empty(a0.size)
            k_left = np.empty(a0.size, dtype=int)
            v_right = np.empty(a0.size)
            k_right = np.empty(a0.size, dtype=int)
            for j in np.unique(lev):
                sel = lev == j
                v_left[sel] = self._val[j][a0[sel]]
                k_left[sel] = self._idx[j][a0[sel]]
                r = a1[sel] - (1 << j) + 1
                v_right[sel] = self._val[j][r]
                k_right[sel] = self._idx[j][r]
            take_r = v_right < v_left
            kmin = np.where(take_r, k_right, k_left)
            gmin = np.where(take_r, v_right, v_left)
            xv, yv = x[valid], y[valid]
            km, kp = np.maximum(kmin - 1, 0), np.minimum(kmin + 1, n)
            vals = self._val[0]
            lo_is_x = self.t[km] <= xv
            hi_is_y = self.t[kp] >= yv
            lo = np.where(lo_is_x, xv, self.t[km])
            hi = np.where(hi_is_y, yv, self.t[kp])
            flo = np.where(lo_is_x, fx[valid], vals[km])
            fhi = np.where(hi_is_y, fy[valid], vals[kp])
            ref = self._refine(lo, self.t[kmin], hi, flo, gmin, fhi)
            best[valid] = np.minimum(best[valid], np.minimum(gmin, ref))
        return best.reshape(shape)

    def _refine(self, lo, mid, hi, flo, fmid, fhi):
        """Parabolic step through (lo, mid, hi) (+ optional golden-section steps)."""
        f = self.f
        a, b = mid - lo, mid - hi
        num = a * a * (fmid - fhi) - b * b * (fmid - flo)
        den = a * (fmid - fhi) - b * (fmid - flo)
        with np.errstate(all="ignore"):
            p = np.where(den != 0.0, mid - 0.5 * num / den, mid)
        p = np.clip(np.nan_to_num(p, nan=0.0), lo, hi)
        best = f(p)
        if self.iters > 0:
            w = 0.25 * (hi - lo)
            best = np.minimum(best, self._golden(np.maximum(lo, p - w), np.minimum(hi, p + w)))
        return best

    def _golden(self, lo, hi):
        f = self.f
        c = hi - _GOLDEN * (hi - lo)
        d = lo + _GOLDEN * (hi - lo)
        fc, fd = f(c), f(d)
        for _ in range(self.iters):
            left = fc < fd
            lo = np.where(left, lo, c)
            hi = np.where(left, d, hi)
            xn = np.where(left, hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo))
            fn = f(xn)
            c, d, fc, fd = (
                np.where(left, xn, d),
                np.where(left, c, xn),
                np.where(left, fn, fd),
                np.where(left, fc, fn),
            )
        return np.minimum(fc, fd)


# ---------------------------------------------------------------------------
# copulas with a given diagonal
# ---------------------------------------------------------------------------


class BertinoCopula(InversionSamplingCopula):
    r"""Bertino copula of a diagonal :math:`\delta` (Bertino 1977; Fredricks & Nelsen 1997).

    .. math::

       B_\delta(u,v) = \min(u,v) - \min_{t\in[u\wedge v,\,u\vee v]}\bigl(t-\delta(t)\bigr).

    :math:`B_\delta` is a symmetric copula with diagonal section
    :math:`\delta`, and :math:`B_\delta\le C` for every copula :math:`C`
    with diagonal section :math:`\delta` (Fredricks & Nelsen 1997; Nelsen
    2006, Sect. 3.2.6).  Indeed, for :math:`u\le t\le v`,
    :math:`C(u,v)\ge C(u,t)\ge C(t,t)-(t-u)=u-(t-\delta(t))`.

    Parameters
    ----------
    delta : Diagonal, callable or copula
        A diagonal (a copula is replaced by its diagonal section).
    check : bool
        Validate (D1)--(D3) on a grid (raise ``ValueError`` otherwise).
    grid : int
        Grid size of the interval-minimum search (see :class:`IntervalMin`).

    Notes
    -----
    With :math:`m(u,v)=\min_{[u\wedge v,u\vee v]}\hat\delta`,
    :math:`\hat\delta(t)=t-\delta(t)`, the conditional distribution is
    :math:`\partial_1 B_\delta(u,v) = \mathbf 1\{u<v\}-\hat\delta'(u)\,
    \mathbf 1\{m(u,v)=\hat\delta(u)\}` (almost everywhere).

    Examples
    --------
    >>> from copul.theory.diagonal import BertinoCopula
    >>> B = BertinoCopula(lambda t: t**2)       # Bertino copula of Pi's diagonal
    >>> round(B.cdf(0.3, 0.8), 12)              # 0.3 - min(0.3 - 0.09, 0.8 - 0.64)
    0.14
    """

    def __init__(self, delta: Any, check: bool = True, grid: int = 16384):
        self.delta = as_diagonal(delta)
        if check:
            res = check_diagonal(self.delta)
            if not res.is_diagonal:
                raise ValueError(f"{self.delta!r} is not a diagonal: {res}")
        self._min = IntervalMin(self.delta.hat, n=grid)
        super().__init__()

    def __repr__(self) -> str:
        return f"BertinoCopula({self.delta!r})"

    __str__ = __repr__

    @property
    def is_symmetric(self) -> bool:
        return True

    def _cdf(self, u, v):
        return np.minimum(u, v) - self._min(u, v)

    def _h1(self, u, v):
        # B = min(u,v) - m(u^v, u v v); d/du m = hat'(u) iff the minimum of
        # hat over the interval is attained at the endpoint u (else 0)
        m = self._min(u, v)
        at_u = self.delta.hat(u) <= m + 1e-14
        dhat = 1.0 - np.asarray(self.delta.derivative(u), dtype=float)
        return np.where(u < v, 1.0, 0.0) - np.where(at_u, dhat, 0.0)

    def _h2(self, u, v):
        return self._h1(v, u)


class DiagonalCopula(InversionSamplingCopula):
    r"""Diagonal copula of Fredricks & Nelsen (1997).

    .. math::

       K_\delta(u,v) = \min\Bigl(u, v, \tfrac12\bigl(\delta(u)+\delta(v)\bigr)\Bigr).

    For a diagonal :math:`\delta`, :math:`K_\delta` is a symmetric copula
    with diagonal section :math:`\delta`, and it is the largest symmetric
    one: :math:`C\le K_\delta` for every symmetric copula :math:`C` with
    :math:`\delta_C=\delta` (Fredricks & Nelsen 1997; Nelsen 2006, Sect.
    3.2.6).  The bound follows from the nonnegative :math:`C`-volume of
    :math:`[u,v]^2`: :math:`C(u,v)+C(v,u)\le\delta(u)+\delta(v)`.

    Parameters
    ----------
    delta : Diagonal, callable or copula
    check : bool
        Validate (D1)--(D3) on a grid (raise ``ValueError`` otherwise).

    Examples
    --------
    >>> from copul.theory.diagonal import DiagonalCopula
    >>> K = DiagonalCopula(lambda t: t**2)
    >>> round(K.cdf(0.3, 0.8), 12)              # (0.09 + 0.64) / 2
    0.3
    """

    def __init__(self, delta: Any, check: bool = True):
        self.delta = as_diagonal(delta)
        if check:
            res = check_diagonal(self.delta)
            if not res.is_diagonal:
                raise ValueError(f"{self.delta!r} is not a diagonal: {res}")
        super().__init__()

    def __repr__(self) -> str:
        return f"DiagonalCopula({self.delta!r})"

    __str__ = __repr__

    @property
    def is_symmetric(self) -> bool:
        return True

    def _cdf(self, u, v):
        d = self.delta.vectorized
        return np.minimum(np.minimum(u, v), 0.5 * (d(u) + d(v)))

    def _h1(self, u, v):
        d = self.delta.vectorized
        half = 0.5 * (d(u) + d(v))
        dd = np.asarray(self.delta.derivative(u), dtype=float)
        return np.where(
            (u <= v) & (u <= half), 1.0, np.where(half < np.minimum(u, v), 0.5 * dd, 0.0)
        )

    def _h2(self, u, v):
        return self._h1(v, u)


def bertino_copula(delta: Any, check: bool = True) -> BertinoCopula:
    """Bertino copula :math:`B_\\delta` of a diagonal, see :class:`BertinoCopula`."""
    return BertinoCopula(delta, check=check)


def diagonal_copula(delta: Any, check: bool = True) -> DiagonalCopula:
    """Fredricks–Nelsen diagonal copula :math:`K_\\delta`, see :class:`DiagonalCopula`."""
    return DiagonalCopula(delta, check=check)


def copula_with_diagonal(delta: Any, kind: str = "bertino", check: bool = True):
    r"""A copula with prescribed diagonal section.

    Parameters
    ----------
    delta : Diagonal, callable or copula
        A diagonal.
    kind : {"bertino", "diagonal"}
        ``"bertino"``: the smallest copula with diagonal :math:`\delta`
        (:class:`BertinoCopula`); ``"diagonal"``: the largest symmetric one
        (:class:`DiagonalCopula`).
    check : bool
        Validate the diagonal axioms first.

    Returns
    -------
    BertinoCopula or DiagonalCopula
    """
    kind = kind.lower()
    if kind == "bertino":
        return BertinoCopula(delta, check=check)
    if kind in ("diagonal", "fredricks-nelsen", "k"):
        return DiagonalCopula(delta, check=check)
    raise ValueError("kind must be 'bertino' or 'diagonal'")


# ---------------------------------------------------------------------------
# dependence quantities from diagonals
# ---------------------------------------------------------------------------


def footrule_from_diagonal(delta: Any) -> float:
    r"""Spearman's footrule :math:`\phi = 6\int_0^1\delta(t)\,dt - 2` (Nelsen 2006, Sect. 5.1)."""
    d = as_diagonal(delta).vectorized
    val, _ = integrate_1d(d, 0.0, 1.0, atol=1e-13, rtol=1e-11)
    return 6.0 * val - 2.0


def gini_from_diagonals(delta: Any, omega: Callable | None = None) -> float:
    r"""Gini's :math:`\gamma = 4\int_0^1[\delta(t)+\omega(t)]\,dt - 2` (Nelsen 2006, Sect. 5.1).

    Parameters
    ----------
    delta : Diagonal, callable or copula
        Diagonal section (if a copula is passed and ``omega`` is ``None``,
        its opposite diagonal section is used as well).
    omega : callable, optional
        Opposite diagonal section :math:`\omega(t) = C(t,1-t)`.
    """
    if omega is None:
        if not (hasattr(delta, "cdf") and hasattr(delta, "dim")):
            raise ValueError("omega is required unless a copula is passed")
        omega = opposite_diagonal(delta)
    d = as_diagonal(delta).vectorized
    om = _as_vectorized(omega)
    val, _ = integrate_1d(lambda t: d(t) + om(t), 0.0, 1.0, atol=1e-13, rtol=1e-11)
    return 4.0 * val - 2.0


def blomqvist_from_diagonal(delta: Any) -> float:
    r"""Blomqvist's :math:`\beta = 4\delta(\tfrac12) - 1`."""
    return 4.0 * float(as_diagonal(delta)(0.5)) - 1.0


def tail_dependence_from_diagonal(delta: Any) -> tuple[float, float]:
    r"""Tail dependence coefficients from the diagonal section.

    .. math::

       \lambda_L = \delta'(0^+) = \lim_{t\downarrow0}\frac{\delta(t)}{t},
       \qquad
       \lambda_U = 2-\delta'(1^-) = \lim_{t\uparrow1}\frac{1-2t+\delta(t)}{1-t}

    (Nelsen 2006, Sect. 5.4), evaluated on geometric sequences and
    extrapolated (see :func:`copul.measures.numeric.lambda_l_from_cdf`).

    Returns
    -------
    (lambda_L, lambda_U) : tuple of float
    """
    cdf = as_diagonal(delta)._as_cdf()
    return float(lambda_l_from_cdf(cdf)), float(lambda_u_from_cdf(cdf))
