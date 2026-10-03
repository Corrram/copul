r"""
Numerical Archimedean copulas and vectorized generator utilities.

A bivariate Archimedean copula is

.. math::

   C(u,v) = \psi\bigl(\varphi(u) + \varphi(v)\bigr),

with a continuous, strictly decreasing, convex generator
:math:`\varphi:[0,1]\to[0,\infty]`, :math:`\varphi(1)=0`, and its
pseudo-inverse :math:`\psi = \varphi^{[-1]}`, :math:`\psi(s)=0` for
:math:`s\ge\varphi(0)` (Nelsen, 2006, Thm. 4.1.4).  The generator is
*strict* if :math:`\varphi(0)=\infty`.  For :math:`C(u,v)>0`

.. math::

   \partial_1 C(u,v) = \frac{\varphi'(u)}{\varphi'(C(u,v))},\qquad
   c(u,v) = -\frac{\varphi''(C)\,\varphi'(u)\,\varphi'(v)}{\varphi'(C)^3},

and both vanish on the zero set :math:`\{C=0\}`.  For a non-strict generator
the zero curve :math:`\varphi(u)+\varphi(v)=\varphi(0)` carries the singular
mass :math:`-\varphi(0)/\varphi'(0^+)` (Nelsen, 2006, Sec. 4.3).

This module provides

* :class:`ArchimedeanGenerator` -- the vectorized numeric generator data
  (:math:`\varphi`, :math:`\varphi'`, :math:`\varphi''`, :math:`\psi`,
  :math:`\varphi(0)`) of an Archimedean copula, as returned by the
  ``_archimedean_generator()`` hooks of the Archimedean families;
* :class:`NumericArchimedeanCopula` -- a fully usable Archimedean copula
  given by numerical callables (any of :math:`\varphi`, :math:`\psi` and
  optionally their derivatives), with exact conditional inversion
  (sampling without frailties), closed-form Kendall's :math:`\tau` and the
  Archimedean theory methods;
* small vectorized numerical tools (bracketed Illinois root finding, finite
  differences) shared by :mod:`copul.theory.archimedean`.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer, Ch. 4.
Genest, C. & Rivest, L.-P. (1993). Statistical inference procedures for
bivariate Archimedean copulas. *JASA* 88, 1034--1043.
McNeil, A. J. & Nešlehová, J. (2009). Multivariate Archimedean copulas,
d-monotone functions and l1-norm symmetric distributions. *Ann. Statist.*
37, 3059--3097.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from copul.family.archimedean._theory_mixins import (
    ArchimedeanGeneratorMixin,
    BivArchimedeanTheoryMixin,
)
from copul.family.constructions._base import NumericBivCopula

__all__ = [
    "ArchimedeanGenerator",
    "ArchimedeanGeneratorMixin",
    "BivArchimedeanTheoryMixin",
    "NumericArchimedeanCopula",
    "fd_derivative",
    "solve_increasing",
]

_TINY = 1e-300


# ---------------------------------------------------------------------------
# numerical tools
# ---------------------------------------------------------------------------


def _call(f: Callable, x) -> np.ndarray:
    """``f(x)`` as a float array of the shape of ``x`` (errors silenced)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        out = np.asarray(f(x), dtype=float)
    if out.shape != x.shape:
        out = np.broadcast_to(out, x.shape).copy()
    return out


def solve_increasing(
    g: Callable,
    lo,
    hi,
    glo=None,
    ghi=None,
    *,
    xtol: float = 4e-16,
    maxiter: int = 200,
) -> np.ndarray:
    r"""Vectorized root of nondecreasing functions on brackets.

    Finds :math:`x_i\in[lo_i, hi_i]` with :math:`g(x_i, i) = 0` for
    functions with :math:`g(lo_i,i)\le 0\le g(hi_i,i)`, using the Illinois
    variant of regula falsi (superlinear on smooth parts, never leaves the
    bracket) with a bisection step whenever the secant point is unusable and
    every eighth iteration.

    Parameters
    ----------
    g : callable
        ``g(x, idx)`` evaluated on the active entries; ``idx`` holds their
        indices into the flattened problem.
    lo, hi : array_like
        Brackets (broadcast to a common shape).
    glo, ghi : array_like, optional
        Values of ``g`` at the bracket ends (evaluated if omitted).
    xtol : float
        Relative bracket width at which an entry counts as converged.
    maxiter : int
        Maximal number of iterations.

    Returns
    -------
    numpy.ndarray
        The roots, in the broadcast shape of ``lo`` and ``hi``.
    """
    lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
    shape = lo.shape
    a = lo.ravel().copy()
    b = hi.ravel().copy()
    idx_all = np.arange(a.size)
    with np.errstate(all="ignore"):
        fa = (
            np.asarray(g(a, idx_all), float)
            if glo is None
            else np.broadcast_to(np.asarray(glo, float), shape).ravel().copy()
        )
        fb = (
            np.asarray(g(b, idx_all), float)
            if ghi is None
            else np.broadcast_to(np.asarray(ghi, float), shape).ravel().copy()
        )
    x = 0.5 * (a + b)
    x = np.where(fa >= 0, a, np.where(fb <= 0, b, x))
    side = np.zeros(a.size, dtype=np.int8)
    active = np.flatnonzero((fa < 0) & (fb > 0) & (b - a > xtol * (1.0 + np.abs(a))))
    for it in range(maxiter):
        if active.size == 0:
            break
        A, B, FA, FB = a[active], b[active], fa[active], fb[active]
        mid = 0.5 * (A + B)
        with np.errstate(all="ignore"):
            c = (A * FB - B * FA) / (FB - FA)
        bad = ~np.isfinite(c) | (c <= A) | (c >= B)
        if it % 8 == 7:
            bad[:] = True
        c = np.where(bad, mid, c)
        with np.errstate(all="ignore"):
            fc = np.asarray(g(c, active), float)
        fc = np.where(np.isnan(fc), 0.0, fc)
        neg = fc < 0
        s_old = side[active]
        FB = np.where(neg & (s_old == -1), 0.5 * FB, FB)
        FA = np.where(~neg & (s_old == 1), 0.5 * FA, FA)
        A = np.where(neg, c, A)
        FA = np.where(neg, fc, FA)
        B = np.where(neg, B, c)
        FB = np.where(neg, FB, fc)
        a[active], b[active], fa[active], fb[active] = A, B, FA, FB
        side[active] = np.where(neg, -1, 1)
        x[active] = c
        conv = (fc == 0) | (xtol * (1.0 + np.abs(c)) >= B - A)
        active = active[~conv]
    return x.reshape(shape)


def fd_derivative(
    f: Callable,
    x,
    order: int = 1,
    lower: float = -np.inf,
    upper: float = np.inf,
    rel: float = 2e-3,
    hmin: float = 1e-7,
) -> np.ndarray:
    r"""Vectorized fourth-order finite differences of ``f`` on ``[lower, upper]``.

    The step is ``rel`` times the distance of ``x`` to the nearest finite
    bound (at least ``rel * hmin``), or ``rel * max(|x|, 1)`` without bounds;
    central five-point stencils are used where they fit into the domain,
    one-sided five-point stencils otherwise.

    Parameters
    ----------
    f : callable
        Vectorized function.
    x : array_like
        Evaluation points.
    order : {1, 2}
        Order of the derivative.
    lower, upper : float
        Domain of ``f``.
    rel, hmin : float
        Relative step and minimal distance used for the step.

    Returns
    -------
    numpy.ndarray
    """
    x = np.asarray(x, dtype=float)
    dist = np.minimum(x - lower, upper - x)
    scale = np.where(np.isfinite(dist), dist, np.maximum(np.abs(x), 1.0))
    h = rel * np.maximum(scale, hmin)
    fwd = (x - 4 * h < lower) & (x + 4 * h <= upper)
    bwd = (x + 4 * h > upper) & ~fwd
    sgn = np.where(bwd, -1.0, 1.0)
    one_sided = fwd | bwd
    hs = sgn * h
    pts = np.stack([x + k * hs for k in range(5)])  # one-sided nodes
    cen = np.stack([x + k * h for k in (-2, -1, 0, 1, 2)])
    nodes = np.where(one_sided, pts, cen)
    vals = _call(f, nodes)
    f0, f1, f2, f3, f4 = vals
    if order == 1:
        d_c = (f0 - 8 * f1 + 8 * f3 - f4) / (12 * h)
        d_o = (-25 * f0 + 48 * f1 - 36 * f2 + 16 * f3 - 3 * f4) / (12 * hs)
    elif order == 2:
        d_c = (-f0 + 16 * f1 - 30 * f2 + 16 * f3 - f4) / (12 * h**2)
        d_o = (35 * f0 - 104 * f1 + 114 * f2 - 56 * f3 + 11 * f4) / (12 * h**2)
    else:  # pragma: no cover - not needed
        raise ValueError("order must be 1 or 2")
    return np.where(one_sided, d_o, d_c)


# ---------------------------------------------------------------------------
# generator data
# ---------------------------------------------------------------------------


@dataclass
class ArchimedeanGenerator:
    r"""Vectorized numeric generator of a bivariate Archimedean copula.

    Attributes
    ----------
    phi : callable
        Generator :math:`\varphi(t)` on :math:`[0,1]`.
    dphi : callable
        Derivative :math:`\varphi'(t)` (negative on :math:`(0,1)`).
    psi : callable
        Pseudo-inverse :math:`\psi(s)=\varphi^{[-1]}(s)` on
        :math:`[0,\infty)`, zero for :math:`s\ge\varphi(0)`.
    phi0 : float
        :math:`\varphi(0^+)` (``inf`` for strict generators).
    d2phi : callable or None
        Second derivative :math:`\varphi''(t)` (finite differences of
        ``dphi`` if ``None``).
    ratio : callable or None
        :math:`\varphi(t)/\varphi'(t)` computed without overflow (the
        quotient of ``phi`` and ``dphi`` if ``None``).
    source : str
        Where the callables come from (``"sympy"``, ``"log-scale"``,
        ``"numeric"``).
    """

    phi: Callable
    dphi: Callable
    psi: Callable
    phi0: float
    d2phi: Callable | None = None
    ratio: Callable | None = None
    source: str = ""

    @property
    def is_strict(self) -> bool:
        r"""Whether :math:`\varphi(0)=\infty`."""
        return not math.isfinite(self.phi0)

    def phi_over_dphi(self, t) -> np.ndarray:
        r""":math:`\varphi(t)/\varphi'(t)` (vectorized; ``0`` at :math:`t=1`)."""
        t = np.asarray(t, dtype=float)
        if self.ratio is not None:
            r = _call(self.ratio, t)
        else:
            with np.errstate(all="ignore"):
                r = _call(self.phi, t) / _call(self.dphi, t)
        bad = ~np.isfinite(r)
        if np.any(bad):
            # limits: phi/phi' -> 0 at t = 1 and, for strict generators, at t = 0;
            # -> -(zero-curve mass) at t = 0 for non-strict generators
            low = -self.zero_curve_mass() if not self.is_strict else 0.0
            r = np.where(bad, np.where(t < 0.5, low, 0.0), r)
        return np.where(t >= 1.0, 0.0, r)

    def second_derivative(self, t) -> np.ndarray:
        r""":math:`\varphi''(t)` (class formula or finite differences)."""
        if self.d2phi is not None:
            return _call(self.d2phi, t)
        return fd_derivative(self.dphi, t, 1, 0.0, 1.0)

    def zero_curve_mass(self) -> float:
        r"""Singular mass :math:`-\varphi(0)/\varphi'(0^+)` of the zero curve.

        Zero for strict generators and for :math:`\varphi'(0^+)=-\infty`
        (Nelsen, 2006, Sec. 4.3).
        """
        if self.is_strict:
            return 0.0
        t = np.array([1e-300, 1e-200, 1e-100, 1e-60, 1e-30, 1e-16])
        d = _call(self.dphi, t)
        ok = np.isfinite(d) & (d < 0)
        if not np.any(ok):
            return 0.0
        dphi0 = float(d[np.flatnonzero(ok)[0]])
        mass = float(np.clip(-self.phi0 / dphi0, 0.0, 1.0))
        # phi'(0+) = -inf shows up as a huge finite slope at t = 1e-300
        return mass if mass > 1e-14 else 0.0


def _strip_support_factors(expr):
    """Remove ``Heaviside``/indicator ``Piecewise`` factors and ``Max(0, .)`` of a
    pseudo-inverse expression (the support is handled separately)."""
    import sympy as sp

    expr = expr.replace(sp.Heaviside, lambda *a: sp.Integer(1))

    def _pw(*args):
        vals = [a[0] for a in args]
        if len(args) == 2 and vals[1] == 0:
            return vals[0]
        return sp.Piecewise(*args)

    expr = expr.replace(sp.Piecewise, _pw)

    def _max(*args):
        rest = [a for a in args if a != 0]
        return rest[0] if len(rest) == 1 else sp.Max(*args)

    return expr.replace(sp.Max, _max)


def generator_from_sympy(copula) -> ArchimedeanGenerator:
    r"""Numeric generator data from the symbolic generator of an Archimedean family.

    :math:`\varphi`, :math:`\varphi'`, :math:`\varphi''` and the overflow-free
    quotient :math:`\varphi/\varphi'` are lambdified (parameters
    substituted); the symbolic pseudo-inverse is used if it reproduces
    :math:`\psi(\varphi(t))=t`, otherwise :math:`\varphi` is inverted
    numerically.  Cached on the instance per parameter value.
    """
    import sympy as sp

    from copul.measures.backend import _param_key
    from copul.numerics import to_numpy_callable

    key = _param_key(copula)
    hit = copula.__dict__.get("_copul_archimedean_generator")
    if hit is not None and hit[0] == key:
        return hit[1]
    t = copula.t
    expr = copula.generator.func
    at0 = None
    if isinstance(expr, sp.Piecewise) and len(expr.args) == 2:
        expr, at0 = expr.args[0][0], expr.args[1][0]
    free = expr.free_symbols - {t}
    if free:
        raise ValueError(f"{type(copula).__name__} has free parameters {sorted(map(str, free))}.")
    phi = to_numpy_callable(expr, [t])
    d1 = sp.diff(expr, t)
    dphi = to_numpy_callable(d1, [t])
    d2phi = to_numpy_callable(sp.diff(expr, t, 2), [t])
    lazy: dict = {}

    def ratio(x):
        # phi / phi', simplified once on first use (avoids overflow of the quotient)
        f = lazy.get("ratio")
        if f is None:
            ratio_expr = expr / d1
            try:
                simp = sp.simplify(ratio_expr)
                if sp.count_ops(simp) <= sp.count_ops(ratio_expr):
                    ratio_expr = simp
            except Exception:  # pragma: no cover - simplification is optional
                pass
            f = lazy["ratio"] = to_numpy_callable(ratio_expr, [t])
        return f(x)

    phi0 = math.inf
    try:
        if at0 is not None and at0.is_number:
            phi0 = float(at0) if bool(at0.is_finite) else math.inf
        else:
            v = float(_call(phi, np.array([0.0]))[0])
            phi0 = v if np.isfinite(v) else math.inf
    except Exception:  # pragma: no cover
        pass
    tt = np.array([0.03, 0.2, 0.5, 0.8, 0.97])
    psi = None
    try:
        inv = copula.inv_generator
        inv_expr = _strip_support_factors(getattr(inv, "func", inv))
        y = copula.y
        if not (inv_expr.free_symbols - {y}):
            raw_psi = to_numpy_callable(inv_expr, [y])
            if np.allclose(_call(raw_psi, _call(phi, tt)), tt, rtol=1e-9, atol=1e-12):
                psi = raw_psi
    except Exception:
        psi = None
    if psi is None:
        psi = NumericArchimedeanCopula(phi=phi, dphi=dphi, d2phi=d2phi, phi0=phi0)._psi
        source = "sympy+inversion"
    else:
        source = "sympy"
        raw = psi

        def psi(s, _raw=raw, _phi0=phi0):
            s = np.maximum(np.asarray(s, float), 0.0)
            out = _call(_raw, s)
            out = np.where(s >= _phi0, 0.0, out)
            out = np.where(np.isnan(out) & (s > 1.0), 0.0, out)
            return np.clip(np.where(s <= 0.0, 1.0, out), 0.0, 1.0)

    gen = ArchimedeanGenerator(
        phi=phi, dphi=dphi, psi=psi, phi0=phi0, d2phi=d2phi, ratio=ratio, source=source
    )
    copula.__dict__["_copul_archimedean_generator"] = (key, gen)
    return gen


# ---------------------------------------------------------------------------
# numeric Archimedean copula
# ---------------------------------------------------------------------------


def _phi_table_t() -> np.ndarray:
    small = np.exp(np.linspace(-700.0, -1.0, 400))
    mid = np.linspace(np.exp(-1.0), 0.999, 300)
    near1 = 1.0 - np.geomspace(1e-3, 1e-15, 60)
    return np.unique(np.concatenate([[0.0], small, mid, near1, [1.0]]))


def _psi_table_s(phi0: float) -> np.ndarray:
    if math.isfinite(phi0):
        inner = phi0 * np.concatenate([np.geomspace(1e-16, 1e-3, 60), np.linspace(1e-3, 1.0, 400)])
        return np.unique(np.concatenate([[0.0], inner]))
    return np.unique(np.concatenate([[0.0], np.geomspace(1e-16, 1e300, 1200)]))


class NumericArchimedeanCopula(BivArchimedeanTheoryMixin, NumericBivCopula):
    r"""Bivariate Archimedean copula given by numerical callables.

    At least one of the generator :math:`\varphi` and its (pseudo-)inverse
    :math:`\psi=\varphi^{[-1]}` must be given; the other one is obtained by
    vectorized monotone inversion (table bracketing and Illinois iterations,
    accurate to machine precision).  Missing derivatives are computed from
    the given ones (:math:`\varphi' = 1/\psi'(\varphi)`,
    :math:`\varphi''=-\psi''(\varphi)/\psi'(\varphi)^3`) or by fourth-order
    finite differences.

    The copula is fully usable: ``cdf``, ``pdf``, ``cond_distr_1/2`` and
    their inverses, ``rvs`` (exact conditional inversion, no frailty
    needed), every dependence measure (Kendall's :math:`\tau` by the
    one-dimensional formula :math:`\tau = 1 + 4\int_0^1\varphi/\varphi'`,
    Nelsen 2006, Cor. 5.1.4) and the Archimedean theory methods
    (``kendall_distribution``, ``generator_properties``, ``zero_curve``...).

    Parameters
    ----------
    phi : callable, optional
        Vectorized generator :math:`\varphi` on :math:`[0,1]`.
    psi : callable, optional
        Vectorized inverse generator :math:`\psi` on :math:`[0,\infty)`.
    dphi, d2phi : callable, optional
        :math:`\varphi'` and :math:`\varphi''`.
    dpsi, d2psi : callable, optional
        :math:`\psi'` and :math:`\psi''`.
    phi0 : float, optional
        :math:`\varphi(0^+)` (``inf`` for strict generators); detected
        automatically if omitted.
    name : str, optional
        Name used in ``repr``.

    Notes
    -----
    Conditional inversion: :math:`\partial_1C(u,v)=w` is equivalent to
    :math:`\varphi'(c)=\varphi'(u)/w` for :math:`c=C(u,v)`; if this has no
    solution :math:`c>0` (only possible for non-strict generators with
    :math:`\varphi'(0^+)>-\infty`) the conditional quantile lies on the zero
    curve.  Then :math:`v=\psi(\varphi(c)-\varphi(u))`.

    Examples
    --------
    >>> import numpy as np
    >>> from copul.family.archimedean.numeric_archimedean import NumericArchimedeanCopula
    >>> C = NumericArchimedeanCopula(psi=lambda s: (1 + s) ** -0.5)  # Clayton(2)
    >>> round(C.cdf(0.3, 0.7), 10) == round((0.3**-2 + 0.7**-2 - 1) ** -0.5, 10)
    True
    """

    def __init__(
        self,
        phi: Callable | None = None,
        psi: Callable | None = None,
        *,
        dphi: Callable | None = None,
        d2phi: Callable | None = None,
        dpsi: Callable | None = None,
        d2psi: Callable | None = None,
        phi0: float | None = None,
        name: str | None = None,
    ):
        if phi is None and psi is None:
            raise ValueError("NumericArchimedeanCopula needs phi or psi.")
        super().__init__()
        self._phi_user = phi
        self._psi_user = psi
        self._dphi_user = dphi
        self._d2phi_user = d2phi
        self._dpsi_user = dpsi
        self._d2psi_user = d2psi
        self._name = name
        self._phi0 = self._detect_phi0() if phi0 is None else float(phi0)
        self._tables: dict = {}
        self._mass = None

    # -- generator ---------------------------------------------------------------
    def _detect_phi0(self) -> float:
        if self._phi_user is not None:
            with np.errstate(all="ignore"):
                v = _call(self._phi_user, np.array([0.0, 1e-300]))
            if np.isfinite(v[0]):
                return float(v[0])
            return math.inf
        # psi given: support end sup{s : psi(s) > 0}
        s = np.concatenate([[0.0], np.geomspace(1e-12, 1e300, 1300)])
        p = _call(self._psi_user, s)
        zero = np.flatnonzero(np.isfinite(p) & (p <= 0.0))
        if zero.size == 0:
            return math.inf
        k = zero[0]
        lo, hi = s[k - 1], s[k]
        for _ in range(200):
            m = 0.5 * (lo + hi)
            if m <= lo or m >= hi:
                break
            if _call(self._psi_user, np.array([m]))[0] > 0:
                lo = m
            else:
                hi = m
        return float(hi)

    @property
    def phi0(self) -> float:
        r""":math:`\varphi(0^+)` (``inf`` for strict generators)."""
        return self._phi0

    @property
    def is_strict(self) -> bool:
        r"""Whether the generator is strict (:math:`\varphi(0)=\infty`)."""
        return not math.isfinite(self._phi0)

    def _phi(self, t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        if self._phi_user is not None:
            out = _call(self._phi_user, t)
        else:
            out = self._invert_psi(t)
        out = np.where(t >= 1.0, 0.0, out)
        return np.where(t <= 0.0, self._phi0, out)

    def _psi(self, s):
        s = np.maximum(np.asarray(s, dtype=float), 0.0)
        if self._psi_user is not None:
            out = _call(self._psi_user, s)
        else:
            out = self._invert_phi(s)
        out = np.where(s >= self._phi0, 0.0, out)
        out = np.where(np.isnan(out) & (s > 1.0), 0.0, out)
        return np.clip(np.where(s <= 0.0, 1.0, out), 0.0, 1.0)

    def _dpsi(self, s):
        r""":math:`\psi'(s)` (given, or finite differences with relative steps, which
        also resolve singular derivatives at :math:`s=0` such as Gumbel's)."""
        s = np.maximum(np.asarray(s, dtype=float), 0.0)
        if self._dpsi_user is not None:
            return _call(self._dpsi_user, s)
        return fd_derivative(self._psi_user, s, 1, 0.0, self._phi0, hmin=0.0)

    def _dphi(self, t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        if self._dphi_user is not None:
            return _call(self._dphi_user, t)
        if self._dpsi_user is not None:
            with np.errstate(all="ignore"):
                return 1.0 / _call(self._dpsi_user, self._phi(t))
        return fd_derivative(self._phi, t, 1, 0.0, 1.0)

    def _d2phi(self, t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        if self._d2phi_user is not None:
            return _call(self._d2phi_user, t)
        if self._d2psi_user is not None and self._dpsi_user is not None:
            s = self._phi(t)
            with np.errstate(all="ignore"):
                return -_call(self._d2psi_user, s) / _call(self._dpsi_user, s) ** 3
        if self._dphi_user is not None or self._dpsi_user is not None:
            return fd_derivative(self._dphi, t, 1, 0.0, 1.0)
        return fd_derivative(self._phi, t, 2, 0.0, 1.0)

    def _table(self, key):
        tab = self._tables.get(key)
        if tab is not None:
            return tab
        if key == "phi":  # tabulated psi for inverting psi (target t -> s)
            s = _psi_table_s(self._phi0)
            p = _call(self._psi_user, s)
            p = np.where(np.isfinite(p), p, 0.0)
            p[0] = 1.0
            # enforce monotonicity of the table (decreasing in s)
            p = np.minimum.accumulate(np.clip(p, 0.0, 1.0))
            tab = (s, p)
        else:  # tabulated phi for inverting phi (target s -> t)
            t = _phi_table_t()
            f = _call(self._phi_user, t)
            f[-1] = 0.0
            f = np.where(np.isnan(f), np.inf, f)
            f = np.maximum.accumulate(f[::-1])[::-1]  # decreasing in t
            tab = (t, f)
        self._tables[key] = tab
        return tab

    def _invert_psi(self, t):
        r""":math:`\varphi(t)=\inf\{s: \psi(s)\le t\}` for given :math:`\psi`."""
        t = np.asarray(t, dtype=float)
        s_tab, p_tab = self._table("phi")
        flat = t.ravel()
        out = np.full(flat.shape, np.nan)
        out[flat >= 1.0] = 0.0
        out[flat <= 0.0] = self._phi0
        inner = np.flatnonzero((flat > 0.0) & (flat < 1.0))
        if inner.size:
            tt = flat[inner]
            # p_tab decreasing: k = first index with p_tab[k] <= t
            k = np.searchsorted(-p_tab, -tt, side="left")
            beyond = k >= s_tab.size
            k = np.clip(k, 1, s_tab.size - 1)
            lo, hi = s_tab[k - 1], s_tab[k]
            psi = self._psi_user

            def g(x, idx):  # increasing in x: t - psi(x)
                return tt[idx] - _call(psi, x)

            res = solve_increasing(g, lo, hi, tt - p_tab[k - 1], tt - p_tab[k])
            res = np.where(beyond, np.inf if not math.isfinite(self._phi0) else self._phi0, res)
            out[inner] = res
        return out.reshape(t.shape)

    def _invert_phi(self, s):
        r""":math:`\psi(s)` by inverting the given :math:`\varphi`."""
        s = np.asarray(s, dtype=float)
        t_tab, f_tab = self._table("psi")
        flat = s.ravel()
        out = np.zeros(flat.shape)
        out[flat <= 0.0] = 1.0
        inner = np.flatnonzero((flat > 0.0) & (flat < self._phi0))
        if inner.size:
            ss = flat[inner]
            # f_tab decreasing in t: k = first index with f_tab[k] <= s
            k = np.searchsorted(-f_tab, -ss, side="left")
            k = np.clip(k, 1, t_tab.size - 1)
            lo, hi = t_tab[k - 1], t_tab[k]
            phi = self._phi_user

            def g(x, idx):  # increasing in t: s - phi(t)
                return ss[idx] - _call(phi, x)

            out[inner] = solve_increasing(g, lo, hi, ss - f_tab[k - 1], ss - f_tab[k])
        return out.reshape(s.shape)

    @property
    def generator(self) -> Callable:
        r"""Vectorized generator :math:`\varphi`."""
        return self._phi

    @property
    def inv_generator(self) -> Callable:
        r"""Vectorized pseudo-inverse :math:`\psi=\varphi^{[-1]}`."""
        return self._psi

    def _archimedean_generator(self) -> ArchimedeanGenerator:
        """Hook for :func:`copul.theory.archimedean.archimedean_generator`."""
        return ArchimedeanGenerator(
            phi=self._phi,
            dphi=self._dphi,
            psi=self._psi,
            phi0=self._phi0,
            d2phi=self._d2phi,
            source="numeric",
        )

    # -- copula ------------------------------------------------------------------------
    def _cdf(self, u, v):
        s = self._phi(u) + self._phi(v)
        c = self._psi(s)
        c = np.where((u <= 0) | (v <= 0), 0.0, c)
        return np.where(u >= 1, v, np.where(v >= 1, u, c))

    def _h(self, a, b):
        c = self.cdf_vectorized(a, b)
        with np.errstate(all="ignore"):
            h = self._dphi(np.maximum(a, _TINY)) / self._dphi(np.maximum(c, _TINY))
        h = np.where(c <= 0.0, 0.0, h)
        h = np.where(np.isfinite(h), h, np.where(b >= a, 1.0, 0.0))
        h = np.where(b >= 1.0, 1.0, np.where(b <= 0.0, 0.0, h))
        return np.clip(h, 0.0, 1.0)

    def _h1(self, u, v):
        return self._h(u, v)

    def _h2(self, u, v):
        return self._h(v, u)

    def _pdf(self, u, v):
        c = self.cdf_vectorized(u, v)
        cc = np.maximum(c, _TINY)
        with np.errstate(all="ignore"):
            d = (
                -self._d2phi(cc)
                * self._dphi(np.maximum(u, _TINY))
                * self._dphi(np.maximum(v, _TINY))
                / self._dphi(cc) ** 3
            )
        d = np.where(c <= 0.0, 0.0, d)
        return np.where(np.isfinite(d), d, 0.0)

    def _h_inv(self, x, w):
        r"""Conditional quantile :math:`v` of :math:`V\mid U=x` at level ``w``."""
        x, w = np.broadcast_arrays(np.asarray(x, float), np.asarray(w, float))
        shape = x.shape
        x = np.clip(x.ravel(), 0.0, 1.0)
        w = np.clip(w.ravel(), 0.0, 1.0)
        out = np.where(w >= 1.0, 1.0, 0.0)
        out = np.where((x >= 1.0) & (w < 1.0) & (w > 0.0), w, out)
        act = np.flatnonzero((w > 0.0) & (w < 1.0) & (x > 0.0) & (x < 1.0))
        if act.size:
            xa, wa = x[act], w[act]
            if self._psi_user is not None and self._phi_user is None:
                out[act] = self._h_inv_psi(xa, wa)
            else:
                out[act] = self._h_inv_phi(xa, wa)
        return np.clip(out, 0.0, 1.0).reshape(shape)

    def _h_inv_phi(self, xa, wa):
        r"""Solve :math:`\varphi'(c)=\varphi'(u)/w` for :math:`c=C(u,v)` (in :math:`\log c`)."""
        target = self._dphi(xa) / wa
        d0 = self._dphi0_value()
        on_curve = target <= d0  # no c > 0 solves it: quantile on the zero curve
        c = np.zeros(xa.size)
        sol = np.flatnonzero(~on_curve)
        if sol.size:
            tg = target[sol]
            dphi = self._dphi

            def g(y, idx):  # increasing in y = log(c) (phi convex)
                return _call(dphi, np.exp(y)) - tg[idx]

            lo = np.full(sol.size, math.log(_TINY))
            hi = np.log(xa[sol])
            lo_val = _call(dphi, np.exp(lo)) - tg
            lo_val = np.where(np.isfinite(lo_val), lo_val, -np.inf)
            hi_val = _call(dphi, xa[sol]) - tg
            c[sol] = np.exp(solve_increasing(g, lo, hi, lo_val, hi_val))
        end = self._phi0 if not self.is_strict else np.inf
        phic = np.where(c > 0.0, self._phi(c), end)
        return self._psi(np.maximum(phic - self._phi(xa), 0.0))

    def _h_inv_psi(self, xa, wa):
        r"""Solve :math:`\psi'(s)=w\,\psi'(\varphi(u))` for :math:`s=\varphi(u)+\varphi(v)`."""
        a = self._phi(xa)
        target = wa * self._dpsi(a)  # in (psi'(a), 0)
        if self.is_strict:
            s_hi = np.full(xa.size, 1e300)
            end = np.zeros(xa.size)
        else:
            s_hi = np.full(xa.size, self._phi0)
            end = np.full(
                xa.size, 1.0 / self._dphi0_value() if self._dphi0_value() > -np.inf else 0.0
            )
        s = np.full(xa.size, np.inf)
        sol = np.flatnonzero((target <= end) & (a > 0))
        if sol.size:
            tg = target[sol]
            dpsi = self._dpsi

            def g(y, idx):  # psi' nondecreasing (psi convex)
                val = _call(dpsi, np.exp(y)) - tg[idx]
                return np.where(np.isfinite(val), val, -tg[idx])

            lo = np.log(a[sol])
            hi = np.log(s_hi[sol])
            lo_val = _call(dpsi, a[sol]) - tg
            hi_val = end[sol] - tg
            s[sol] = np.exp(solve_increasing(g, lo, hi, np.minimum(lo_val, 0.0), hi_val))
        if not self.is_strict:
            s = np.where(np.isfinite(s), s, self._phi0)  # quantile on the zero curve
        return self._psi(np.maximum(s - a, 0.0))

    def _dphi0_value(self) -> float:
        r""":math:`\varphi'(0^+)` (``-inf`` if unbounded)."""
        if self.is_strict:
            return -math.inf
        t = np.array([1e-300, 1e-200, 1e-100, 1e-30])
        d = self._dphi(t)
        ok = np.isfinite(d)
        return float(d[np.flatnonzero(ok)[0]]) if np.any(ok) else -math.inf

    def _rvs(self, n, rng):
        u = rng.random(n)
        w = rng.random(n)
        return np.column_stack([u, self._h_inv(u, w)])

    def _numeric_callables(self):
        d = super()._numeric_callables()
        d["h1_inv"] = self._h_inv
        d["h2_inv"] = self._h_inv
        d["rvs"] = lambda n, rng: self._rvs(int(n), rng)
        return d

    @property
    def zero_curve_mass(self) -> float:
        r"""Singular mass :math:`-\varphi(0)/\varphi'(0^+)` on the zero curve."""
        if self._mass is None:
            self._mass = self._archimedean_generator().zero_curve_mass()
        return self._mass

    @property
    def is_absolutely_continuous(self) -> bool:
        return self.zero_curve_mass <= 1e-12

    @property
    def is_symmetric(self) -> bool:
        return True

    # -- measures --------------------------------------------------------------------
    def kendalls_tau(self, *args, **kwargs):
        r"""Kendall's :math:`\tau = 1 + 4\int_0^1 \varphi(t)/\varphi'(t)\,dt`.

        References
        ----------
        Nelsen (2006), Corollary 5.1.4.
        """
        from copul.measures.quadrature import integrate_1d

        gen = self._archimedean_generator()
        val, _ = integrate_1d(gen.phi_over_dphi, atol=1e-13, rtol=1e-11)
        return float(1.0 + 4.0 * val)

    def __repr__(self):
        if self._name:
            return f"NumericArchimedeanCopula({self._name})"
        return "NumericArchimedeanCopula()"

    __str__ = __repr__
