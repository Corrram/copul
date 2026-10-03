r"""
Extreme-value copula theory and tail limits.

Functions for *any* bivariate copula object of :mod:`copul` (closed forms
for extreme-value copulas, numerical limits otherwise):

* **Pickands functions**: validity check (:func:`check_pickands`), the
  Pickands function / stable tail dependence function / extremal
  coefficient of an extreme-value copula or of the extreme-value attractor
  of any copula (:func:`pickands_function`, :func:`stable_tail_dependence`,
  :func:`extremal_coefficient`);
* **characterisation**: max-stability (:func:`max_stability_defect`,
  :func:`is_max_stable`, :func:`is_extreme_value`);
* **domains of attraction**: the extreme-value attractor
  :math:`C^*(u,v)=\lim_n C(u^{1/n},v^{1/n})^n` as a fully usable numerical
  extreme-value copula (:func:`ev_attractor`);
* **tail copulas** :math:`\Lambda_L,\Lambda_U` (:func:`tail_copula`);
* **estimation**: the Pickands and Capéraà--Fougères--Genest estimators of
  :math:`A` with endpoint corrections (:func:`pickands_estimator`);
* **constructions**: :func:`ev_copula_from_pickands` (numerical
  :math:`A`).

Notation
--------
:math:`C_A(u,v)=\exp(-\ell(-\log u,-\log v))`,
:math:`\ell(x,y)=(x+y)A(y/(x+y))`, extremal coefficient
:math:`\theta=\ell(1,1)=2A(1/2)\in[1,2]`, :math:`C_A(u,u)=u^{\theta}` and
:math:`\lambda_U=2-\theta`.

References
----------
Pickands, J. (1981). Multivariate extreme value distributions. *Bull. Int.
Statist. Inst.* 49, 859--878.
Capéraà, P., Fougères, A.-L. & Genest, C. (1997). A nonparametric
estimation procedure for bivariate extreme value copulas. *Biometrika* 84,
567--577.
Capéraà, P., Fougères, A.-L. & Genest, C. (2000). Bivariate distributions
with given extreme value attractor. *J. Multivariate Anal.* 72, 30--49.
Genest, C. & Segers, J. (2009). Rank-based inference for bivariate
extreme-value copulas. *Ann. Statist.* 37, 2990--3022.
Gudendorf, G. & Segers, J. (2010). Extreme-value copulas. In *Copula
Theory and Its Applications*, Lecture Notes in Statistics 198, 127--145.
Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Ch. 2 and 4.
Schmidt, R. & Stadtmüller, U. (2006). Non-parametric estimation of tail
dependence. *Scand. J. Statist.* 33, 307--335.
Demarta, S. & McNeil, A. J. (2005). The t copula and related copulas.
*Int. Statist. Rev.* 73, 111--129.
Genest, C. & Rivest, L.-P. (1989). A characterization of Gumbel's family
of extreme value distributions. *Statist. Probab. Lett.* 8, 207--211.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from copul.family.archimedean.numeric_archimedean import _call
from copul.family.extreme_value.numeric_extreme_value import NumericExtremeValueCopula

__all__ = [
    "PickandsReport",
    "check_pickands",
    "ev_attractor",
    "ev_copula_from_pickands",
    "extremal_coefficient",
    "is_extreme_value",
    "is_max_stable",
    "max_stability_defect",
    "pickands_estimator",
    "pickands_function",
    "stable_tail_dependence",
    "tail_copula",
]


def _finish(out, scalar: bool):
    out = np.asarray(out, dtype=float)
    return float(out.reshape(-1)[0]) if scalar else out


def _require_specified(C, what: str) -> None:
    from copul.measures.backend import free_parameters

    free = free_parameters(C)
    if free:
        raise ValueError(
            f"{what} needs a fully specified copula; {type(C).__name__} has free parameters {free}."
        )


def _backend(C):
    from copul.measures.backend import numeric_backend

    return numeric_backend(C)


# ---------------------------------------------------------------------------
# Pickands function of an extreme-value copula object
# ---------------------------------------------------------------------------


def _pickands_callables(C):
    r"""``(A, A', A'')`` (vectorised) of an extreme-value copula object, or ``None``.

    Recognises :class:`BivExtremeValueCopula` objects (class hook
    ``_pickands_numpy`` or the lambdified SymPy Pickands function), the
    independence copula (:math:`A\equiv1`) and :math:`M`
    (:math:`A=\max(t,1-t)`).
    """
    from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
    from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
    from copul.family.frechet.upper_frechet import UpperFrechet

    if isinstance(C, BivIndependenceCopula) or type(C).__name__ == "IndependenceCopula":
        one = np.ones_like
        return (
            lambda t: one(np.asarray(t, float)),
            lambda t: 0.0 * np.asarray(t, float),
            lambda t: 0.0 * np.asarray(t, float),
        )
    if isinstance(C, UpperFrechet):
        return (
            lambda t: np.maximum(np.asarray(t, float), 1.0 - np.asarray(t, float)),
            lambda t: np.where(np.asarray(t, float) < 0.5, -1.0, 1.0),
            lambda t: 0.0 * np.asarray(t, float),
        )
    if not isinstance(C, BivExtremeValueCopula):
        return None
    hook = getattr(C, "_pickands_numpy", None)
    if callable(hook):
        res = hook()
        if isinstance(res, tuple):
            A, dA, d2A = (*res, None, None)[:3]
        else:
            A, dA, d2A = res, None, None
    else:
        import sympy as sp

        from copul.numerics import to_numpy_callable

        pk = C.pickands
        expr = getattr(pk, "func", pk)
        if not isinstance(expr, sp.Expr):
            return None
        syms = list(expr.free_symbols)
        if len(syms) > 1:
            return None
        t = syms[0] if syms else sp.Symbol("t")
        A = to_numpy_callable(expr, [t])
        dA = d2A = None
        try:
            dA = to_numpy_callable(sp.diff(expr, t), [t])
            d2A = to_numpy_callable(sp.diff(expr, t, 2), [t])
        except Exception:  # pragma: no cover
            pass
        tt = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        if not np.all(np.isfinite(_call(A, tt))):
            return None
        for f in (dA, d2A):
            if f is not None and not np.all(np.isfinite(_call(f, tt))):
                dA = d2A = None

    def A_clean(t):
        t = np.clip(np.asarray(t, float), 0.0, 1.0)
        a = _call(A, t)
        a = np.where(
            (t <= 0) | (t >= 1) | ~np.isfinite(a), np.where((t <= 0) | (t >= 1), 1.0, a), a
        )
        return np.clip(a, np.maximum(t, 1.0 - t), 1.0)

    dA_f = (lambda t: _call(dA, t)) if dA is not None else None
    d2A_f = (lambda t: _call(d2A, t)) if d2A is not None else None
    return A_clean, dA_f, d2A_f


# ---------------------------------------------------------------------------
# Pickands check
# ---------------------------------------------------------------------------


@dataclass
class PickandsReport:
    r"""Result of :func:`check_pickands`.

    Attributes
    ----------
    valid : bool
        :math:`A` is a Pickands dependence function: convex with
        :math:`\max(t,1-t)\le A(t)\le1` (hence :math:`A(0)=A(1)=1`).
    endpoints_ok, bounds_ok, convex_ok : bool
        The individual conditions on the grid.
    max_bound_violation : float
        :math:`\max_t \max(\max(t,1-t)-A(t),\, A(t)-1, 0)`.
    max_convexity_violation : float
        Largest negative second difference (scaled by the step).
    extremal_coefficient : float
        :math:`\theta=2A(1/2)`.
    lambda_U : float
        :math:`2-2A(1/2)`.
    symmetric : bool
        :math:`A(t)=A(1-t)` (exchangeable copula).
    details : dict
        Grid size, locations of the worst violations.
    """

    valid: bool
    endpoints_ok: bool
    bounds_ok: bool
    convex_ok: bool
    max_bound_violation: float
    max_convexity_violation: float
    extremal_coefficient: float
    lambda_U: float
    symmetric: bool
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.valid)


def check_pickands(A, n: int = 2001, tol: float = 1e-10) -> PickandsReport:
    r"""Check that :math:`A` is a Pickands dependence function.

    :math:`C_A` is a copula iff :math:`A` is convex with
    :math:`\max(t,1-t)\le A(t)\le 1` on :math:`[0,1]` (Pickands 1981;
    Gudendorf & Segers 2010, Sec. 2).  Checked on the uniform grid of ``n``
    points: bounds (with tolerance ``tol``) and nonnegative second
    differences (:math:`\ge -\mathrm{tol}`).

    Parameters
    ----------
    A : callable or copula
        A vectorised Pickands function or an extreme-value copula object.
    n : int
        Grid size.
    tol : float
        Tolerance.

    Returns
    -------
    PickandsReport

    Examples
    --------
    >>> from copul.theory.extreme_value import check_pickands
    >>> check_pickands(lambda t: (t**3 + (1 - t) ** 3) ** (1 / 3)).valid   # Gumbel(3)
    True
    >>> import numpy as np
    >>> check_pickands(lambda t: 1 - 0.3 * np.sin(np.pi * t) ** 4).valid   # not convex
    False
    """
    if hasattr(A, "cdf"):
        _require_specified(A, "check_pickands")
        pc = _pickands_callables(A)
        if pc is None:
            raise TypeError(f"{type(A).__name__} is not an extreme-value copula object.")
        A = pc[0]
    t = np.linspace(0.0, 1.0, int(n))
    a = _call(A, t)
    finite = bool(np.all(np.isfinite(a)))
    a = np.where(np.isfinite(a), a, np.inf)
    lower = np.maximum(t, 1.0 - t)
    viol = np.maximum.reduce([lower - a, a - 1.0, np.zeros_like(a)])
    bvi = float(np.max(viol))
    endpoints_ok = bool(abs(a[0] - 1.0) <= tol and abs(a[-1] - 1.0) <= tol)
    bounds_ok = finite and bvi <= tol
    d2 = a[:-2] - 2.0 * a[1:-1] + a[2:]
    cvi = float(max(0.0, -np.min(d2))) if finite else math.inf
    convex_ok = finite and cvi <= tol
    ah = float(_call(A, np.array([0.5]))[0])
    sym = bool(np.allclose(a, a[::-1], atol=max(tol, 1e-12)))
    return PickandsReport(
        valid=bool(endpoints_ok and bounds_ok and convex_ok),
        endpoints_ok=endpoints_ok,
        bounds_ok=bool(bounds_ok),
        convex_ok=bool(convex_ok),
        max_bound_violation=bvi,
        max_convexity_violation=cvi,
        extremal_coefficient=2.0 * ah,
        lambda_U=2.0 - 2.0 * ah,
        symmetric=sym,
        details={
            "n": int(n),
            "worst_bound_at": float(t[int(np.argmax(viol))]),
            "worst_convexity_at": float(t[1 + int(np.argmin(d2))]) if finite else None,
        },
    )


# ---------------------------------------------------------------------------
# max-stability
# ---------------------------------------------------------------------------


def max_stability_defect(C, n: int = 12, exponents=(0.3, 2.0, 5.0)) -> float:
    r"""Maximal violation of max-stability on a grid.

    .. math::

       \max_{u,v,s}\bigl|C(u^s,v^s) - C(u,v)^s\bigr|

    over the interior grid :math:`\{i/(n+1)\}^2` and the exponents ``s``.
    Extreme-value copulas are exactly the max-stable copulas
    (:math:`C(u^s,v^s)=C(u,v)^s` for all :math:`s>0`; Gudendorf & Segers
    2010, Sec. 2).

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    n : int
        Grid points per axis.
    exponents : sequence of float
        Exponents :math:`s`.

    Returns
    -------
    float
    """
    _require_specified(C, "max_stability_defect")
    cdf = _backend(C).cdf
    g = np.arange(1, n + 1) / (n + 1.0)
    U, V = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
    base = np.asarray(cdf(U, V), float)
    worst = 0.0
    for s in exponents:
        lhs = np.asarray(cdf(U**s, V**s), float)
        worst = max(worst, float(np.max(np.abs(lhs - base**s))))
    return worst


def is_max_stable(C, tol: float = 1e-9, **kwargs) -> bool:
    r"""Whether :func:`max_stability_defect` is at most ``tol``."""
    return bool(max_stability_defect(C, **kwargs) <= tol)


def is_extreme_value(C, tol: float = 1e-9, **kwargs) -> bool:
    r"""Numerical test whether ``C`` is an extreme-value copula.

    A copula is an extreme-value copula iff it is max-stable (Gudendorf &
    Segers 2010, Sec. 2); see :func:`max_stability_defect`.  E.g. the
    Gumbel--Hougaard copula (also Archimedean), :math:`\Pi` and :math:`M` are
    extreme-value copulas, Clayton and Frank copulas are not.
    """
    return is_max_stable(C, tol=tol, **kwargs)


# ---------------------------------------------------------------------------
# tail copulas
# ---------------------------------------------------------------------------


def _extrapolate(rs: np.ndarray) -> tuple[float, float]:
    r"""Limit of a sequence :math:`r_k=r(4^{-k})` (Aitken/Richardson).

    The sequence is truncated where its increments grow (floating-point
    breakdown at tiny :math:`s`); then the Aitken :math:`\Delta^2` step with
    the empirically estimated rate is applied to the last reliable triple
    (exact for :math:`r(s)=\lambda+a s^\kappa`).
    """
    rs = np.asarray(rs, float)
    ok = np.isfinite(rs)
    if not np.any(ok):
        return math.nan, math.inf
    stop = np.flatnonzero(~ok)
    rs = rs[: stop[0]] if stop.size else rs
    if rs.size < 3:
        return float(rs[-1]), math.inf
    d = np.diff(rs)
    scale = max(1.0, float(np.max(np.abs(rs))))
    m = rs.size
    for i in range(1, d.size):
        if abs(d[i]) > 2.0 * abs(d[i - 1]) + 1e-13 * scale:
            m = i + 1
            break
    rs = rs[:m]
    if rs.size < 3:
        return float(rs[-1]), float(abs(d[0]))
    r0, r1, r2 = rs[-3], rs[-2], rs[-1]
    d1, d2 = r1 - r0, r2 - r1
    if d1 != 0 and abs(d2) < abs(d1):
        q = d2 / d1
        corr = d2 * q / (1.0 - q)
        return float(r2 + corr), float(abs(corr) + abs(d2) * abs(q))
    return float(r2), float(abs(d2) + abs(d1))


def _tail_limits(C, x, y, upper: bool):
    r"""Numerical :math:`\Lambda_L` or :math:`\Lambda_U` at points ``(x, y)`` (flat arrays)."""
    from copul.measures.quadrature import integrate_1d_batch

    h1 = _backend(C).h1
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = np.zeros(x.size)
    err = np.zeros(x.size)
    act = np.flatnonzero((x > 0) & (y > 0))
    if act.size == 0:
        return out, err
    k = np.arange(1, 16 if upper else 31, dtype=float)
    s = 4.0**-k
    # shrink s for large (x, y) so that s x, s y < 1 / 2
    xa, ya = x[act], y[act]
    big = np.maximum(np.maximum(xa, ya), 1.0)
    K = s.size
    sr = (s[None, :] / big[:, None]).ravel()
    xr = np.repeat(xa, K)
    yr = np.repeat(ya, K)

    if upper:

        def f(z, r):
            return xr[r] * (
                1.0 - np.asarray(h1(1.0 - sr[r] * xr[r] * z, 1.0 - sr[r] * yr[r]), float)
            )
    else:

        def f(z, r):
            return xr[r] * np.asarray(h1(sr[r] * xr[r] * z, sr[r] * yr[r]), float)

    vals, _ = integrate_1d_batch(f, np.zeros(sr.size), np.ones(sr.size), atol=1e-14, rtol=1e-11)
    vals = vals.reshape(act.size, K)
    for i, j in enumerate(act):
        lam, e = _extrapolate(vals[i])
        out[j] = np.clip(lam, 0.0, min(x[j], y[j]))
        err[j] = e
    return out, err


def tail_copula(C, x, y, lower: bool = True, *, return_error: bool = False):
    r"""Lower or upper tail copula of ``C``.

    .. math::

       \Lambda_L(x,y) = \lim_{s\downarrow0}\frac{C(sx,sy)}{s},\qquad
       \Lambda_U(x,y) = \lim_{s\downarrow0}\frac{\hat C(sx,sy)}{s},

    with the survival copula :math:`\hat C(a,b)=a+b-1+C(1-a,1-b)`; they are
    homogeneous of order one, :math:`\Lambda(1,1)=\lambda` is the tail
    dependence coefficient and :math:`\Lambda_U(x,y)=x+y-\ell(x,y)` with
    the stable tail dependence function of the extreme-value attractor
    (Schmidt & Stadtmüller 2006; Joe 2014, Ch. 2).

    Numerically, :math:`C(sx,sy)/s=x\int_0^1\partial_1C(sxz,sy)\,dz` and
    :math:`\hat C(sx,sy)/s=x\int_0^1(1-\partial_1C(1-sxz,1-sy))\,dz` are
    computed without cancellation for :math:`s=4^{-k}` and extrapolated to
    :math:`s=0` (Aitken/Richardson).  For extreme-value copulas
    :math:`\Lambda_U` is evaluated exactly from :math:`A`.

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    x, y : float or array_like
        Points in :math:`[0,\infty)^2` (broadcast).
    lower : bool
        :math:`\Lambda_L` (``True``) or :math:`\Lambda_U` (``False``).
    return_error : bool
        Also return the extrapolation error estimates.

    Returns
    -------
    float or numpy.ndarray (and the error estimates)
    """
    _require_specified(C, "tail_copula")
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    shape = x.shape
    xf, yf = x.ravel(), y.ravel()
    pc = None if lower else _pickands_callables(C)
    if pc is not None:
        s = xf + yf
        with np.errstate(all="ignore"):
            t = np.where(s > 0, yf / s, 0.5)
        val = np.where(s > 0, s - s * pc[0](t), 0.0)
        err = np.zeros_like(val)
    else:
        val, err = _tail_limits(C, xf, yf, upper=not lower)
    val = val.reshape(shape)
    if return_error:
        return _finish(val, scalar), _finish(err.reshape(shape), scalar)
    return _finish(val, scalar)


# ---------------------------------------------------------------------------
# stable tail dependence, Pickands function and extremal coefficient
# ---------------------------------------------------------------------------


def stable_tail_dependence(C, x, y):
    r"""Stable tail dependence function :math:`\ell(x,y)`.

    For an extreme-value copula :math:`\ell(x,y)=(x+y)A(y/(x+y))`; for any
    copula in the max-domain of attraction of an extreme-value copula
    :math:`C^*`

    .. math::

       \ell(x,y)=\lim_{s\downarrow0}\frac{1-C(1-sx,1-sy)}{s}
               = x + y - \Lambda_U(x,y),

    the stable tail dependence function of :math:`C^*` (Gudendorf & Segers
    2010, Sec. 2; Joe 2014, Ch. 2), evaluated through
    :func:`tail_copula`.

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    x, y : float or array_like
        Points in :math:`[0,\infty)^2`.

    Returns
    -------
    float or numpy.ndarray
    """
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    lam = np.asarray(tail_copula(C, x, y, lower=False), float)
    return _finish(x + y - lam, scalar)


def pickands_function(C, t):
    r"""Pickands function :math:`A(t)` of ``C`` or of its extreme-value attractor.

    Exact for extreme-value copula objects; otherwise
    :math:`A^*(t)=\ell(1-t,t)=1-\Lambda_U(1-t,t)` of the attractor
    :math:`C^*(u,v)=\lim_n C(u^{1/n},v^{1/n})^n` (see :func:`ev_attractor`).

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    t : float or array_like
        Points in :math:`[0,1]`.

    Returns
    -------
    float or numpy.ndarray
    """
    _require_specified(C, "pickands_function")
    scalar = np.ndim(t) == 0
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    pc = _pickands_callables(C)
    if pc is not None:
        return _finish(pc[0](t), scalar)
    lam = np.asarray(tail_copula(C, 1.0 - t, t, lower=False), float)
    a = np.clip(1.0 - lam, np.maximum(t, 1.0 - t), 1.0)
    return _finish(a, scalar)


def extremal_coefficient(C) -> float:
    r"""Extremal coefficient :math:`\theta=\ell(1,1)=2A(1/2)\in[1,2]`.

    For an extreme-value copula :math:`C(u,u)=u^\theta`, i.e.
    :math:`\theta=\log C(u,u)/\log u` for every :math:`u\in(0,1)`; in
    general the extremal coefficient of the attractor,
    :math:`\theta=2-\lambda_U` (Gudendorf & Segers 2010, Sec. 2;
    Joe 2014, Ch. 2).  :math:`\theta=1` for :math:`M`, :math:`2` for
    independence.

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.

    Returns
    -------
    float
    """
    return float(2.0 * pickands_function(C, 0.5))


# ---------------------------------------------------------------------------
# constructions
# ---------------------------------------------------------------------------


def ev_copula_from_pickands(
    A: Callable,
    dA: Callable | None = None,
    d2A: Callable | None = None,
    *,
    check: bool = True,
    tol: float = 1e-9,
    name: str | None = None,
    absolutely_continuous: bool | None = None,
) -> NumericExtremeValueCopula:
    r"""Extreme-value copula :math:`C_A` from a numerical Pickands function.

    Parameters
    ----------
    A : callable
        Vectorised Pickands dependence function.
    dA, d2A : callable, optional
        Its derivatives (finite differences otherwise).  For kinked
        :math:`A` (singular components) pass the one-sided derivative
        ``dA`` for accurate measures.
    check : bool
        Raise ``ValueError`` unless :func:`check_pickands` passes.
    tol : float
        Tolerance of the check.
    name : str, optional
        Name used in ``repr``.
    absolutely_continuous : bool, optional
        See :class:`NumericExtremeValueCopula`.

    Returns
    -------
    NumericExtremeValueCopula
        Fully usable: ``cdf``, ``pdf``, conditional distributions and their
        inverses, ``rvs``, every measure (:math:`\rho,\tau` by the
        one-dimensional Pickands formulas), the theory functions.

    Examples
    --------
    >>> from copul.theory.extreme_value import ev_copula_from_pickands
    >>> C = ev_copula_from_pickands(lambda t: 1 - 0.5 * t * (1 - t))  # Tawn's mixed model
    >>> round(C.lambda_U(), 12)
    0.25
    """
    if check:
        rep = check_pickands(A, tol=tol)
        if not rep.valid:
            raise ValueError(f"not a Pickands dependence function: {rep}")
    return NumericExtremeValueCopula(
        A, dA, d2A, name=name, absolutely_continuous=absolutely_continuous
    )


def ev_attractor(C, deg: int = 48, *, independence_tol: float = 1e-7, check: bool = True):
    r"""Extreme-value attractor of a copula as a numerical extreme-value copula.

    .. math::

       C^*(u,v)=\lim_{n\to\infty}C(u^{1/n},v^{1/n})^n
              =\exp\bigl(-\ell(-\log u,-\log v)\bigr),\qquad
       \ell(x,y)=\lim_{s\downarrow0}\frac{1-C(1-sx,1-sy)}{s},

    whenever the limit exists (Gudendorf & Segers 2010, Sec. 2; Joe 2014,
    Ch. 2).  :math:`A^*(t)=\ell(1-t,t)` is computed by
    :func:`pickands_function` at Chebyshev points; the representation
    :math:`A^*(t)=1+t(1-t)\,D(t)` with a Chebyshev interpolant :math:`D` of
    degree ``deg`` gives :math:`A^*(0)=A^*(1)=1` exactly and smooth
    derivatives.  Extreme-value copulas are their own attractors; copulas
    without upper tail dependence in the sense
    :math:`\max_t|A^*(t)-1|\le` ``independence_tol`` are attracted by the
    independence copula, which is returned exactly.

    Known attractors: Gumbel--Hougaard (itself); Clayton, Frank, Gaussian
    with :math:`\rho<1` (independence); survival Clayton (Galambos); the
    Student-t copula (the t-EV copula, Demarta & McNeil 2005); Archimedean
    copulas whose generator is regularly varying at 1,
    :math:`\varphi(1-s)\sim s^{m}L(s)`, e.g. Joe and BB1 (Gumbel with
    parameter :math:`m`; Genest & Rivest 1989; Capéraà, Fougères & Genest
    2000).

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    deg : int
        Degree of the Chebyshev interpolant.
    independence_tol : float
        Snap to the independence copula below this deviation.
    check : bool
        Warn if the interpolated :math:`A^*` violates the Pickands
        conditions by more than :math:`10^{-6}` (e.g. a slowly converging
        tail limit).

    Returns
    -------
    NumericExtremeValueCopula
        Its ``pickands_values`` attribute holds the computed nodes and
        values, ``pickands_report`` the result of :func:`check_pickands`.
    """
    _require_specified(C, "ev_attractor")
    pc = _pickands_callables(C)
    if pc is not None:
        A, dA, d2A = pc
        out = NumericExtremeValueCopula(A, dA, d2A, name=f"attractor of {type(C).__name__}")
        out.pickands_values = None
        out.pickands_report = check_pickands(A, tol=1e-8)
        return out
    from numpy.polynomial import Chebyshev

    store = {}

    def D(t):
        t = np.asarray(t, float)
        a = np.asarray(pickands_function(C, t), float)
        store["t"], store["A"] = t, a
        return (a - 1.0) / (t * (1.0 - t))

    p = Chebyshev.interpolate(D, int(deg), domain=[0.0, 1.0])
    if np.max(np.abs(store["A"] - 1.0)) <= independence_tol:
        out = NumericExtremeValueCopula(
            lambda t: np.ones_like(np.asarray(t, float)),
            lambda t: 0.0 * np.asarray(t, float),
            lambda t: 0.0 * np.asarray(t, float),
            name=f"attractor of {type(C).__name__}: independence",
            absolutely_continuous=True,
        )
    else:
        dp, d2p = p.deriv(1), p.deriv(2)

        def A(t):
            t = np.asarray(t, float)
            return 1.0 + t * (1.0 - t) * p(t)

        def dA(t):
            t = np.asarray(t, float)
            return (1.0 - 2.0 * t) * p(t) + t * (1.0 - t) * dp(t)

        def d2A(t):
            t = np.asarray(t, float)
            return np.maximum(
                -2.0 * p(t) + 2.0 * (1.0 - 2.0 * t) * dp(t) + t * (1.0 - t) * d2p(t), 0.0
            )

        out = NumericExtremeValueCopula(A, dA, d2A, name=f"attractor of {type(C).__name__}")
    out.pickands_values = (store["t"], store["A"])
    out.pickands_report = check_pickands(out.pickands.value, tol=1e-6)
    if check:
        raw = check_pickands(lambda t: 1.0 + t * (1.0 - t) * p(t), tol=1e-6)
        if not raw.valid:
            warnings.warn(
                "the computed attractor violates the Pickands conditions "
                f"(bounds {raw.max_bound_violation:.2e}, convexity "
                f"{raw.max_convexity_violation:.2e}); the tail limit may converge slowly.",
                stacklevel=2,
            )
    return out


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------


def _greatest_convex_minorant(t: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Lower convex hull of the points (t_i, a_i), evaluated at t (t sorted)."""
    hull: list[int] = []
    for i in range(t.size):
        while len(hull) >= 2:
            i0, i1 = hull[-2], hull[-1]
            cross = (t[i1] - t[i0]) * (a[i] - a[i0]) - (a[i1] - a[i0]) * (t[i] - t[i0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(t, t[hull], a[hull])


def pickands_estimator(
    data,
    t=None,
    *,
    method: str = "cfg",
    endpoint_correction: bool = True,
    pseudo_obs: bool = True,
    convexify: bool = False,
):
    r"""Nonparametric estimators of the Pickands dependence function.

    With (pseudo-)observations :math:`(U_i,V_i)`, :math:`S_i=-\log U_i`,
    :math:`T_i=-\log V_i` and

    .. math::

       \xi_i(t)=\min\Bigl(\frac{S_i}{1-t},\frac{T_i}{t}\Bigr)
       \sim \mathrm{Exp}(A(t))\quad\text{under } C_A,

    * **Pickands** (1981): :math:`1/\hat A_P(t)=\frac1n\sum_i\xi_i(t)`; with
      endpoint correction (Deheuvels 1991; Genest & Segers 2009)
      :math:`1/\hat A(t)=1/\hat A_P(t)-(1-t)(1/\hat A_P(0)-1)-t(1/\hat A_P(1)-1)`;
    * **Capéraà--Fougères--Genest** (1997):
      :math:`\log\hat A_{CFG}(t)=-\frac1n\sum_i\log\xi_i(t)-\gamma`
      (:math:`\gamma` Euler's constant); with endpoint correction
      :math:`\log\hat A(t)=-\frac1n\sum_i\log\xi_i(t)
      +\frac{1-t}{n}\sum_i\log\xi_i(0)+\frac{t}{n}\sum_i\log\xi_i(1)`,
      the rank-based estimator of Genest & Segers (2009).

    Both corrected estimators satisfy :math:`\hat A(0)=\hat A(1)=1`; the
    CFG estimator is usually preferable (smaller asymptotic variance).

    Parameters
    ----------
    data : array_like of shape (n, 2)
        Observations (any margins if ``pseudo_obs``; uniform otherwise).
    t : float or array_like, optional
        Evaluation points (default: 101 equidistant points).
    method : {"cfg", "pickands"}
        Estimator.
    endpoint_correction : bool
        Apply the endpoint corrections above.
    pseudo_obs : bool
        Replace the data by their normalised ranks :math:`R_i/(n+1)`.
    convexify : bool
        Project the estimate onto the Pickands functions by clipping to
        :math:`[\max(t,1-t),1]` and taking the greatest convex minorant on
        the grid (cf. Hall & Tajvidi 2000; Fils-Villetard, Guillou & Segers
        2008) -- requires an increasing grid ``t`` containing 0 and 1.

    Returns
    -------
    float or numpy.ndarray

    References
    ----------
    Pickands (1981); Deheuvels, P. (1991), On the limiting behavior of the
    Pickands estimator for bivariate extreme-value distributions, *Statist.
    Probab. Lett.* 12, 429--439; Capéraà, Fougères & Genest (1997); Genest &
    Segers (2009); Hall, P. & Tajvidi, N. (2000), Distribution and
    dependence-function estimation for bivariate extreme-value
    distributions, *Bernoulli* 6, 835--844.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("data must have shape (n, 2)")
    if pseudo_obs:
        from scipy.stats import rankdata

        n = x.shape[0]
        x = np.column_stack([rankdata(x[:, 0]), rankdata(x[:, 1])]) / (n + 1.0)
    if np.any((x <= 0) | (x >= 1)):
        raise ValueError("uniform observations must lie in (0, 1)")
    scalar = t is not None and np.ndim(t) == 0
    tt = np.linspace(0.0, 1.0, 101) if t is None else np.atleast_1d(np.asarray(t, float))
    S = -np.log(x[:, 0])
    T = -np.log(x[:, 1])
    method = str(method).lower()
    if method not in ("cfg", "pickands"):
        raise ValueError(f"unknown method {method!r}")

    def xi(t):
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(t < 1.0, S[:, None] / (1.0 - t[None, :]), np.inf)
            b = np.where(t > 0.0, T[:, None] / t[None, :], np.inf)
        return np.minimum(a, b)

    ends = np.array([0.0, 1.0])
    out = np.empty(tt.size)
    for lo in range(0, tt.size, 256):
        chunk = tt[lo : lo + 256]
        xc = xi(chunk)
        if method == "pickands":
            inv = np.mean(xc, axis=0)
            if endpoint_correction:
                inv0, inv1 = np.mean(xi(ends), axis=0)
                inv = inv - (1.0 - chunk) * (inv0 - 1.0) - chunk * (inv1 - 1.0)
            out[lo : lo + 256] = 1.0 / inv
        else:
            m = np.mean(np.log(xc), axis=0)
            if endpoint_correction:
                m0, m1 = np.mean(np.log(xi(ends)), axis=0)
                out[lo : lo + 256] = np.exp(-m + (1.0 - chunk) * m0 + chunk * m1)
            else:
                out[lo : lo + 256] = np.exp(-m - np.euler_gamma)
    if convexify:
        out = np.clip(out, np.maximum(tt, 1.0 - tt), 1.0)
        if tt.size >= 3 and np.all(np.diff(tt) > 0):
            out = _greatest_convex_minorant(tt, out)
    return float(out[0]) if scalar else out
