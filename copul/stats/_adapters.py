r"""
Vectorized numerical access to copula objects for :mod:`copul.stats`.

The functions below work with both the current copula API (scalar/SymPy
``cdf``/``pdf`` plus :func:`copul.measures.numeric_backend`) and a unified
vectorized ``cdf(u, v)`` / ``pdf(u, v)`` / ``logpdf(u, v)`` API on the copula
classes: a class method is used when it accepts arrays and returns an array of
the right shape, otherwise the numeric backend.

:class:`ParametricLogDensity` compiles the log-density of a family *with its
parameters as arguments* (one SymPy ``lambdify`` per family instead of one per
parameter value), which makes likelihood optimization fast; it falls back to
per-instance evaluation for families without a symbolic density.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import sympy as sp

from copul.measures.backend import free_parameters, numeric_backend

log = logging.getLogger(__name__)

__all__ = ["ParametricLogDensity", "cdf", "logpdf", "pdf"]

# per-class memo of the working route ("method" or "backend")
_ROUTE: dict[tuple[type, str], str] = {}

_TEST_U = np.array([0.13, 0.31, 0.52, 0.77, 0.9, 0.42, 0.66, 0.05, 0.95])
_TEST_V = np.array([0.71, 0.25, 0.47, 0.83, 0.12, 0.58, 0.36, 0.08, 0.9])


def _try_method(copula, name: str, u: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    fn = getattr(copula, name, None)
    if not callable(fn):
        return None
    try:
        with np.errstate(all="ignore"):
            r = fn(u, v)
        a = np.asarray(r, dtype=float)
    except Exception:
        return None
    if a.shape != u.shape:
        return None
    return a


def _vectorized(copula, name: str, u, v, backend_fn: Callable) -> np.ndarray:
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    key = (type(copula), name)
    route = _ROUTE.get(key)
    if route != "backend" and u.ndim >= 1 and u.size >= 2:
        a = _try_method(copula, name, u, v)
        if a is not None:
            _ROUTE[key] = "method"
            return a
        _ROUTE[key] = "backend"
    with np.errstate(all="ignore"):
        return np.asarray(backend_fn(u, v), dtype=float).reshape(u.shape)


def cdf(copula, u, v) -> np.ndarray:
    """Vectorized cdf :math:`C(u, v)` of a fully specified copula."""
    return _vectorized(copula, "cdf", u, v, lambda a, b: numeric_backend(copula).cdf(a, b))


def pdf(copula, u, v) -> np.ndarray:
    """Vectorized density :math:`c(u, v)` of a fully specified copula."""
    return _vectorized(copula, "pdf", u, v, lambda a, b: numeric_backend(copula).pdf(a, b))


def logpdf(copula, u, v) -> np.ndarray:
    r"""Vectorized log-density :math:`\log c(u, v)`.

    Tries ``copula.logpdf(u, v)``, then ``log(copula.pdf(u, v))``, then the
    numeric backend.  Zero densities give ``-inf``.
    """
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    key = (type(copula), "logpdf")
    if _ROUTE.get(key) != "backend" and u.size >= 2:
        a = _try_method(copula, "logpdf", u, v)
        if a is not None:
            _ROUTE[key] = "method"
            return a
        _ROUTE[key] = "backend"
    p = pdf(copula, u, v)
    with np.errstate(all="ignore"):
        out = np.log(p)
    return np.where(np.isnan(out), -np.inf, out)


# ---------------------------------------------------------------------------
# parametric log-density
# ---------------------------------------------------------------------------


def _symbol_map(copula) -> dict[str, sp.Symbol]:
    out = {}
    for p in list(getattr(type(copula), "params", None) or []) + list(
        getattr(copula, "params", None) or []
    ):
        if isinstance(p, sp.Symbol):
            out.setdefault(str(p), p)
    return out


def _symbolic_pdf(copula):
    """SymPy density expression of a (partially specified) family, its
    ``(u, v)`` symbols, or ``(None, None)``."""
    try:
        w = copula.pdf()
    except Exception:
        return None, None
    expr = getattr(w, "func", w)
    if not isinstance(expr, sp.Expr):
        return None, None
    u, v = getattr(copula, "u", None), getattr(copula, "v", None)
    if not isinstance(u, sp.Symbol) or not isinstance(v, sp.Symbol):
        return None, None
    return expr, (u, v)


class ParametricLogDensity:
    r"""Log-density :math:`\log c_\theta(u, v)` of a family as a function of
    its free parameters.

    The family's symbolic density (with the parameters as symbols) is
    compiled once.  Such closed forms are frequently valid only on part of
    the parameter space (e.g. the Clayton formula ignores the support
    restriction :math:`u^{-\theta}+v^{-\theta}>1` for :math:`\theta<0` and is
    a removable singularity at :math:`\theta=0`), so the expression is
    validated against per-instance evaluation (:func:`logpdf`) at reference
    parameters along each coordinate line of the parameter box, including
    zero-density points.  The *trusted region* is, per parameter, the union
    of the runs of consecutive successful references (ends refined by
    bisection, extended to the bound when a run reaches the end of the
    reference grid).  Inside it the compiled
    expression is used, elsewhere per-instance evaluation.

    Parameters
    ----------
    base : copula object
        Family instance whose free (symbolic) parameters are ``names``; the
        other parameters stay fixed.
    names : sequence of str
        Free parameters, in the order of the parameter vector.
    make : callable
        ``make(theta_vector) -> copula instance``.
    center : sequence of float
        Point of the parameter box through which the coordinate lines of
        reference values pass.
    grids : sequence of array_like
        Reference values for each parameter (increasing).
    bounds : sequence of (float, float)
        Parameter intervals.
    """

    def __init__(
        self,
        base,
        names: Sequence[str],
        make: Callable[[np.ndarray], Any],
        center: Sequence[float],
        grids: Sequence[Sequence[float]],
        bounds: Sequence[tuple[float, float]],
    ) -> None:
        self.names = list(names)
        self.make = make
        self.source = "instance"
        self.box: list[list[tuple[float, float]]] | None = None
        self._fast: Callable | None = None
        self._slow: Callable | None = None
        try:
            self._compile(
                base,
                np.asarray(center, dtype=float),
                [np.asarray(g, dtype=float) for g in grids],
                list(bounds),
            )
        except Exception as e:  # pragma: no cover - diagnostic
            log.debug("symbolic log-density failed for %s: %s", type(base).__name__, e)
            self._fast = self._slow = None
            self.box = None

    # -- compilation ------------------------------------------------------
    def _compile(self, base, center, grids, bounds) -> None:
        # a class-level vectorized logpdf (unified API) is preferred as is
        if _try_method(self.make(center), "logpdf", _TEST_U, _TEST_V) is not None:
            return
        expr, uv = _symbolic_pdf(base)
        if expr is None:
            return
        # match parameters by name (the expression's symbols may carry
        # assumptions that the class-level symbols do not)
        smap = {str(s): s for s in expr.free_symbols if s not in uv}
        smap.update({k: s for k, s in _symbol_map(base).items() if k not in smap})
        try:
            syms = [smap[n] for n in self.names]
        except KeyError:
            return
        if expr.free_symbols - set(uv) - set(syms):
            return
        from copul.measures.backend import _lambdify

        args = [*uv, *syms]
        slow = _lambdify(sp.log(expr), args)
        try:
            fast = _lambdify(sp.expand_log(sp.log(expr), force=True), args)
        except Exception:
            fast = None

        reference = _reference_density(base, self.names, self.make)
        box = []
        fast_ok = fast is not None
        for i, (grid, (lo, hi)) in enumerate(zip(grids, bounds)):

            def check(g, i=i):
                nonlocal fast_ok
                ref = center.copy()
                ref[i] = g
                target = reference(ref)
                if target is None:
                    return False
                good = _agrees(slow, ref, target)
                if good and fast_ok:
                    fast_ok = _agrees(fast, ref, target)
                return good

            status = [check(g) for g in grid]
            runs = _runs(status)
            if not runs:
                return
            ivs = []
            for a, b in runs:
                # refine the ends of each trusted interval by bisection
                if a == 0:
                    left = lo
                else:
                    bad_x, left = grid[a - 1], grid[a]
                    for _ in range(_BISECT):
                        mid = 0.5 * (bad_x + left)
                        if check(mid):
                            left = mid
                        else:
                            bad_x = mid
                if b == len(grid) - 1:
                    right = hi
                else:
                    right, bad_x = grid[b], grid[b + 1]
                    for _ in range(_BISECT):
                        mid = 0.5 * (right + bad_x)
                        if check(mid):
                            right = mid
                        else:
                            bad_x = mid
                ivs.append((float(left), float(right)))
            box.append(ivs)
        self._slow = slow
        self._fast = fast if fast_ok else None
        self.box = box
        self.source = "symbolic"

    # -- evaluation -------------------------------------------------------
    @property
    def has_fast(self) -> bool:
        """Whether a log-expanded (numerically preferable) expression is used."""
        return self._fast is not None

    def trusted(self, theta: Sequence[float]) -> bool:
        """Whether ``theta`` lies in the region where the compiled expression
        was validated (per parameter a union of intervals)."""
        if self._slow is None or self.box is None:
            return False
        return all(any(a <= t <= b for a, b in ivs) for t, ivs in zip(theta, self.box))

    def __call__(
        self, theta: Sequence[float], u: np.ndarray, v: np.ndarray, fast: bool = True
    ) -> np.ndarray:
        """``log c_theta(u, v)`` (``-inf`` where the density vanishes or is
        undefined); ``fast=False`` skips the log-expanded expression."""
        theta = np.asarray(theta, dtype=float)
        if not self.trusted(theta):
            return logpdf(self.make(theta), u, v)
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        with np.errstate(all="ignore"):
            out = None
            if fast and self._fast is not None:
                try:
                    out = np.broadcast_to(
                        np.asarray(self._fast(u, v, *theta), dtype=float), np.shape(u)
                    ).copy()
                except Exception:
                    out = None
            if out is None:
                out = np.full(np.shape(u), np.nan)
            bad = ~np.isfinite(out)
            if np.any(bad):
                sl = np.broadcast_to(
                    np.asarray(self._slow(u[bad], v[bad], *theta), dtype=float),
                    (int(bad.sum()),),
                )
                out[bad] = sl
        return np.where(np.isnan(out), -np.inf, out)


_BISECT = 12


class ParametricCDF:
    r"""The family's symbolic cdf :math:`C_\theta(u, v)` compiled once with
    the free parameters as arguments.

    It is validated against per-instance evaluation (:func:`cdf`) at the
    parameter vectors ``check_at``; :attr:`ok` is ``False`` if no symbolic cdf
    exists or validation fails.  Intended for many evaluations at nearby
    parameters (parametric bootstrap).
    """

    def __init__(self, base, names: Sequence[str], make: Callable, check_at) -> None:
        self.ok = False
        self._f = None
        try:
            w = base.cdf()
            expr = getattr(w, "func", w)
            uv = (base.u, base.v)
            if not isinstance(expr, sp.Expr):
                return
            smap = {str(s): s for s in expr.free_symbols if s not in uv}
            syms = [smap.get(n, sp.Symbol(n)) for n in names]
            if expr.free_symbols - set(uv) - set(syms):
                return
            from copul.measures.backend import _lambdify

            f = _lambdify(expr, [*uv, *syms])
            for theta in np.atleast_2d(np.asarray(check_at, dtype=float)):
                with np.errstate(all="ignore"):
                    val = np.broadcast_to(
                        np.asarray(f(_VAL_U, _VAL_V, *theta), dtype=float), _VAL_U.shape
                    )
                ref = cdf(make(theta), _VAL_U, _VAL_V)
                if not np.allclose(val, ref, rtol=1e-8, atol=1e-10):
                    return
            self._f = f
            self.ok = True
        except Exception as e:
            log.debug("parametric cdf failed for %s: %s", type(base).__name__, e)

    def __call__(self, theta: Sequence[float], u, v) -> np.ndarray:
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        with np.errstate(all="ignore"):
            c = np.broadcast_to(np.asarray(self._f(u, v, *theta), dtype=float), u.shape)
        return np.clip(c, np.maximum(u + v - 1.0, 0.0), np.minimum(u, v))


# validation points: interior, near all four corners and both diagonals
_VAL_U = np.concatenate([_TEST_U, [0.03, 0.97, 0.02, 0.98, 0.2, 0.6]])
_VAL_V = np.concatenate([_TEST_V, [0.97, 0.03, 0.02, 0.98, 0.1, 0.9]])


def _reference_density(base, names: Sequence[str], make: Callable):
    r"""Reference density at the validation points as a function of the
    parameter vector: mixed central differences
    :math:`\Delta_h^2 C / (4h^2)` (Richardson-extrapolated, :math:`h = 4\cdot10^{-5}`) of the family's
    symbolic cdf (compiled once), else per-instance :func:`pdf`."""
    fd = None
    try:
        w = base.cdf()
        cexpr = getattr(w, "func", w)
        uv = (base.u, base.v)
        if isinstance(cexpr, sp.Expr):
            smap = {str(s): s for s in cexpr.free_symbols if s not in uv}
            syms = [smap.get(n, sp.Symbol(n)) for n in names]
            if not cexpr.free_symbols - set(uv) - set(syms):
                from copul.measures.backend import _lambdify

                fd = _lambdify(cexpr, [*uv, *syms])
    except Exception as e:
        log.debug("no parametric cdf for %s: %s", type(base).__name__, e)
        fd = None
    h = 4e-5

    def reference(theta: np.ndarray) -> np.ndarray | None:
        if fd is not None:
            try:
                with np.errstate(all="ignore"):

                    def F(a, b):
                        return np.broadcast_to(
                            np.asarray(fd(a, b, *theta), dtype=float), _VAL_U.shape
                        )

                    def mixed(k):
                        return (
                            F(_VAL_U + k, _VAL_V + k)
                            - F(_VAL_U + k, _VAL_V - k)
                            - F(_VAL_U - k, _VAL_V + k)
                            + F(_VAL_U - k, _VAL_V - k)
                        ) / (4 * k * k)

                    # Richardson extrapolation of the central difference
                    d = (4.0 * mixed(0.5 * h) - mixed(h)) / 3.0
                if np.all(np.isfinite(d)):
                    return d
            except Exception:
                pass
            return None
        try:
            return pdf(make(theta), _VAL_U, _VAL_V)
        except Exception:
            return None

    return reference


def _agrees(f, ref: np.ndarray, target: np.ndarray) -> bool:
    """Compiled log-density ``f(u, v, *ref)`` agrees with the reference
    density ``target`` (finite differences, hence tolerance ``1e-4``),
    including zero-density points."""
    try:
        with np.errstate(all="ignore"):
            val = np.broadcast_to(np.asarray(f(_VAL_U, _VAL_V, *ref), dtype=float), _VAL_U.shape)
            dens = np.exp(val)
    except Exception:
        return False
    pos = target > 1e-6
    if not np.any(pos) or np.any(pos & ~np.isfinite(dens)):
        return False
    dens = np.where(np.isfinite(dens), dens, 0.0)
    return bool(np.all(np.abs(dens - target) <= 1e-4 * np.maximum(1.0, np.abs(target))))


def _runs(status: Sequence[bool]) -> list[tuple[int, int]]:
    """Maximal runs ``(first, last)`` of consecutive ``True`` entries."""
    out: list[tuple[int, int]] = []
    start = None
    for i, ok in enumerate(status):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(status) - 1))
    return out


def free_param_names(copula) -> list[str]:
    """Names of the free (symbolic) parameters of ``copula``."""
    return list(free_parameters(copula))


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def sample(copula, n: int, random_state: Any = None, method: str = "auto") -> np.ndarray:
    r"""Seeded sample of size ``n`` from a bivariate copula.

    Parameters
    ----------
    method : {"auto", "conditional", "rvs"}
        ``"conditional"``: conditional distribution method (Nelsen, 2006,
        Sec. 2.9): :math:`U, W` i.i.d. uniform and
        :math:`V = h_1^{-1}(W \mid U)` with :math:`h_1 = \partial_1 C`,
        inverted by vectorized bisection (60 steps, exact to double
        precision); ``"rvs"``: the copula's own ``rvs``; ``"auto"``: the
        conditional method when a closed-form/symbolic :math:`h_1` is
        available, else ``rvs``.

    Returns
    -------
    numpy.ndarray of shape (n, 2)
    """
    from copul.stats._utils import as_rng

    rng = as_rng(random_state)
    n = int(n)
    if method not in ("auto", "conditional", "rvs"):
        raise ValueError("method must be 'auto', 'conditional' or 'rvs'")
    h1 = None
    if method != "rvs":
        try:
            be = numeric_backend(copula)
            h1 = be.get("h1")
            if method == "auto" and be.source.get("h1") == "finite_differences":
                h1 = None
        except Exception as e:
            if method == "conditional":
                raise
            log.debug("no h1 for conditional sampling of %s: %s", type(copula).__name__, e)
            h1 = None
    if h1 is None:
        seed = int(rng.integers(0, 2**31 - 1))
        out = np.asarray(copula.rvs(n, random_state=seed), dtype=float)
        return out.reshape(n, 2)
    u = np.clip(rng.random(n), 1e-12, 1.0 - 1e-12)
    w = rng.random(n)
    lo = np.zeros(n)
    hi = np.ones(n)
    with np.errstate(all="ignore"):
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = np.asarray(h1(u, mid), dtype=float) < w
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
    return np.column_stack([u, 0.5 * (lo + hi)])
