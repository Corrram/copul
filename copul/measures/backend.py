r"""
Vectorized NumPy callables for copula objects.

:func:`numeric_backend` turns any bivariate copula object of :mod:`copul`
into a :class:`NumericBackend` exposing vectorized ``cdf(u, v)``,
``h1(u, v)`` (:math:`\partial_1 C`), ``h2(u, v)`` (:math:`\partial_2 C`) and
``pdf(u, v)`` functions.  Sources, in order of priority:

1. a class hook ``_numeric_callables()`` returning a dict with (some of) the
   keys ``"cdf"``, ``"h1"``, ``"h2"``, ``"pdf"`` (used e.g. by the Gaussian
   and Student-t copulas);
2. for bivariate extreme-value copulas, closed formulas in terms of the
   Pickands function :math:`A` and its derivatives;
3. a class-provided vectorized ``cdf_vectorized(u, v)`` or an array-accepting
   ``cdf``/``cond_distr`` (checkerboard convention ``cdf(points)`` with
   ``points`` of shape ``(n, 2)``), validated against scalar evaluation;
4. the SymPy CDF expression with all parameters substituted, lambdified via
   :func:`copul.numerics.to_numpy_callable`; partial derivatives and the
   density by ``sympy.diff`` followed by lambdify;
5. central finite differences of the CDF (last resort).

All functions are evaluated with ``np.errstate(all="ignore")``; non-finite
values are repaired: CDF values are clipped to the Fréchet--Hoeffding bounds
(a ``nan`` becomes the midpoint of the bounds, which only happens in
underflow corners where the bounds nearly coincide), h-function values are
clipped to :math:`[0,1]` and non-finite ones replaced by finite differences of
the CDF, densities are made non-negative.

The backend is cached on the copula instance (keyed by its parameter values).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import sympy as sp

from copul.measures.numeric import NumericCopula, _fd_partial
from copul.numerics import NUMPY_SAFE_MAP, drop_distributions

log = logging.getLogger(__name__)

__all__ = ["NumericBackend", "free_parameters", "numeric_backend"]

# fixed interior test points used for validating vectorized implementations
_TEST_U = np.array([0.13, 0.31, 0.52, 0.77, 0.9, 0.42, 0.66])
_TEST_V = np.array([0.71, 0.25, 0.47, 0.83, 0.12, 0.58, 0.36])


def _hyper(ap, bq, z):
    from scipy.special import hyp1f1, hyp2f1

    ap, bq = tuple(ap), tuple(bq)
    if len(ap) == 2 and len(bq) == 1:
        return hyp2f1(ap[0], ap[1], bq[0], z)
    if len(ap) == 1 and len(bq) == 1:
        return hyp1f1(ap[0], bq[0], z)
    raise NotImplementedError("hyper with these orders")


def _extra_modules():
    from scipy import special, stats

    return {
        **NUMPY_SAFE_MAP,
        "hyper": _hyper,
        "re": np.real,
        "im": np.imag,
        "LambertW": lambda x, k=0: np.real(special.lambertw(x, k)),
        "lowergamma": lambda a, x: special.gamma(a) * special.gammainc(a, x),
        "uppergamma": lambda a, x: special.gamma(a) * special.gammaincc(a, x),
        "polylog": _polylog,
        "Ei": special.expi,
        "stdtr": special.stdtr,
        "ndtr": special.ndtr,
        "norm_cdf": stats.norm.cdf,
    }


def _polylog(s, z):
    # only dilogarithm is needed (Frank-type expressions)
    from scipy.special import spence

    if s == 2:
        return spence(1 - np.asarray(z))
    raise NotImplementedError("polylog of order != 2")


def _lambdify(expr, syms):
    expr = drop_distributions(expr)
    f = sp.lambdify(tuple(syms), expr, modules=[_extra_modules(), "numpy"])
    return f


def _as_float_array(x, shape):
    a = np.asarray(x)
    if np.iscomplexobj(a):
        a = np.where(np.abs(a.imag) <= 1e-12 * np.maximum(1.0, np.abs(a.real)), a.real, np.nan)
    a = np.asarray(a, dtype=float)
    if a.shape != shape:
        a = np.broadcast_to(a, shape).copy()
    return a


def _to_float(x) -> float:
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    try:
        return float(x)
    except Exception:
        pass
    f = getattr(x, "func", None)
    if f is not None and f is not x:
        return _to_float(f)
    if isinstance(x, sp.Basic):
        v = complex(sp.N(x))
        if abs(v.imag) > 1e-12 * max(1.0, abs(v.real)):
            raise TypeError(f"complex value {v}")
        return v.real
    raise TypeError(f"cannot convert {type(x).__name__} to float")


def free_parameters(copula) -> list:
    """Names of the parameters of ``copula`` that are still symbolic."""
    names = []
    for p in list(getattr(copula, "params", None) or []):
        name = str(p)
        try:
            val = getattr(copula, name, p)
        except Exception:
            val = p
        if isinstance(val, sp.Basic) and not val.is_number:
            names.append(name)
    return names


def _param_key(copula):
    items = []
    cls_params = list(getattr(type(copula), "params", None) or [])
    inst_params = list(getattr(copula, "params", None) or [])
    for p in cls_params + inst_params:
        name = str(p)
        try:
            items.append((name, repr(getattr(copula, name, None))))
        except Exception:
            items.append((name, "?"))
    return (type(copula).__name__, tuple(items))


# ---------------------------------------------------------------------------
# scalar access helpers (validation only)
# ---------------------------------------------------------------------------


def _scalar_cdf(copula, u, v) -> float:
    errors = []
    for call in (
        lambda: copula.cdf(u=u, v=v),
        lambda: copula.cdf(u, v),
    ):
        try:
            return _to_float(call())
        except Exception as e:  # pragma: no cover - diagnostic
            errors.append(e)
    raise TypeError(f"scalar cdf evaluation failed: {errors}")


def _scalar_cond(copula, i, u, v) -> float:
    name = f"cond_distr_{i}"
    for call in (
        lambda: getattr(copula, name)(u=u, v=v),
        lambda: getattr(copula, name)(u, v),
        lambda: copula.cond_distr(i, u, v),
    ):
        try:
            return _to_float(call())
        except Exception:
            continue
    raise TypeError("scalar conditional distribution evaluation failed")


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class NumericBackend(NumericCopula):
    """Vectorized callables of a specific copula (see module docstring).

    The ingredients ``cdf``, ``h1``, ``h2`` and ``pdf`` are attributes that
    are built lazily on first access (``backend.h1(u, v)`` works directly).

    Attributes
    ----------
    source : dict
        Which mechanism produced each ingredient (e.g. ``{"cdf": "sympy"}``).
    prefer_h : bool
        ``True`` when h-functions are cheaper/more accurate than the CDF
        (the engine then uses the h-function formulas for rho and nu).
    breaks : tuple or None
        Known discontinuity lines ``(u_breaks, v_breaks)`` (checkerboards).
    """

    def __init__(self, copula_name: str = ""):
        self._f: dict[str, Callable | None] = {}
        self._builders: dict[str, Callable[[], Callable | None]] = {}
        self.copula_name = copula_name
        self.source: dict[str, str] = {}
        self.prefer_h = False
        super().__init__()

    def _built(self, what: str) -> bool:
        return self._f.get(what) is not None

    def get(self, what: str):
        f = self._f.get(what)
        if f is not None:
            return f
        builder = self._builders.pop(what, None)
        if builder is not None:
            try:
                f = builder()
            except Exception as e:
                log.debug("backend builder %s failed for %s: %s", what, self.copula_name, e)
                f = None
        if f is None:
            self.source[what] = "finite_differences"
            if what in ("h1", "h2"):
                f = _fd_partial(self.get("cdf"), 0 if what == "h1" else 1)
            elif what == "pdf":
                f = _fd_partial(self.get("h1"), 1, clip=None)
            else:
                raise ValueError(f"No numerical {what} available for {self.copula_name}.")
        self._f[what] = f
        return f

    def has_native_h(self) -> bool:
        """Whether ``h1`` comes from a closed/symbolic source (not finite differences)."""
        self.get("h1")
        return self.source.get("h1") != "finite_differences"

    def __repr__(self):  # pragma: no cover - cosmetic
        return f"NumericBackend({self.copula_name}, source={self.source})"


def _lazy(name):
    def fget(self):
        return (
            self.get(name) if (self._built(name) or name in self._builders) else self._f.get(name)
        )

    def fset(self, value):
        self._f[name] = value

    return property(fget, fset, doc=f"Vectorized ``{name}(u, v)`` (built lazily).")


for _name in ("cdf", "h1", "h2", "pdf"):
    setattr(NumericBackend, _name, _lazy(_name))


# ------------------------------- cleaning ----------------------------------


def _clean_cdf(raw: Callable) -> Callable:
    def cdf(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        with np.errstate(all="ignore"):
            c = _as_float_array(raw(u, v), u.shape)
        lo = np.maximum(u + v - 1.0, 0.0)
        hi = np.minimum(u, v)
        bad = ~np.isfinite(c)
        if np.any(bad):
            c = np.where(bad, 0.5 * (lo + hi), c)
        return np.clip(c, lo, hi)

    return cdf


def _clean_h(raw: Callable, fallback: Callable) -> Callable:
    def h(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        with np.errstate(all="ignore"):
            val = _as_float_array(raw(u, v), u.shape)
        bad = ~np.isfinite(val)
        if np.any(bad):
            val = val.copy()
            val[bad] = fallback(u[bad], v[bad])
        return np.clip(val, 0.0, 1.0)

    return h


def _clean_pdf(raw: Callable, fallback: Callable) -> Callable:
    def pdf(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        with np.errstate(all="ignore"):
            val = _as_float_array(raw(u, v), u.shape)
        bad = ~np.isfinite(val)
        if np.any(bad):
            val = val.copy()
            val[bad] = fallback(u[bad], v[bad])
        return np.maximum(val, 0.0)

    return pdf


def _close(a, b, rtol=1e-7, atol=1e-9):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return bool(np.all(np.isfinite(a)) and np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))


# ------------------------------ sources ------------------------------------


def _uv_symbols(expr):
    syms = {str(s): s for s in expr.free_symbols}
    extra = set(syms) - {"u", "v"}
    if extra:
        return None
    u = syms.get("u", sp.Symbol("u"))
    v = syms.get("v", sp.Symbol("v"))
    return u, v


def _sympy_cdf_expr(copula):
    """SymPy expression of C(u, v) (all parameters substituted) or None."""
    try:
        w = copula.cdf
        if callable(w) and not hasattr(w, "func"):
            w = w()
        expr = getattr(w, "func", w)
    except Exception:
        return None, None
    if not isinstance(expr, sp.Expr):
        return None, None
    if expr.has(sp.Integral, sp.Derivative, sp.Subs) or any(
        isinstance(a, sp.core.function.AppliedUndef) for a in expr.atoms(sp.Function)
    ):
        return None, None
    uv = _uv_symbols(expr)
    if uv is None:
        return None, None
    return expr, uv


def _vectorized_candidates(copula):
    """Yield (name, callable) candidates for a vectorized cdf."""
    if hasattr(copula, "cdf_vectorized"):
        yield "cdf_vectorized", lambda u, v: copula.cdf_vectorized(u, v)

    def points_call(u, v):
        pts = np.column_stack([np.ravel(u), np.ravel(v)])
        return np.asarray(copula.cdf(pts), float).reshape(np.shape(u))

    yield "cdf(points)", points_call


def _cond_candidates(copula, i):
    def points_call(u, v):
        pts = np.column_stack([np.ravel(u), np.ravel(v)])
        return np.asarray(copula.cond_distr(i, pts), float).reshape(np.shape(u))

    yield f"cond_distr({i}, points)", points_call

    def pair_call(u, v):
        out = getattr(copula, f"cond_distr_{i}")(np.ravel(u), np.ravel(v))
        return np.asarray(out, float).reshape(np.shape(u))

    yield f"cond_distr_{i}(u, v)", pair_call


_CORNER_EPS = np.array([1e-6, 1e-6, 1e-4])
_CORNER_U = np.concatenate([_CORNER_EPS, 1 - _CORNER_EPS])
_CORNER_V = np.concatenate([2 * _CORNER_EPS, 1 - 2 * _CORNER_EPS])


def _validate_corners(f, reference) -> bool:
    """Compare C/eps near (0,0) and the survival part near (1,1)."""
    if reference is None or not np.all(np.isfinite(reference)):
        return True
    try:
        with np.errstate(all="ignore"):
            val = _as_float_array(f(_CORNER_U.copy(), _CORNER_V.copy()), _CORNER_U.shape)
    except Exception:
        return False
    eps = np.concatenate([_CORNER_EPS, _CORNER_EPS])
    shift = np.concatenate([np.zeros(3), 1 - 3 * _CORNER_EPS])
    a = (val - shift) / eps
    b = (reference - shift) / eps
    return bool(np.all(np.isfinite(a)) and np.all(np.abs(a - b) <= 1e-5 + 1e-5 * np.abs(b)))


def _validate(f, reference, rtol=1e-7, atol=1e-9) -> bool:
    try:
        with np.errstate(all="ignore"):
            val = _as_float_array(f(_TEST_U.copy(), _TEST_V.copy()), _TEST_U.shape)
    except Exception as e:
        log.debug("candidate failed: %s", e)
        return False
    return _close(val, reference, rtol=rtol, atol=atol)


def _fd_dA(A):
    def dA(x):
        x = np.asarray(x, float)
        h = np.minimum(1e-6, 0.5 * np.minimum(x, 1 - x))
        h = np.where(h > 0, h, 1e-12)
        return (A(x + h) - A(x - h)) / (2 * h)

    return dA


def _fd_d2A(A):
    def d2A(x):
        x = np.asarray(x, float)
        h = np.minimum(1e-4, 0.5 * np.minimum(x, 1 - x))
        h = np.where(h > 0, h, 1e-12)
        return (A(x + h) - 2 * A(x) + A(x - h)) / h**2

    return d2A


def _pickands_numeric(copula):
    """Vectorized ``(A, A', A'')`` of a bivariate EV copula, or ``None``.

    ``A`` comes from a class hook ``_pickands_numpy()`` (vectorized A, whose
    derivatives are then taken by finite differences) or from lambdifying the
    SymPy Pickands expression and its derivatives.
    """
    tt = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
    hook = getattr(copula, "_pickands_numpy", None)
    if callable(hook):
        A = hook()
        return A, _fd_dA(A), _fd_d2A(A)
    pk = copula.pickands
    A_expr = getattr(pk, "func", pk)
    if not isinstance(A_expr, sp.Expr):
        return None
    t = None
    for s_ in A_expr.free_symbols:
        if str(s_) == "t":
            t = s_
        else:
            return None  # free parameters
    if t is None:
        t = sp.Symbol("t")
    # sympy's Piecewise((1, Eq(t, 1)), ...) guards are harmless
    A = _lambdify(A_expr, [t])
    with np.errstate(all="ignore"):
        a_test = _as_float_array(A(tt), tt.shape)
    if not np.all(np.isfinite(a_test)):
        return None
    expensive = A_expr.has(sp.hyper, sp.meijerg)
    dA = d2A = None
    if not expensive:
        try:
            dA = _lambdify(sp.diff(A_expr, t), [t])
            with np.errstate(all="ignore"):
                if not np.all(np.isfinite(_as_float_array(dA(tt), tt.shape))):
                    raise ValueError
        except Exception:
            dA = None
        try:
            d2A = _lambdify(sp.diff(A_expr, t, 2), [t])
            with np.errstate(all="ignore"):
                if not np.all(np.isfinite(_as_float_array(d2A(tt), tt.shape))):
                    raise ValueError
        except Exception:
            d2A = None
    return A, dA or _fd_dA(A), d2A or _fd_d2A(A)


def _ev_callables(copula):
    """cdf / h1 / h2 / pdf of a bivariate extreme-value copula from A(t)."""
    pn = _pickands_numeric(copula)
    if pn is None:
        return None
    A, dA, d2A = pn

    def parts(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        x = -np.log(u)
        y = -np.log(v)
        s = x + y
        tt_ = np.clip(y / s, 0.0, 1.0)
        a = _as_float_array(A(tt_), u.shape)
        c = np.exp(-s * a)
        return u, v, s, tt_, a, c

    def cdf(u, v):
        return parts(u, v)[-1]

    def h1(u, v):
        u, v, s, tt_, a, c = parts(u, v)
        da = _as_float_array(dA(tt_), u.shape)
        return c * (a - tt_ * da) / u

    def h2(u, v):
        u, v, s, tt_, a, c = parts(u, v)
        da = _as_float_array(dA(tt_), u.shape)
        return c * (a + (1 - tt_) * da) / v

    def pdf(u, v):
        u, v, s, tt_, a, c = parts(u, v)
        da = _as_float_array(dA(tt_), u.shape)
        d2a = _as_float_array(d2A(tt_), u.shape)
        return c / (u * v) * ((a - tt_ * da) * (a + (1 - tt_) * da) + d2a * tt_ * (1 - tt_) / s)

    return {"cdf": cdf, "h1": h1, "h2": h2, "pdf": pdf}


def _checkerboard_callables(copula):
    """O(1)-per-point callables for bivariate checkerboard copulas.

    With cell indices ``i = floor(m u)``, ``j = floor(n v)`` and local
    coordinates ``fu, fv`` in ``[0, 1)``::

        C = P[i, j] + fu R[i, j] + fv Q[i, j] + w_ij K(fu, fv)

    where ``P``, ``R``, ``Q`` are cumulative sums of the weights and
    ``K = fu fv`` (BivCheckPi), ``min(fu, fv)`` (BivCheckMin) or
    ``max(fu + fv - 1, 0)`` (BivCheckW).  Validated against the class' own
    scalar ``cdf`` before use.
    """
    try:
        from copul.checkerboard.biv_check_min import BivCheckMin
        from copul.checkerboard.biv_check_pi import BivCheckPi
        from copul.checkerboard.biv_check_w import BivCheckW
    except Exception:  # pragma: no cover
        return None
    if not isinstance(copula, BivCheckPi):
        return None
    if isinstance(copula, BivCheckMin):
        kind = "min"
    elif isinstance(copula, BivCheckW):
        kind = "w"
    elif type(copula).cdf is BivCheckPi.cdf or type(copula).__name__ == "BivCheckPi":
        kind = "pi"
    else:
        return None
    W = np.asarray(copula.matr, dtype=float)
    W = W / W.sum()
    m, n = W.shape
    P = np.zeros((m + 1, n + 1))
    P[1:, 1:] = W.cumsum(0).cumsum(1)
    R = np.zeros((m, n + 1))
    R[:, 1:] = W.cumsum(1)
    Q = np.zeros((m + 1, n))
    Q[1:, :] = W.cumsum(0)

    def loc(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        x, y = m * u, n * v
        i = np.clip(np.floor(x).astype(int), 0, m - 1)
        j = np.clip(np.floor(y).astype(int), 0, n - 1)
        return i, j, x - i, y - j

    def cdf(u, v):
        i, j, fu, fv = loc(u, v)
        w = W[i, j]
        if kind == "pi":
            k = fu * fv
        elif kind == "min":
            k = np.minimum(fu, fv)
        else:
            k = np.maximum(fu + fv - 1.0, 0.0)
        return P[i, j] + fu * R[i, j] + fv * Q[i, j] + w * k

    def h1(u, v):
        i, j, fu, fv = loc(u, v)
        w = W[i, j]
        if kind == "pi":
            dk = fv
        elif kind == "min":
            dk = (fu < fv).astype(float)
        else:
            dk = (fu + fv > 1.0).astype(float)
        return m * (R[i, j] + w * dk)

    def h2(u, v):
        i, j, fu, fv = loc(u, v)
        w = W[i, j]
        if kind == "pi":
            dk = fu
        elif kind == "min":
            dk = (fv < fu).astype(float)
        else:
            dk = (fu + fv > 1.0).astype(float)
        return n * (Q[i, j] + w * dk)

    def pdf(u, v):
        i, j, _, _ = loc(u, v)
        if kind == "pi":
            return m * n * W[i, j]
        return np.zeros(np.shape(i))

    try:
        ref = np.array([_scalar_cdf(copula, a, b) for a, b in zip(_TEST_U, _TEST_V)])
    except Exception:
        return None
    if not _close(cdf(_TEST_U, _TEST_V), ref, rtol=1e-9, atol=1e-12):
        return None
    breaks = (np.arange(1, m) / m, np.arange(1, n) / n)
    return {"cdf": cdf, "h1": h1, "h2": h2, "pdf": pdf, "breaks": breaks}


def _build(copula) -> NumericBackend:
    name = type(copula).__name__
    be = NumericBackend(name)
    provided: dict[str, Callable] = {}
    sources: dict[str, str] = {}

    # 1. class hook ------------------------------------------------------
    hook = getattr(copula, "_numeric_callables", None)
    if callable(hook):
        try:
            d = hook() or {}
            for k in ("cdf", "h1", "h2", "pdf"):
                if d.get(k) is not None:
                    provided[k] = d[k]
                    sources[k] = "class_hook"
            if d.get("prefer_h"):
                be.prefer_h = True
            if d.get("breaks") is not None:
                be.breaks = d["breaks"]
        except Exception as e:
            log.debug("_numeric_callables failed for %s: %s", name, e)

    # 1b. checkerboard copulas (piecewise closed forms) -------------------------
    if not provided:
        try:
            d = _checkerboard_callables(copula)
            if d:
                be.breaks = d.pop("breaks", None)
                for k, f in d.items():
                    provided[k] = f
                    sources[k] = "checkerboard"
        except Exception as e:
            log.debug("checkerboard callables failed for %s: %s", name, e)

    # 2. extreme-value copulas ----------------------------------------------
    if len(provided) < 4:
        try:
            from copul.family.extreme_value.biv_extreme_value_copula import (
                BivExtremeValueCopula,
            )

            if isinstance(copula, BivExtremeValueCopula):
                d = _ev_callables(copula)
                if d:
                    for k, f in d.items():
                        if k not in provided:
                            provided[k] = f
                            sources[k] = "pickands"
        except Exception as e:
            log.debug("EV callables failed for %s: %s", name, e)

    # 3./4. vectorized class methods and SymPy expression ------------------------
    expr, uv = (None, None)
    if "cdf" not in provided or "h1" not in provided:
        expr, uv = _sympy_cdf_expr(copula)
    sym_cdf = None
    if expr is not None:
        try:
            sym_cdf = _lambdify(expr, uv)
            with np.errstate(all="ignore"):
                ref = _as_float_array(sym_cdf(_TEST_U, _TEST_V), _TEST_U.shape)
            if not np.all(np.isfinite(ref)):
                raise ValueError("non-finite values at test points")
        except Exception as e:
            log.debug("lambdify of cdf failed for %s: %s", name, e)
            sym_cdf, expr = None, None

    if "cdf" not in provided:
        # reference values for validation
        if sym_cdf is not None:
            with np.errstate(all="ignore"):
                ref = _as_float_array(sym_cdf(_TEST_U, _TEST_V), _TEST_U.shape)
        else:
            try:
                ref = np.array([_scalar_cdf(copula, a, b) for a, b in zip(_TEST_U, _TEST_V)])
            except Exception:
                ref = None
        # corner points: class-provided implementations sometimes use
        # shortcuts near the boundary (e.g. returning min(u, v) once the
        # generator sum is tiny), which spoil tail coefficients
        corner_ref = None
        if sym_cdf is not None:
            with np.errstate(all="ignore"):
                corner_ref = _as_float_array(sym_cdf(_CORNER_U, _CORNER_V), _CORNER_U.shape)
        else:
            try:
                corner_ref = np.array(
                    [_scalar_cdf(copula, a, b) for a, b in zip(_CORNER_U, _CORNER_V)]
                )
            except Exception:
                corner_ref = None
        chosen = None
        if ref is not None:
            for cname, cand in _vectorized_candidates(copula):
                if _validate(cand, ref) and _validate_corners(cand, corner_ref):
                    chosen = (cname, cand)
                    break
        if chosen is not None:
            provided["cdf"], sources["cdf"] = chosen[1], chosen[0]
        elif sym_cdf is not None:
            provided["cdf"], sources["cdf"] = sym_cdf, "sympy"
        elif ref is not None:
            log.debug("using scalar cdf loop for %s (slow)", name)
            scal = np.vectorize(lambda a, b: _scalar_cdf(copula, float(a), float(b)))
            provided["cdf"], sources["cdf"] = scal, "scalar"
        else:
            raise TypeError(f"Cannot obtain a numerical CDF for {name}.")

    be.cdf = _clean_cdf(provided["cdf"])
    be.source["cdf"] = sources["cdf"]
    fd1 = _fd_partial(be.cdf, 0)
    fd2 = _fd_partial(be.cdf, 1)

    def make_h(i):
        key = f"h{i}"
        fd = fd1 if i == 1 else fd2

        def builder():
            if key in provided:
                be.source[key] = sources[key]
                return _clean_h(provided[key], fd)
            if expr is not None:
                try:
                    d = sp.diff(expr, uv[i - 1])
                    f = _lambdify(d, uv)
                    with np.errstate(all="ignore"):
                        val = _as_float_array(f(_TEST_U, _TEST_V), _TEST_U.shape)
                    if np.all(np.isfinite(val)) and _close(
                        val, fd(_TEST_U, _TEST_V), rtol=1e-4, atol=1e-5
                    ):
                        be.source[key] = "sympy"
                        return _clean_h(f, fd)
                except Exception as e:
                    log.debug("sympy derivative %s failed for %s: %s", key, name, e)
            ref = fd(_TEST_U, _TEST_V)
            for cname, cand in _cond_candidates(copula, i):
                if _validate(cand, ref, rtol=1e-4, atol=1e-5):
                    be.source[key] = cname
                    return _clean_h(cand, fd)
            return None

        return builder

    be._builders["h1"] = make_h(1)
    be._builders["h2"] = make_h(2)

    def pdf_builder():
        fd = _fd_partial(be.get("h1"), 1, clip=None)
        if "pdf" in provided:
            be.source["pdf"] = sources["pdf"]
            return _clean_pdf(provided["pdf"], fd)
        if expr is not None:
            try:
                d = sp.diff(expr, uv[0], uv[1])
                f = _lambdify(d, uv)
                with np.errstate(all="ignore"):
                    val = _as_float_array(f(_TEST_U, _TEST_V), _TEST_U.shape)
                if np.all(np.isfinite(val)):
                    be.source["pdf"] = "sympy"
                    return _clean_pdf(f, fd)
            except Exception as e:
                log.debug("sympy pdf failed for %s: %s", name, e)
        return None

    be._builders["pdf"] = pdf_builder
    return be


def special_numeric(copula, key, rtol, atol):
    r"""One-dimensional representations for specific families, or ``None``.

    * Archimedean, Kendall's tau: :math:`1 + 4\int_0^1 \varphi/\varphi'\,dt`;
    * extreme-value, Spearman's rho: :math:`12\int_0^1 (1+A)^{-2}dt - 3`;
    * extreme-value, Kendall's tau (integrated by parts, valid for kinked A):
      :math:`\int_0^1 A'(t)\,[t(1-t)A'(t) - (1-2t)A(t)]/A(t)^2\,dt`.

    Returns ``(value, error, source)``.
    """
    from copul.measures.quadrature import integrate_1d

    if key not in ("tau", "rho"):
        return None
    try:
        from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
        from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
    except Exception:  # pragma: no cover
        return None
    if isinstance(copula, BivArchimedeanCopula) and key == "tau":
        gen = getattr(copula.generator, "func", None)
        t = getattr(copula, "t", None)
        if not isinstance(gen, sp.Expr) or t is None or (gen.free_symbols - {t}):
            return None
        raw = gen.args[0][0] if isinstance(gen, sp.Piecewise) else gen
        ratio = raw / sp.diff(raw, t)
        try:
            simp = sp.simplify(ratio)
            if sp.count_ops(simp) <= sp.count_ops(ratio):
                ratio = simp
        except Exception:
            pass
        f = _lambdify(ratio, [t])
        tt = np.array([0.1, 0.5, 0.9])
        with np.errstate(all="ignore"):
            if not np.all(np.isfinite(_as_float_array(f(tt), tt.shape))):
                return None

        def g(x):
            with np.errstate(all="ignore"):
                return _as_float_array(f(x), np.shape(x))

        val, err = integrate_1d(g, rtol=rtol, atol=atol / 4)
        if not np.isfinite(val):
            return None
        return 1 + 4 * val, 4 * err, "archimedean_generator"
    if isinstance(copula, BivExtremeValueCopula):
        pn = _pickands_numeric(copula)
        if pn is None:
            return None
        A, dA, _ = pn
        if key == "rho":

            def g(x):
                with np.errstate(all="ignore"):
                    return (1.0 + _as_float_array(A(x), np.shape(x))) ** -2

            val, err = integrate_1d(g, rtol=rtol, atol=atol / 12)
            return 12 * val - 3, 12 * err, "pickands_1d"

        def g(x):
            with np.errstate(all="ignore"):
                a = _as_float_array(A(x), np.shape(x))
                da = _as_float_array(dA(x), np.shape(x))
                return da * (x * (1 - x) * da - (1 - 2 * x) * a) / a**2

        val, err = integrate_1d(g, rtol=rtol, atol=atol)
        if not np.isfinite(val):
            return None
        return val, err, "pickands_1d"
    return None


def numeric_backend(copula, refresh: bool = False) -> NumericBackend:
    """Return (and cache) the :class:`NumericBackend` of a fully specified copula.

    Raises
    ------
    ValueError
        If the copula still has free symbolic parameters.
    """
    free = free_parameters(copula)
    if free:
        raise ValueError(
            f"{type(copula).__name__} has free parameters {free}; numerical "
            "evaluation needs a fully specified copula."
        )
    key = _param_key(copula)
    cached = getattr(copula, "_copul_numeric_backend", None)
    if not refresh and cached is not None and cached[0] == key:
        return cached[1]
    be = _build(copula)
    try:
        object.__setattr__(copula, "_copul_numeric_backend", (key, be))
    except Exception:  # pragma: no cover - objects without __dict__
        pass
    return be
