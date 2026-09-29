r"""
Uniform, vectorized numerical evaluation API of bivariate copulas.

Every bivariate copula of :mod:`copul` exposes

* ``cdf(u, v)``, ``pdf(u, v)``, ``logpdf(u, v)``,
* ``cond_distr_1(u, v)`` :math:`=\partial_1 C(u,v)=P(V\le v\mid U=u)`,
  ``cond_distr_2(u, v)`` :math:`=\partial_2 C(u,v)=P(U\le u\mid V=v)`,
* ``cond_distr_1_inv(u, w)`` (the :math:`w`-quantile of :math:`V\mid U=u`),
  ``cond_distr_2_inv(v, w)`` (the :math:`w`-quantile of :math:`U\mid V=v`),
* ``survival_function(u, v)`` :math:`=P(U>u,V>v)=1-u-v+C(u,v)`,
* ``rvs(n, random_state=None)``.

Call conventions of the evaluation methods (``f`` any of the above except
``rvs``):

* ``f(u, v)`` with scalars returns a Python ``float``;
* ``f(u, v)`` with array-likes returns an ``ndarray`` of the broadcast shape;
* ``f(P)`` with ``P`` of shape ``(N, 2)`` returns an ``ndarray`` of length
  ``N`` (a single point ``f([u, v])`` returns a ``float``);
* ``f(u=..., v=...)`` is equivalent to ``f(u, v)``.

Arguments outside :math:`[0, 1]` are clipped (densities vanish outside the
unit square).  For copulas with a singular component ``pdf`` evaluates the
density of the absolutely continuous part; families without one raise
:class:`~copul.exceptions.PropertyUnavailableException` as before.  Parameter values may be passed
as keywords (``Clayton().cdf(P, theta=2)``); a copula with free parameters
raises a ``ValueError`` on array input.  Called **without** arguments (or
with symbolic / partial arguments such as ``cdf(v=0.5)``) ``cdf``, ``pdf``
and ``cond_distr_*`` keep their symbolic behaviour and return SymPy
wrappers.

The numerical work is delegated to :func:`copul.measures.backend.numeric_backend`
(class hooks, closed forms of families, Pickands representation of
extreme-value copulas, lambdified SymPy expressions).  The evaluation methods
of the families are wrapped once per class (see :func:`install_numeric_api`)
such that the numerical route never touches SymPy in the hot path.
"""

from __future__ import annotations

import contextlib
import functools
import inspect

import numpy as np
import sympy as sp

__all__ = [
    "NUMERIC_METHODS",
    "evaluate",
    "install_numeric_api",
    "numeric_bypass",
    "parse_numeric_call",
]

#: evaluation methods whose numerical calls are routed to the numeric backend
NUMERIC_METHODS = ("cdf", "pdf", "cond_distr_1", "cond_distr_2")

_NOT_NUMERIC = object()
_UNSET = object()


# ---------------------------------------------------------------------------
# bypass (used while the numeric backend itself calls the original methods)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def numeric_bypass(copula):
    """Route calls of the evaluation methods of ``copula`` to the originals.

    Used by :mod:`copul.measures.backend`, which builds the vectorized
    callables *from* the family implementations.
    """
    try:
        d = copula.__dict__
    except AttributeError:  # pragma: no cover - objects without __dict__
        yield
        return
    d["_copul_numeric_bypass"] = d.get("_copul_numeric_bypass", 0) + 1
    try:
        yield
    finally:
        d["_copul_numeric_bypass"] -= 1


def bypassed(copula, fn):
    """Wrap ``fn`` such that it runs inside :func:`numeric_bypass`."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with numeric_bypass(copula):
            return fn(*args, **kwargs)

    return wrapper


def _is_bypassed(copula) -> bool:
    try:
        return bool(copula.__dict__.get("_copul_numeric_bypass", 0))
    except AttributeError:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _numeric_array(x):
    """``x`` as a float ndarray if it is a plain numeric value, else ``None``."""
    if x is None or isinstance(x, (bool, np.bool_)):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return np.asarray(x, dtype=float)
    if isinstance(x, sp.Basic):
        return None
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False) if x.dtype.kind in "biuf" else None
    if isinstance(x, (list, tuple)):
        try:
            a = np.asarray(x)
        except Exception:
            return None
        return a.astype(float) if a.dtype.kind in "biuf" and a.size > 0 else None
    return None


def _param_names(copula) -> set[str]:
    names = {str(p) for p in (getattr(type(copula), "params", None) or [])}
    names |= {str(p) for p in (getattr(copula, "params", None) or [])}
    return names


def parse_numeric_call(copula, args, kwargs, first="u", second="v"):
    """Parse the arguments of a numerical evaluation call.

    Returns ``(x, y, scalar, params)`` -- broadcast float arrays, whether the
    result is a scalar, and parameter keywords -- or ``None`` if the call is
    not a (complete) numerical evaluation and should be handled
    symbolically.
    """
    kwargs = dict(kwargs)
    kx = kwargs.pop(first, None)
    ky = kwargs.pop(second, None)
    params = {}
    if kwargs:
        names = _param_names(copula)
        for k, val in kwargs.items():
            if k not in names or _numeric_array(val) is None or np.ndim(val) != 0:
                return None
            params[k] = float(val)
    args = tuple(a for a in args if a is not None)
    if kx is not None or ky is not None:
        if args or kx is None or ky is None:
            return None
        args = (kx, ky)
    if len(args) == 2:
        x = _numeric_array(args[0])
        y = _numeric_array(args[1])
        if x is None or y is None:
            return None
        scalar = x.ndim == 0 and y.ndim == 0
        try:
            x, y = np.broadcast_arrays(x, y)
        except ValueError as e:
            raise ValueError(f"Arguments cannot be broadcast together: {e}") from None
        return x, y, scalar, params
    if len(args) == 1:
        a = _numeric_array(args[0])
        if a is None or a.ndim not in (1, 2) or a.shape[-1] != 2:
            return None
        if a.ndim == 1:
            return a[0], a[1], True, params
        return a[:, 0], a[:, 1], False, params
    return None


def _finish(out, scalar):
    out = np.asarray(out, dtype=float)
    return float(out) if scalar else out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _free_parameters(copula):
    from copul.measures.backend import free_parameters

    return free_parameters(copula)


def _resolve_copula(copula, params, array_input: bool, what: str):
    """Substitute parameter keywords; ``None`` if still symbolic (scalar input)."""
    if params:
        copula = copula(**params)
    free = _free_parameters(copula)
    if free:
        if array_input:
            raise ValueError(
                f"{type(copula).__name__}.{what}: numerical evaluation needs a fully "
                f"specified copula, but the parameters {free} are free. Pass them "
                f"as keywords (e.g. {what}(u, v, {free[0]}=...)) or instantiate the "
                "copula with parameter values."
            )
        return None
    return copula


def is_absolutely_continuous(copula) -> bool:
    try:
        return bool(copula.is_absolutely_continuous)
    except Exception:
        return True


def _backend(copula):
    from copul.measures.backend import numeric_backend

    return numeric_backend(copula)


def _eval_backend(copula, what, x, y):
    """Evaluate ingredient ``what`` of the numeric backend with exact boundaries."""
    be = _backend(copula)
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    if what == "cdf":
        return be.cdf(x, y)
    if what in ("cond_distr_1", "cond_distr_2"):
        f = be.h1 if what == "cond_distr_1" else be.h2
        cond, other = (x, y) if what == "cond_distr_1" else (y, x)
        # evaluate at interior conditioning values, fix the trivial boundaries
        eps = 1e-12
        cond_c = np.clip(cond, eps, 1.0 - eps)
        if what == "cond_distr_1":
            val = f(cond_c, y)
        else:
            val = f(x, cond_c)
        val = np.where(other <= 0.0, 0.0, np.where(other >= 1.0, 1.0, val))
        return np.clip(val, 0.0, 1.0)
    if what == "pdf":
        return be.pdf(x, y)
    if what == "logpdf":
        return be.logpdf(x, y)
    raise KeyError(what)  # pragma: no cover


def evaluate(copula, what, args, kwargs, original=None):
    """Numerical evaluation of ``what`` or ``_NOT_NUMERIC``.

    ``original`` (the family implementation) is used for classes with a
    native numerical API and for non-absolutely-continuous densities.
    """
    parsed = parse_numeric_call(copula, args, kwargs)
    if parsed is None:
        return _NOT_NUMERIC
    x, y, scalar, params = parsed
    cop = _resolve_copula(copula, params, not scalar, what)
    if cop is None:
        return _NOT_NUMERIC
    native = getattr(type(cop), "_numeric_native", False)
    if what == "pdf" and cop is not copula and not native and not is_absolutely_continuous(cop):
        _ = cop.pdf  # families without a density raise PropertyUnavailableException
    if native and original is not None and what != "logpdf":
        xc = np.clip(x, 0.0, 1.0)
        yc = np.clip(y, 0.0, 1.0)
        if scalar:
            xc, yc = float(xc), float(yc)
        if cop is copula:
            out = original(cop, xc, yc)
        else:
            out = getattr(cop, what)(xc, yc)
        if what == "pdf":
            out = np.where(_outside(x, y), 0.0, out)
        return _finish(out, scalar)
    from copul.measures.backend import NumericUnavailableError

    try:
        with np.errstate(all="ignore"):
            out = _eval_backend(cop, what, x, y)
    except NumericUnavailableError:
        if cop is copula:
            return _NOT_NUMERIC  # let the family implementation decide
        raise
    if what == "pdf":
        out = np.where(_outside(x, y), 0.0, out)
    return _finish(out, scalar)


def _outside(x, y):
    """Points outside the closed unit square (where the density vanishes)."""
    return (x < 0.0) | (x > 1.0) | (y < 0.0) | (y > 1.0)


def evaluate_inverse(copula, which, args, kwargs):
    """``cond_distr_{which}_inv``: quantile of the conditional distribution."""
    first, second = ("u", "w") if which == 1 else ("v", "w")
    parsed = parse_numeric_call(copula, args, kwargs, first=first, second=second)
    if parsed is None:
        raise TypeError(f"cond_distr_{which}_inv expects numerical arguments ({first}, {second}).")
    x, w, scalar, params = parsed
    cop = _resolve_copula(copula, params, True, f"cond_distr_{which}_inv")
    be = _backend(cop)
    x = np.clip(x, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0)
    with np.errstate(all="ignore"):
        out = be.h1_inv(x, w) if which == 1 else be.h2_inv(x, w)
    return _finish(out, scalar)


def evaluate_survival(copula, args, kwargs):
    parsed = parse_numeric_call(copula, args, kwargs)
    if parsed is None:
        raise TypeError("survival_function expects numerical arguments (u, v).")
    x, y, scalar, params = parsed
    cop = _resolve_copula(copula, params, True, "survival_function")
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    c = evaluate(cop, "cdf", (x, y), {}, original=_original_of(cop, "cdf"))
    out = np.clip(1.0 - x - y + np.asarray(c, dtype=float), 0.0, None)
    return _finish(out, scalar)


def evaluate_logpdf(copula, args, kwargs):
    parsed = parse_numeric_call(copula, args, kwargs)
    if parsed is None:
        raise TypeError("logpdf expects numerical arguments (u, v).")
    x, y, scalar, params = parsed
    cop = _resolve_copula(copula, params, True, "logpdf")
    if getattr(type(cop), "_numeric_native", False):
        with np.errstate(divide="ignore"):
            out = np.log(np.asarray(cop.pdf(x, y), dtype=float))
        return _finish(out, scalar)
    if not is_absolutely_continuous(cop):
        _ = cop.pdf  # families without a density raise PropertyUnavailableException
    with np.errstate(all="ignore"):
        out = _eval_backend(cop, "logpdf", x, y)
    out = np.where(_outside(x, y), -np.inf, out)
    return _finish(out, scalar)


def sample(copula, n, random_state=None):
    """Draw ``n`` samples of a fully specified bivariate copula (see ``rvs``)."""
    from copul.checkerboard._biv_engine import resolve_rng

    cop = _resolve_copula(copula, {}, True, "rvs")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = resolve_rng(random_state)
    if n == 0:
        return np.empty((0, 2))
    with np.errstate(all="ignore"):
        return _backend(cop).rvs(n, rng)


def _original_of(copula, name):
    attr = inspect.getattr_static(type(copula), name, None)
    return getattr(attr, "__copul_original__", None)


# ---------------------------------------------------------------------------
# class installation
# ---------------------------------------------------------------------------


class _SymbolicProxy:
    """Lazy stand-in for the value of a property-style evaluation attribute.

    ``copula.cdf`` of families implementing ``cdf`` as a property returns this
    proxy: calling it with numerical arguments uses the numeric backend
    (without building the SymPy object); everything else (attribute access,
    arithmetic, ``isinstance`` checks, symbolic calls) is forwarded to the
    original property value, which is computed on first use.
    """

    __slots__ = ("_copula", "_getter", "_name", "_real")

    def __init__(self, copula, name, getter):
        object.__setattr__(self, "_copula", copula)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_getter", getter)
        object.__setattr__(self, "_real", _UNSET)

    def _resolve(self):
        real = object.__getattribute__(self, "_real")
        if real is _UNSET:
            real = self._getter(self._copula)
            object.__setattr__(self, "_real", real)
        return real

    def __call__(self, *args, **kwargs):
        if args or kwargs:
            res = evaluate(self._copula, self._name, args, kwargs, original=None)
            if res is not _NOT_NUMERIC:
                return res
            return _call_symbolic(
                self._copula, self._name, lambda: self._resolve()(*args, **kwargs), args, kwargs
            )
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    @property
    def __class__(self):
        return type(self._resolve())

    def __repr__(self):
        return repr(self._resolve())

    def __str__(self):
        return str(self._resolve())

    def __float__(self):
        return float(self._resolve())

    def __bool__(self):
        return bool(self._resolve())

    def __hash__(self):
        return hash(self._resolve())

    def __eq__(self, other):
        return self._resolve() == other

    def __ne__(self, other):
        return self._resolve() != other

    def __neg__(self):
        return -self._resolve()

    def __abs__(self):
        return abs(self._resolve())

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self):
        return len(self._resolve())

    def __getitem__(self, item):
        return self._resolve()[item]

    def _sympy_(self):
        real = self._resolve()
        return sp.sympify(getattr(real, "func", real))


def _binop(name):
    def op(self, other):
        return getattr(self._resolve(), name)(other)

    op.__name__ = name
    return op


for _op in (
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
):
    setattr(_SymbolicProxy, _op, _binop(_op))


def _call_symbolic(copula, name, call, args, kwargs):
    """Run the family implementation; explain failures of numerical calls
    on copulas with free parameters."""
    try:
        return call()
    except (TypeError, ValueError) as e:
        if parse_numeric_call(copula, args, kwargs) is not None:
            free = _free_parameters(copula)
            if free:
                raise ValueError(
                    f"{type(copula).__name__}.{name}: cannot evaluate numerically while the "
                    f"parameters {free} are free; pass them as keywords (e.g. "
                    f"{name}(u, v, {free[0]}=...)) or instantiate the copula with "
                    "parameter values."
                ) from e
        raise


def _make_method(name, func):
    @functools.wraps(func)
    def method(self, *args, **kwargs):
        if (args or kwargs) and not _is_bypassed(self):
            res = evaluate(self, name, args, kwargs, original=func)
            if res is not _NOT_NUMERIC:
                return res
            return _call_symbolic(self, name, lambda: func(self, *args, **kwargs), args, kwargs)
        return func(self, *args, **kwargs)

    method.__copul_numeric__ = True
    method.__copul_original__ = func
    return method


def _make_property(name, prop):
    getter = prop.fget if isinstance(prop, property) else prop.func
    cached = isinstance(prop, functools.cached_property)
    cache_key = f"_copul_cached_{name}"

    def real_getter(copula):
        if not cached:
            return getter(copula)
        from copul.measures.backend import _param_key

        key = _param_key(copula)
        hit = copula.__dict__.get(cache_key)
        if hit is not None and hit[0] == key:
            return hit[1]
        val = getter(copula)
        copula.__dict__[cache_key] = (key, val)
        return val

    def fget(self):
        if _is_bypassed(self):
            return real_getter(self)
        proxy = _SymbolicProxy(self, name, real_getter)
        if name == "pdf" and not is_absolutely_continuous(self):
            # eager evaluation keeps the family semantics for singular copulas
            # (e.g. PropertyUnavailableException on attribute access)
            proxy._resolve()
        return proxy

    fset = prop.fset if isinstance(prop, property) else None
    new = property(fget, fset, doc=getattr(prop, "__doc__", None))
    new.fget.__copul_numeric__ = True
    new.fget.__copul_original__ = None
    return new


def _is_wrapped(attr) -> bool:
    if isinstance(attr, property):
        return getattr(attr.fget, "__copul_numeric__", False)
    return getattr(attr, "__copul_numeric__", False)


def install_numeric_api(cls) -> None:
    """Wrap the evaluation methods resolved on ``cls`` (see module docstring)."""
    for name in NUMERIC_METHODS:
        resolved = None
        for klass in cls.__mro__:
            if name in klass.__dict__:
                resolved = klass.__dict__[name]
                break
        if resolved is None or _is_wrapped(resolved):
            continue
        if isinstance(resolved, (property, functools.cached_property)):
            new = _make_property(name, resolved)
        elif inspect.isfunction(resolved):
            new = _make_method(name, resolved)
        else:
            continue
        if isinstance(resolved, functools.cached_property):
            # cached_property needs __set_name__; our replacement is a plain property
            pass
        setattr(cls, name, new)
