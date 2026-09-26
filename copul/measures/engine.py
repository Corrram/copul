r"""
Front-end and method dispatch for dependence measures.

Dispatch semantics (used by every measure method of a bivariate copula, e.g.
``copula.spearmans_rho(method=...)``, and by :func:`compute`):

* **free symbolic parameters** -- the family's (or the base class')
  symbolic implementation is called and its SymPy result returned
  unchanged (``method`` ``"numeric"``/``"mc"`` raise :class:`ValueError`);
* **fully specified copula**

  - ``"auto"`` (default): a *closed form* (a subclass override that is not
    marked as symbolic-only) is tried first; if it raises, returns ``nan``,
    a non-number or an unevaluated ``sympy.Integral``, the numerical
    engine is used instead (logged at DEBUG level).  The result is always a
    Python ``float``;
  - ``"closed"``: only the closed form (error if there is none);
  - ``"numeric"``: adaptive quadrature via
    :func:`~copul.measures.backend.numeric_backend`;
  - ``"symbolic"``: the SymPy route (the family's symbolic-only override
    if present, otherwise the generic base-class integration);
  - ``"mc"``: empirical estimator on ``n_samples`` samples of
    ``copula.rvs``.

The class integration lives in :class:`~copul.family.core.biv_core_copula.BivCoreCopula`:
its ``__init_subclass__`` wraps every subclass override of a measure method
with :func:`make_dispatcher`.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import math
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from copul.measures.backend import _to_float, free_parameters, numeric_backend, special_numeric
from copul.measures.numeric import DEFAULT_ATOL, DEFAULT_RTOL
from copul.measures.numeric import evaluate as _evaluate_numeric
from copul.measures.registry import _iter_keys, get_measure, resolve_key

log = logging.getLogger(__name__)

__all__ = [
    "LEGACY_METHODS",
    "MEASURE_METHODS",
    "METHODS",
    "MeasureResult",
    "compute",
    "make_dispatcher",
    "symbolic_measure",
]

#: Canonical measure method names on bivariate copulas.
MEASURE_METHODS = (
    "chatterjees_xi",
    "spearmans_rho",
    "kendalls_tau",
    "blests_nu",
    "spearmans_footrule",
    "ginis_gamma",
    "blomqvists_beta",
    "hoeffdings_d",
    "schweizer_wolff_sigma",
    "uniform_distance",
    "lp_distance",
    "blum_kiefer_rosenblatt",
    "mutual_information",
    "lambda_L",
    "lambda_U",
)

#: Deprecated method names -> canonical names.
LEGACY_METHODS = {
    "gini_gamma": "ginis_gamma",
    "spearman_footrule": "spearmans_footrule",
    "lp_concordance": "lp_distance",
}

METHODS = ("auto", "closed", "numeric", "symbolic", "mc")

# ids of copulas whose measure method is currently executing (nested calls)
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar(
    "copul_active_measures", default=frozenset()
)

_METHOD_TO_KEY = {
    "chatterjees_xi": "xi",
    "spearmans_rho": "rho",
    "kendalls_tau": "tau",
    "blests_nu": "nu",
    "spearmans_footrule": "footrule",
    "ginis_gamma": "gamma",
    "blomqvists_beta": "beta",
    "hoeffdings_d": "hoeffdings_d",
    "schweizer_wolff_sigma": "sigma",
    "uniform_distance": "kappa",
    "lp_distance": "lp",
    "blum_kiefer_rosenblatt": "bkr",
    "mutual_information": "mutual_information",
    "lambda_L": "lambda_l",
    "lambda_U": "lambda_u",
}

# keyword arguments consumed by the engine
_ENGINE_KW = ("rtol", "atol", "n_samples", "random_state")

# measures for which "auto" tries the generic symbolic route before
# numerics when the family provides no closed form (numerical tail
# coefficients are only extrapolations)
_AUTO_SYMBOLIC_FIRST = {"lambda_l", "lambda_u"}
_TAIL_TRUST = 1e-6


@dataclass
class MeasureResult:
    """Result of a measure evaluation.

    Attributes
    ----------
    key : str
        Canonical measure key.
    value : float or sympy.Expr
        The value (SymPy expression for copulas with free parameters).
    error : float or None
        Error estimate (``0.0`` for closed forms, ``None`` if unknown).
    method : str
        ``"closed"``, ``"numeric"``, ``"symbolic"`` or ``"mc"``.
    info : dict
        Additional diagnostics (e.g. numerical sources, fallback reason).
    """

    key: str
    value: Any
    error: float | None
    method: str
    info: dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        err = "" if self.error is None else f", error={self.error:.2g}"
        return f"MeasureResult({self.key}={self.value!r}{err}, method={self.method!r})"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def symbolic_measure(func: Callable) -> Callable:
    """Mark a measure override as a *symbolic integration* (not a closed form).

    Under ``method="auto"`` such overrides are only used while the copula
    has free parameters; fully specified copulas are evaluated numerically.
    With ``method="symbolic"`` they take precedence over the generic
    base-class integration.
    """
    func.__copul_symbolic__ = True
    return func


def _declares(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    p = params.get(name)
    return p is not None and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)


class _Unsupported(Exception):
    pass


def _to_number(val) -> float:
    """Convert a closed-form result to a finite float or raise."""
    if isinstance(val, MeasureResult):
        val = val.value
    inner = getattr(val, "func", None)
    if inner is not None and isinstance(inner, sp.Basic):
        val = inner
    if isinstance(val, sp.Basic):
        if val.has(sp.Integral, sp.Limit) or val.free_symbols:
            raise _Unsupported(f"non-numeric closed form {str(val)[:80]}")
        if val == sp.oo:  # legitimate e.g. for the mutual information of M
            return math.inf
        if val.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise _Unsupported(f"non-finite closed form {val}")
    f = _to_float(val)
    if math.isnan(f):
        raise _Unsupported("closed form returned nan")
    return f


def _impl_info(copula, method_name):
    attr = inspect.getattr_static(type(copula), method_name, None)
    if attr is None:
        return None, True
    info = getattr(attr, "__copul_measure__", None)
    if info is not None:
        return info["impl"], info["is_base"]
    if inspect.isfunction(attr):
        return attr, False
    return None, True


def _base_impl(method_name):
    from copul.family.core.biv_core_copula import BivCoreCopula

    attr = BivCoreCopula.__dict__.get(method_name)
    info = getattr(attr, "__copul_measure__", None)
    return info["impl"] if info else attr


def _class_param_names(copula):
    names = set()
    for p in list(getattr(type(copula), "params", None) or []) + list(
        getattr(copula, "params", None) or []
    ):
        names.add(str(p))
    return names


# ---------------------------------------------------------------------------
# core dispatch
# ---------------------------------------------------------------------------


class _CallSignatureError(TypeError):
    """Invalid arguments passed to a measure method (never triggers a fallback)."""


def dispatch(
    copula, method_name, impl, is_base, args, kwargs, method="auto", nested=False
) -> MeasureResult:
    """Evaluate a measure method call according to the dispatch semantics."""
    method = "auto" if method is None else str(method).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    kwargs = dict(kwargs)
    args = tuple(args)
    eng = {k: kwargs.pop(k) for k in _ENGINE_KW if k in kwargs}
    key = _METHOD_TO_KEY.get(method_name, method_name)

    # -- measure options ---------------------------------------------------
    opts: dict[str, Any] = {}
    if method_name == "chatterjees_xi" and "condition_on_y" in kwargs:
        opts["condition_on_y"] = bool(kwargs.pop("condition_on_y"))
    if method_name == "lp_distance":
        if "p" in kwargs:
            opts["p"] = kwargs.pop("p")
        elif args:
            opts["p"], args = args[0], args[1:]

    # -- parameters given in the call (old ``_set_params`` semantics) -------
    pnames = [str(p) for p in (getattr(copula, "params", None) or [])]
    cls_pnames = _class_param_names(copula)
    param_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in cls_pnames}
    if args and pnames and len(args) <= len(pnames):
        param_kw.update(zip(pnames, args))
        args = ()
    if (
        args
        and method_name == "chatterjees_xi"
        and len(args) == 1
        and isinstance(args[0], (bool, np.bool_))
    ):
        opts["condition_on_y"], args = bool(args[0]), ()
    if param_kw:
        copula._set_params((), param_kw)
    if opts.get("condition_on_y"):
        key = "xi_2"

    symbolic_only = bool(getattr(impl, "__copul_symbolic__", False))

    def call(fn):
        kw = dict(kwargs)
        for o, val in opts.items():
            if _declares(fn, o):
                kw[o] = val
            elif o == "condition_on_y" and not val:
                continue
            else:
                raise _Unsupported(f"{fn.__qualname__} does not support {o}={val!r}")
        for k, val in eng.items():
            if _declares(fn, k):
                kw[k] = val
        try:
            inspect.signature(fn).bind(copula, *args, **kw)
        except TypeError as exc:  # misuse by the caller: never mask it by a fallback
            raise _CallSignatureError(f"{fn.__qualname__}(): {exc}") from None
        except ValueError:  # signature not introspectable
            pass
        return fn(copula, *args, **kw)

    # -- symbolic ---------------------------------------------------------------
    if method == "symbolic":
        fn = impl if (impl is not None and (is_base or symbolic_only)) else _base_impl(method_name)
        if fn is None:
            raise NotImplementedError(f"No symbolic implementation of {method_name}.")
        return MeasureResult(key, call(fn), None, "symbolic")

    free = free_parameters(copula)
    if free:
        if method in ("numeric", "mc"):
            raise ValueError(
                f"{type(copula).__name__} has free parameters {free}; "
                f"method={method!r} needs a fully specified copula."
            )
        if impl is None:
            raise NotImplementedError(f"{type(copula).__name__} has no {method_name}.")
        kind = "symbolic" if (is_base or symbolic_only) else "closed"
        return MeasureResult(key, call(impl), None, kind)

    # -- fully specified ---------------------------------------------------------
    has_closed = impl is not None and not is_base and not symbolic_only
    if method == "auto" and has_closed and nested:
        # nested call from within another closed form of the same copula
        # (e.g. ``super().spearmans_rho() + correction``): return the raw
        # closed-form value; the outermost call handles failures/fallbacks
        return MeasureResult(key, call(impl), None, "closed")
    if method == "closed":
        if not has_closed:
            raise NotImplementedError(
                f"{type(copula).__name__} provides no closed form for {key!r}."
            )
        return MeasureResult(key, _to_number(call(impl)), 0.0, "closed")

    if method == "auto":
        reason = None
        if has_closed:
            try:
                return MeasureResult(key, _to_number(call(impl)), 0.0, "closed")
            except _CallSignatureError:
                raise
            except Exception as e:  # fall back to numerics
                reason = f"closed form failed: {type(e).__name__}: {e}"
                log.debug("%s.%s: %s -> numeric", type(copula).__name__, method_name, reason)
        elif key in _AUTO_SYMBOLIC_FIRST:
            # tail coefficients: the numerical extrapolation is trusted when
            # its error estimate is small; otherwise the symbolic limit is
            # tried (symbolic limits of pseudo-inverse generators are
            # occasionally wrong, numerical ones slow for log-type tails)
            num = None
            try:
                num = _numeric(copula, key, opts, eng)
                if num.error is not None and num.error <= _TAIL_TRUST:
                    return num
            except Exception as e:
                reason = f"numeric tail estimate failed: {type(e).__name__}: {e}"
            try:
                fn = impl if impl is not None else _base_impl(method_name)
                res = MeasureResult(key, _to_number(call(fn)), None, "symbolic")
                if num is not None:
                    res.info["numeric_estimate"] = (num.value, num.error)
                return res
            except Exception as e:
                reason = f"symbolic limit failed: {type(e).__name__}: {e}"
                log.debug("%s.%s: %s -> numeric", type(copula).__name__, method_name, reason)
            if num is not None:
                num.info["fallback_reason"] = reason
                return num
        res = _numeric(copula, key, opts, eng)
        if reason:
            res.info["fallback_reason"] = reason
        return res

    if method == "numeric":
        return _numeric(copula, key, opts, eng)
    return _monte_carlo(copula, key, eng)


def _numeric(copula, key, opts, eng) -> MeasureResult:
    rtol = float(eng.get("rtol", DEFAULT_RTOL))
    atol = float(eng.get("atol", DEFAULT_ATOL))
    free = free_parameters(copula)
    if not free:
        try:
            sp1 = special_numeric(copula, key, rtol, atol)
        except Exception as e:  # pragma: no cover - defensive
            log.debug("special 1D formula failed: %s", e)
            sp1 = None
        if sp1 is not None:
            val, err, source = sp1
            return MeasureResult(key, float(val), float(err), "numeric", {"source": {key: source}})
    be = numeric_backend(copula)
    p = opts.get("p", 2)
    val, err = _evaluate_numeric(key, be, rtol=rtol, atol=atol, p=float(p), prefer_h=be.prefer_h)
    return MeasureResult(key, float(val), float(err), "numeric", {"source": dict(be.source)})


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


def _ranks(x):
    from scipy.stats import rankdata

    return rankdata(x, method="average")


def _monte_carlo(copula, key, eng) -> MeasureResult:
    from scipy import stats

    n = int(eng.get("n_samples", 100_000))
    rs = eng.get("random_state", None)
    try:
        data = copula.rvs(n, random_state=rs)
    except TypeError:
        if rs is not None:
            np.random.seed(rs)
        data = copula.rvs(n)
    data = np.asarray(data, dtype=float)
    x, y = data[:, 0], data[:, 1]
    n = x.size
    R, S = _ranks(x), _ranks(y)
    U, V = R / (n + 1), S / (n + 1)
    if key == "rho":
        val = float(stats.spearmanr(x, y)[0])
    elif key == "tau":
        val = float(stats.kendalltau(x, y)[0])
    elif key in ("xi", "xi_2"):
        from copul.chatterjee import xi_ncalculate

        val = float(xi_ncalculate(x, y) if key == "xi" else xi_ncalculate(y, x))
    elif key == "footrule":
        val = 1 - 3 * np.sum(np.abs(R - S)) / (n**2 - 1)
    elif key == "gamma":
        val = np.sum(np.abs(R + S - n - 1) - np.abs(R - S)) / np.floor(n**2 / 2)
    elif key == "beta":
        val = 4 * np.mean((U <= 0.5) & (V <= 0.5)) - 1
    elif key == "nu":
        val = 2 - 12 * np.mean((1 - U) ** 2 * V)
    else:
        raise NotImplementedError(f"No Monte Carlo estimator for {key!r}.")
    return MeasureResult(
        key, float(val), 1.0 / math.sqrt(n), "mc", {"n_samples": n, "error": "O(n^-1/2) scale"}
    )


# ---------------------------------------------------------------------------
# class integration
# ---------------------------------------------------------------------------


def _dispatch_tracked(copula, method_name, impl, is_base, args, kwargs, method):
    """``dispatch`` recording the copula as active (to detect nested calls)."""
    active = _ACTIVE.get()
    token = _ACTIVE.set(active | {id(copula)})
    try:
        return dispatch(
            copula, method_name, impl, is_base, args, kwargs, method, nested=id(copula) in active
        )
    finally:
        _ACTIVE.reset(token)


def make_dispatcher(method_name: str, impl: Callable, is_base: bool) -> Callable:
    """Wrap ``impl`` (a measure method) with the dispatch logic."""

    @functools.wraps(impl)
    def dispatcher(self, *args, method="auto", **kwargs):
        return _dispatch_tracked(self, method_name, impl, is_base, args, kwargs, method).value

    dispatcher.__copul_measure__ = {"impl": impl, "is_base": is_base, "name": method_name}
    doc = impl.__doc__ or ""
    dispatcher.__doc__ = (
        doc + "\n\n        Dispatch: pass ``method=`` one of ``'auto'`` (default), ``'closed'``,"
        " ``'numeric'``, ``'symbolic'``, ``'mc'``; fully specified copulas return"
        " floats under ``'auto'`` (see :mod:`copul.measures.engine`).\n"
    )
    return dispatcher


def deprecated_alias(old: str, new: str) -> Callable:
    def alias(self, *args, **kwargs):
        warnings.warn(
            f"{old}() is deprecated; use {new}() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, new)(*args, **kwargs)

    alias.__name__ = old
    alias.__qualname__ = old
    alias.__doc__ = f"Deprecated alias of :meth:`{new}`."
    alias.__copul_deprecated_alias__ = new
    return alias


def install_dispatchers(cls, is_base: bool = False) -> None:
    """Wrap the measure methods resolved on ``cls`` with dispatchers.

    Legacy names defined directly on ``cls`` (e.g. ``gini_gamma``) are
    treated as overrides of their canonical successor.
    """
    own = cls.__dict__
    for old, new in LEGACY_METHODS.items():
        f = own.get(old)
        if (
            inspect.isfunction(f)
            and not hasattr(f, "__copul_deprecated_alias__")
            and new not in own
        ):
            setattr(cls, new, f)
            setattr(cls, old, deprecated_alias(old, new))
    for name in MEASURE_METHODS:
        resolved = None
        for klass in cls.__mro__:
            if name in klass.__dict__:
                resolved = klass.__dict__[name]
                break
        if resolved is None or hasattr(resolved, "__copul_measure__"):
            continue
        if not inspect.isfunction(resolved):
            continue
        setattr(cls, name, make_dispatcher(name, resolved, is_base=is_base))


# ---------------------------------------------------------------------------
# public front-end
# ---------------------------------------------------------------------------


def _evaluate_key(copula, key, method, kw) -> MeasureResult:
    spec = get_measure(key)
    kw = {k: v for k, v in kw.items() if k in _ENGINE_KW or k in spec.options}
    for k, v in spec.method_kwargs.items():
        kw.setdefault(k, v)
    impl, is_base = _impl_info(copula, spec.method_name)
    return _dispatch_tracked(copula, spec.method_name, impl, is_base, (), kw, method)


def compute(
    copula,
    measures: str | Iterable[str] = "rho",
    method: str = "auto",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    full_output: bool = False,
    **options,
):
    """Compute one or several dependence measures of a bivariate copula.

    Parameters
    ----------
    copula : copula object
    measures : str or iterable of str
        Measure key(s) or aliases (see :func:`copul.measures.list_measures`).
    method : {"auto", "closed", "numeric", "symbolic", "mc"}
        Evaluation route (see module docstring).
    rtol, atol : float
        Tolerances of the numerical engine (measure scale).
    full_output : bool
        Return :class:`MeasureResult` objects (value, error, method used).
    **options
        Measure options (``p`` for ``"lp"``, ``condition_on_y`` for ``"xi"``)
        and Monte Carlo settings (``n_samples``, ``random_state``).

    Returns
    -------
    float, sympy.Expr, MeasureResult or dict thereof
        A single value for a single key, otherwise ``{key: value}``.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.measures import compute
    >>> round(compute(cp.Clayton(2), "tau"), 12)
    0.5
    >>> r = compute(cp.Clayton(2), "rho", full_output=True)
    >>> r.method
    'numeric'
    """
    kw = dict(options)
    kw["rtol"] = rtol
    kw["atol"] = atol
    single = isinstance(measures, str)
    keys = [resolve_key(measures)] if single else _iter_keys(measures)
    out = {}
    for k in keys:
        res = _evaluate_key(copula, k, method, kw)
        out[k] = res if full_output else res.value
    return out[keys[0]] if single else out
