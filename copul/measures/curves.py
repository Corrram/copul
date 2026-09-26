"""
Parameter sweeps and calibration of one-parameter (sub)families.

* :func:`measure_curve` evaluates dependence measures along the free
  parameter of a family, e.g. ``Clayton().measure_curve(["xi", "rho"])``;
* :func:`from_measure` inverts a measure by root finding on the free
  parameter, e.g. ``Clayton.from_measure("tau", 0.5)`` returns
  ``Clayton(theta=2)``.

Both are exposed as methods of every bivariate copula (see
:class:`~copul.family.core.biv_core_copula.BivCoreCopula`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from copul.measures.backend import free_parameters
from copul.measures.engine import compute
from copul.measures.registry import _iter_keys, get_measure, resolve_key

log = logging.getLogger(__name__)

__all__ = ["MeasureCurve", "default_parameter_values", "from_measure", "measure_curve"]

_SPAN = 20.0  # length of the default range on an unbounded side


@dataclass
class MeasureCurve:
    """Values of dependence measures along a parameter sweep.

    Attributes
    ----------
    family : str
        Class name of the swept copula.
    param : str
        Name of the swept parameter.
    params : numpy.ndarray
        Parameter values.
    data : dict
        ``{key: ndarray}`` of measure values (``nan`` where evaluation
        failed).
    """

    family: str
    param: str
    params: np.ndarray
    data: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[resolve_key(key)]

    def keys(self):
        return self.data.keys()

    def as_dict(self) -> dict[str, np.ndarray]:
        """``{param: params, key1: values1, ...}``."""
        out = {self.param: self.params}
        out.update(self.data)
        return out

    def plot(self, ax=None, **kwargs):
        """Plot all measures against the parameter; returns the axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        for k, vals in self.data.items():
            m = get_measure(k)
            ax.plot(self.params, vals, label=f"${m.symbol}$", **kwargs)
        ax.set_xlabel(self.param)
        ax.set_title(self.family)
        ax.grid(True)
        ax.legend()
        return ax

    def __repr__(self) -> str:
        return (
            f"MeasureCurve({self.family}, {self.param} in "
            f"[{self.params.min():.4g}, {self.params.max():.4g}], n={self.params.size}, "
            f"keys={list(self.data)})"
        )


def _select_param(copula, param: str | None) -> str:
    free = free_parameters(copula)
    if param is None:
        if len(free) != 1:
            raise ValueError(
                f"{type(copula).__name__} has free parameters {free}; "
                "pass param=... (and fix the others) to sweep one of them."
            )
        return free[0]
    param = str(param)
    if param not in free:
        raise ValueError(f"{param!r} is not a free parameter of {type(copula).__name__} ({free}).")
    others = [p for p in free if p != param]
    if others:
        raise ValueError(
            f"Parameters {others} are still free; fix them first, e.g. copula({others[0]}=...)."
        )
    return param


def _interval(copula, param) -> tuple[float, float, bool, bool]:
    iv = None
    try:
        iv = (copula.intervals or {}).get(param)
    except Exception:
        iv = None
    if iv is None:
        return -math.inf, math.inf, True, True
    lo, hi = float(iv.inf), float(iv.sup)
    return lo, hi, bool(iv.left_open), bool(iv.right_open)


def default_parameter_values(copula, param: str | None = None, n: int = 100) -> np.ndarray:
    """``n`` parameter values spread over a sensible part of the interval.

    Bounded sides are used (slightly inside when open); unbounded sides are
    truncated at a distance of 20 from the other end (``[-10, 10]`` if both
    are unbounded).
    """
    param = _select_param(copula, param)
    lo, hi, lo_open, hi_open = _interval(copula, param)
    if math.isinf(lo) and math.isinf(hi):
        lo, hi = -_SPAN / 2, _SPAN / 2
        lo_open = hi_open = False
    elif math.isinf(hi):
        hi, hi_open = lo + _SPAN, False
    elif math.isinf(lo):
        lo, lo_open = hi - _SPAN, False
    eps = 1e-3 * (hi - lo)
    a = lo + eps if lo_open else lo
    b = hi - eps if hi_open else hi
    return np.linspace(a, b, int(n))


def _instance(copula, param, value):
    return copula(**{param: float(value)})


def measure_curve(
    copula,
    keys: str | Iterable[str] = ("rho",),
    param: str | None = None,
    values: Sequence[float] | None = None,
    n: int = 100,
    method: str = "auto",
    **kwargs,
) -> MeasureCurve:
    """Evaluate measures along the free parameter of ``copula``.

    Parameters
    ----------
    copula : copula object with exactly one free parameter (or ``param``)
    keys : str or iterable of str
        Measure keys.
    param : str, optional
        Parameter to sweep.
    values : array_like, optional
        Parameter values (default: :func:`default_parameter_values`).
    n : int
        Number of default values.
    method : str
        Evaluation route passed to :func:`copul.measures.compute`.

    Returns
    -------
    MeasureCurve
    """
    keys = _iter_keys(keys)
    param = _select_param(copula, param)
    vals = (
        default_parameter_values(copula, param, n)
        if values is None
        else np.asarray(values, dtype=float).ravel()
    )
    data = {k: np.full(vals.size, np.nan) for k in keys}
    for i, x in enumerate(vals):
        try:
            c = _instance(copula, param, x)
        except Exception as e:
            log.debug("cannot instantiate %s=%s: %s", param, x, e)
            continue
        for k in keys:
            try:
                data[k][i] = float(compute(c, k, method=method, **kwargs))
            except Exception as e:
                log.debug("measure %s failed at %s=%s: %s", k, param, x, e)
    return MeasureCurve(type(copula).__name__, param, vals, data)


def from_measure(
    obj,
    key: str,
    value: float,
    param: str | None = None,
    bracket: tuple[float, float] | None = None,
    method: str = "auto",
    xtol: float = 1e-12,
    **kwargs,
):
    """Return the family member whose measure ``key`` equals ``value``.

    Parameters
    ----------
    obj : copula class or (partially specified) copula instance
    key : str
        Measure key, e.g. ``"tau"``.
    value : float
        Target value.
    param : str, optional
        Parameter to calibrate (required if several are free).
    bracket : (float, float), optional
        Parameter bracket containing the solution.  By default the
        (truncated) parameter interval is used and expanded on unbounded
        sides until the measure changes sign relative to ``value``.
    method : str
        Evaluation route for the measure.
    xtol : float
        Absolute tolerance on the parameter (Brent's method).

    Raises
    ------
    ValueError
        If no bracket enclosing ``value`` is found.

    Examples
    --------
    >>> import copul as cp
    >>> float(cp.Clayton.from_measure("tau", 0.5).theta)
    2.0
    """
    from scipy.optimize import brentq

    base = obj() if isinstance(obj, type) else obj
    param = _select_param(base, param)
    target = float(value)

    def g(x):
        return float(compute(_instance(base, param, x), key, method=method, **kwargs)) - target

    lo, hi, lo_open, hi_open = _interval(base, param)
    if bracket is None:
        grid = default_parameter_values(base, param, 2)
        a, b = float(grid[0]), float(grid[-1])
    else:
        a, b = map(float, bracket)
    ga, gb = g(a), g(b)
    if not (ga * gb <= 0) and bracket is None:
        # non-monotone measures (xi, distances): scan the default range and
        # take the sign change with the largest parameter values
        xs = default_parameter_values(base, param, 13)
        gs = []
        for x in xs:
            try:
                gs.append(g(float(x)))
            except Exception:
                gs.append(np.nan)
        gs = np.asarray(gs)
        idx = [i for i in range(len(xs) - 1) if gs[i] * gs[i + 1] <= 0]
        if idx:
            i = idx[-1]
            a, b, ga, gb = float(xs[i]), float(xs[i + 1]), gs[i], gs[i + 1]
    # expand unbounded sides geometrically (at most ~2^12 times the range)
    step = max(1.0, b - a)
    for _ in range(12):
        if ga * gb <= 0 or bracket is not None:
            break
        if not (math.isinf(hi) or math.isinf(lo)):
            break
        if math.isinf(hi):
            b += step
            gb = g(b)
        if math.isinf(lo):
            a -= step
            ga = g(a)
        step *= 2
    if not (ga * gb <= 0):
        raise ValueError(
            f"Could not bracket {key}={target} for {type(base).__name__} "
            f"({param} in [{a:.4g}, {b:.4g}] gives {ga + target:.6g} .. {gb + target:.6g})."
        )
    if ga == 0:
        root = a
    elif gb == 0:
        root = b
    else:
        root = brentq(g, a, b, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=200)
    root = float(root)
    # snap to "nice" values (e.g. theta=2 instead of 1.9999999999997)
    r = round(root, 10)
    if abs(r - root) <= 10 * xtol:
        root = float(int(r)) if float(r).is_integer() else r
    return _instance(base, param, root)
