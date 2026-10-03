r"""
Numerical verification of the copula axioms in :math:`d` dimensions.

A function :math:`C:[0,1]^d\to[0,1]` is a :math:`d`-copula iff (Nelsen,
2006, §2.10)

1. it is *grounded*: :math:`C(u)=0` whenever some :math:`u_i=0`;
2. it has *uniform margins*: :math:`C(1,\dots,1,u_i,1,\dots,1)=u_i`;
3. it is *d-increasing*: the :math:`C`-volume
   :math:`V_C([a,b])=\sum_{\varepsilon\in\{0,1\}^d}(-1)^{d-|\varepsilon|}C(c^\varepsilon)`
   of every box :math:`[a,b]\subseteq[0,1]^d` is non-negative.

:func:`is_copula_nd` checks these conditions on a tensor grid: the volumes
of all grid cells are obtained by successive first differences of the grid
values along each axis (a :math:`d`-fold difference is exactly the
inclusion--exclusion formula above).  Passing the check on a grid is
necessary, not sufficient; failing it is a certificate that ``C`` is not a
copula (e.g. :math:`W_3`, whose central cell has negative volume).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["is_copula_nd"]

_MAX_POINTS = 4_000_000


def _resolve(C: Any, d: int | None) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
    from copul.multivariate.base import CopulaND

    if isinstance(C, CopulaND):
        return C._cdf_clean, C.dim
    if hasattr(C, "cdf") and getattr(C, "dim", None) is not None:
        from copul.multivariate.basic import as_copula_nd

        Cn = as_copula_nd(C)
        return Cn._cdf_clean, Cn.dim
    if callable(C):
        if d is None:
            raise ValueError("is_copula_nd(callable) needs the dimension d=.")

        def f(U):
            return np.asarray(C(U), dtype=float).reshape(-1)

        return f, int(d)
    raise TypeError(f"cannot check {type(C).__name__}: need a copula object or a callable.")


def is_copula_nd(
    C: Any,
    d: int | None = None,
    grid: int | Sequence[float] = 11,
    tol: float = 1e-8,
    return_details: bool = False,
):
    r"""Check grounding, uniform margins and :math:`d`-increasingness on a grid.

    Parameters
    ----------
    C : CopulaND, copula object or callable
        A :class:`~copul.multivariate.CopulaND`, any :math:`d`-dimensional
        copula object of copul (see :func:`~copul.multivariate.as_copula_nd`)
        or a callable ``C(U)`` evaluated on ``(N, d)`` arrays.  The *raw*
        values of callables are checked as given; copula objects are checked
        as evaluated by their ``cdf``.
    d : int, optional
        Dimension (required for callables).
    grid : int or sequence of float
        Number of equidistant knots per axis, or the knots themselves (0 and
        1 are added if missing).  The grid has ``len(knots)**d`` points.
    tol : float
        Tolerance for the boundary conditions and for negative cell volumes.
    return_details : bool
        Also return a dict of diagnostics.

    Returns
    -------
    bool or (bool, dict)
        Whether all checks passed; details: ``grounded``, ``margins``,
        ``d_increasing``, ``bounds`` (Fréchet--Hoeffding), ``min_volume``,
        ``total_mass``, ``max_margin_error``, ``n_knots``, ``dim``.

    Examples
    --------
    >>> import numpy as np
    >>> from copul.multivariate import is_copula_nd
    >>> is_copula_nd(lambda U: np.prod(U, axis=1), d=3)
    True
    >>> W3 = lambda U: np.maximum(U.sum(axis=1) - 2.0, 0.0)  # not a copula
    >>> is_copula_nd(W3, d=3)
    False
    """
    f, d = _resolve(C, d)
    if np.ndim(grid) == 0:
        knots = np.linspace(0.0, 1.0, int(grid))
    else:
        knots = np.unique(np.concatenate([[0.0, 1.0], np.asarray(grid, dtype=float)]))
        knots = knots[(knots >= 0.0) & (knots <= 1.0)]
    m = knots.size
    if m < 2:
        raise ValueError("the grid needs at least two knots.")
    if m**d > _MAX_POINTS:
        raise ValueError(f"grid with {m}^{d} points is too large; use fewer knots.")
    mesh = np.meshgrid(*([knots] * d), indexing="ij")
    pts = np.stack([g.ravel() for g in mesh], axis=1)
    with np.errstate(all="ignore"):
        vals = np.asarray(f(pts), dtype=float).reshape((m,) * d)
    finite = bool(np.all(np.isfinite(vals)))
    if not finite:
        details = {"finite": False, "dim": d, "n_knots": m}
        return (False, details) if return_details else False

    grounded = True
    max_margin_err = 0.0
    for k in range(d):
        sl = [slice(None)] * d
        sl[k] = 0
        grounded &= bool(np.all(np.abs(vals[tuple(sl)]) <= tol))
        sl = [-1] * d
        sl[k] = slice(None)
        max_margin_err = max(max_margin_err, float(np.max(np.abs(vals[tuple(sl)] - knots))))
    margins = max_margin_err <= tol

    vol = vals
    for axis in range(d):
        vol = np.diff(vol, axis=axis)
    min_vol = float(vol.min())
    total = float(vol.sum())
    increasing = min_vol >= -tol

    lo = np.maximum(pts.sum(axis=1) - d + 1.0, 0.0).reshape(vals.shape)
    hi = pts.min(axis=1).reshape(vals.shape)
    bounds = bool(np.all(vals >= lo - tol) and np.all(vals <= hi + tol))

    ok = grounded and margins and increasing and bounds
    if not return_details:
        return ok
    return ok, {
        "finite": True,
        "grounded": grounded,
        "margins": margins,
        "d_increasing": increasing,
        "bounds": bounds,
        "min_volume": min_vol,
        "total_mass": total,
        "max_margin_error": max_margin_err,
        "n_knots": m,
        "dim": d,
    }
