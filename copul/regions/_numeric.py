r"""
Grid quadrature of dependence measures from a copula's CDF.

Used to validate boundary copulas independently of their (possibly buggy)
closed-form methods.  Only the CDF is required; :math:`\partial_1C` is replaced
by cell averages :math:`N\,[C(u_{k+1},v)-C(u_k,v)]`, which is exact for
piecewise-linear-in-:math:`u` CDFs and otherwise underestimates
:math:`\int\!\!\int(\partial_1C)^2` by :math:`O(1/N)` near discontinuities of
:math:`\partial_1 C`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from copul.regions.measures import resolve

__all__ = ["cdf_grid", "numeric_measures"]


_EVAL_ERRORS = (TypeError, ValueError, AttributeError, IndexError, NotImplementedError)


def _try(fn) -> np.ndarray | None:
    """Evaluate ``fn()`` as a float array; ``None`` if the copula cannot broadcast."""
    try:
        return np.asarray(fn(), dtype=float)
    except _EVAL_ERRORS:
        return None


def _eval_cdf(copula: Any, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate the CDF on the outer grid ``u x v`` (shape ``(len(u), len(v))``)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    target = (u.size, v.size)
    f = getattr(copula, "cdf_vectorized", None)
    if f is not None:
        out = _try(lambda: f(u[:, None], v[None, :]))
        if out is not None and out.shape == target:
            return out
        U, V = np.meshgrid(u, v, indexing="ij")
        out = _try(lambda: f(U.ravel(), V.ravel()))
        if out is not None and out.size == U.size:
            return out.reshape(target)
    # checkerboards accept an (N, 2) array of points
    U, V = np.meshgrid(u, v, indexing="ij")
    out = _try(lambda: copula.cdf(np.column_stack([U.ravel(), V.ravel()])))
    if out is not None and out.size == U.size:
        return out.reshape(target)
    out = np.empty(target)
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            out[i, j] = float(copula.cdf(float(ui), float(vj)))
    return out


def _eval_diag(copula: Any, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    f = getattr(copula, "cdf_vectorized", None)
    if f is not None:
        out = _try(lambda: f(u, v))
        if out is not None and out.shape == u.shape:
            return out
    out = _try(lambda: copula.cdf(np.column_stack([u, v])))
    if out is not None and out.size == u.size:
        return out.reshape(u.shape)
    return np.array([float(copula.cdf(float(a), float(b))) for a, b in zip(u, v)])


def cdf_grid(copula: Any, N: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CDF on the nodes ``u_k = k/N`` times the midpoints ``v_j = (j+1/2)/N``.

    Returns
    -------
    u, v, C : numpy.ndarray
        ``C[k, j] = C(u_k, v_j)`` with shape ``(N+1, N)``.
    """
    u = np.linspace(0.0, 1.0, N + 1)
    v = (np.arange(N) + 0.5) / N
    C = _eval_cdf(copula, u, v)
    return u, v, C


def numeric_measures(
    copula: Any,
    measures: Iterable[str] = ("xi", "rho", "footrule", "gamma", "beta", "nu"),
    N: int = 400,
) -> dict[str, float]:
    r"""Approximate dependence measures of ``copula`` by grid quadrature.

    Parameters
    ----------
    copula : object
        Anything with ``cdf_vectorized(u, v)`` (preferred) or ``cdf(u, v)``.
    measures : iterable of str
        Measure keys; Kendall's tau uses cell averages of both partial
        derivatives.
    N : int
        Grid resolution.

    Returns
    -------
    dict
        ``{key: value}``.
    """
    keys = [resolve(k) for k in measures]
    out: dict[str, float] = {}
    need_grid = any(k in ("xi", "rho", "nu", "tau") for k in keys)
    if need_grid:
        _u, _v, C = cdf_grid(copula, N)
        h = np.diff(C, axis=0) * N  # cell averages of d1 C, shape (N, N)
        Cmid = 0.5 * (C[1:] + C[:-1])  # C at (u midpoints, v midpoints) approx.
        umid = (np.arange(N) + 0.5) / N
    for k in keys:
        if k == "xi":
            out[k] = float(6.0 * np.mean(h**2) - 2.0)
        elif k == "rho":
            out[k] = float(12.0 * np.mean(Cmid) - 3.0)
        elif k == "nu":
            out[k] = float(24.0 * np.mean((1.0 - umid)[:, None] * Cmid) - 2.0)
        elif k == "tau":
            # d2 C via cell averages in v on the transposed grid
            vv = np.linspace(0.0, 1.0, N + 1)
            C2 = _eval_cdf(copula, umid, vv)
            h2 = np.diff(C2, axis=1) * N  # averages of d2 C over v-cells at u mids
            # align: h has (u-cell, v-mid), h2 has (u-mid, v-cell): same cells
            out[k] = float(1.0 - 4.0 * np.mean(h * h2))
        elif k in ("footrule", "gamma"):
            M = 20 * N
            t = (np.arange(M) + 0.5) / M
            d = float(np.mean(_eval_diag(copula, t, t)))
            if k == "footrule":
                out[k] = 6.0 * d - 2.0
            else:
                a = float(np.mean(_eval_diag(copula, t, 1.0 - t)))
                out[k] = 4.0 * (d + a) - 2.0
        elif k == "beta":
            val = _eval_diag(copula, np.array([0.5]), np.array([0.5]))
            out[k] = float(4.0 * float(np.ravel(val)[0]) - 1.0)
    return out
