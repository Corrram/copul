r"""
Pseudo-observations (normalized ranks).

For a sample :math:`(X_{i1},\dots,X_{id})_{i=1}^n` with ranks
:math:`R_{ij}` of :math:`X_{ij}` among :math:`X_{1j},\dots,X_{nj}` the
pseudo-observations are

.. math::

   \hat U_{ij} = \frac{R_{ij}}{n + 1}\qquad\text{(default)}\quad\text{or}\quad
   \hat U_{ij} = \frac{R_{ij}}{n}.

The :math:`(n+1)` scaling keeps all points inside the open unit cube, which
is required for evaluating log-densities in pseudo-likelihood estimation
(Genest, Ghoudi & Rivest, 1995).

References
----------
* Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric
  estimation procedure of dependence parameters in multivariate families of
  distributions. *Biometrika* 82(3), 543--552.
* Deheuvels, P. (1979). La fonction de dépendance empirique et ses
  propriétés. *Acad. Roy. Belg. Bull. Cl. Sci.* 65, 274--292.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata

from copul._lazy import is_pandas_instance
from copul.stats._utils import RandomLike, as_rng

__all__ = ["pseudo_obs", "ranks"]

_TIES = ("average", "random", "max", "min", "ordinal", "first", "dense")


def ranks(x: Any, ties: str = "average", random_state: RandomLike = None) -> np.ndarray:
    """Column-wise ranks ``1..n`` of ``x`` (1-d or ``(n, d)``).

    Parameters
    ----------
    x : array_like
        Sample, 1-d or one column per variable.
    ties : {"average", "random", "max", "min", "ordinal", "first", "dense"}
        Tie handling.  ``"random"`` breaks ties uniformly at random
        (reproducibly through ``random_state``); ``"first"`` is an alias of
        ``"ordinal"``; the others follow :func:`scipy.stats.rankdata`.
    random_state : int, Generator or None
        Seed for ``ties="random"``.

    Returns
    -------
    numpy.ndarray
        Float array of the same shape as ``x``.
    """
    arr = np.asarray(x.to_numpy() if is_pandas_instance(x, "DataFrame", "Series") else x, float)
    if ties not in _TIES:
        raise ValueError(f"ties must be one of {_TIES}, got {ties!r}.")
    one_d = arr.ndim == 1
    a = arr[:, None] if one_d else arr
    if a.ndim != 2:
        raise ValueError(f"x must be 1-d or 2-d, got shape {arr.shape}.")
    if np.isnan(a).any():
        raise ValueError("x contains NaN values.")
    if ties == "random":
        rng = as_rng(random_state)
        out = np.empty_like(a)
        n = a.shape[0]
        for j in range(a.shape[1]):
            order = np.lexsort((rng.random(n), a[:, j]))
            out[order, j] = np.arange(1, n + 1)
    else:
        method = "ordinal" if ties == "first" else ties
        out = rankdata(a, method=method, axis=0).astype(float)
    return out[:, 0] if one_d else out


def pseudo_obs(
    X: Any,
    ties: str = "average",
    scale: str = "n+1",
    random_state: RandomLike = None,
) -> np.ndarray:
    r"""Pseudo-observations :math:`\hat U_{ij} = R_{ij}/(n+1)` of a sample.

    Parameters
    ----------
    X : array_like of shape (n, d) or (n,)
        Sample (numpy array, list or pandas DataFrame), one column per
        variable.
    ties : {"average", "random", "max", "min", "ordinal", "first", "dense"}
        Tie handling of the ranks (see :func:`ranks`); ``"average"`` gives
        mid-ranks, ``"max"`` the empirical distribution function
        :math:`\hat F_j(X_{ij}) \cdot n`.
    scale : {"n+1", "n"}
        Divide the ranks by :math:`n+1` (default, points in :math:`(0,1)^d`)
        or by :math:`n` (the empirical marginal distribution functions).
    random_state : int, Generator or None
        Seed for ``ties="random"``.

    Returns
    -------
    numpy.ndarray
        Pseudo-observations of the same shape as ``X``.

    Examples
    --------
    >>> pseudo_obs([[1.0, 10.0], [3.0, 30.0], [2.0, 5.0]])
    array([[0.25, 0.5 ],
           [0.75, 0.75],
           [0.5 , 0.25]])
    """
    r = ranks(X, ties=ties, random_state=random_state)
    n = r.shape[0]
    if scale in ("n+1", "n + 1", "np1"):
        return r / (n + 1.0)
    if scale == "n":
        return r / float(n)
    raise ValueError(f"scale must be 'n+1' or 'n', got {scale!r}.")
