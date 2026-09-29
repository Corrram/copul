"""Internal helpers of :mod:`copul.stats` (data validation, rank counts)."""

from __future__ import annotations

from typing import Any

import numpy as np

from copul._lazy import is_pandas_instance

__all__ = [
    "RandomLike",
    "as_data",
    "as_rng",
    "chunk_size",
    "dominance_counts",
    "grid_counts_blocks",
]

#: Anything accepted by :func:`numpy.random.default_rng`.
RandomLike = int | np.random.Generator | np.random.SeedSequence | None

# number of float64 entries processed per vectorized chunk (~32 MB)
_CHUNK_ELEMS = 4_000_000


def as_rng(random_state: Any = None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` (never touches global state)."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.randint(0, 2**31 - 1))
    return np.random.default_rng(random_state)


def as_data(data: Any, min_dim: int = 2, max_dim: int | None = None) -> np.ndarray:
    """Validate ``data`` and return a float array of shape ``(n, d)``.

    Accepts arrays, nested lists, pandas DataFrames and tuples ``(x, y)`` of
    two 1-d samples.
    """
    if is_pandas_instance(data, "DataFrame", "Series"):
        arr = np.asarray(data.to_numpy(), dtype=float)
    elif isinstance(data, tuple) and len(data) >= 2 and np.ndim(data[0]) == 1:
        arr = np.column_stack([np.asarray(c, dtype=float).ravel() for c in data])
    else:
        arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        raise ValueError("data must be two-dimensional with one column per variable.")
    if arr.ndim != 2:
        raise ValueError(f"data must be a 2-d array, got shape {arr.shape}.")
    n, d = arr.shape
    if d < min_dim:
        raise ValueError(f"data must have at least {min_dim} columns, got {d}.")
    if max_dim is not None and d > max_dim:
        raise ValueError(f"data must have at most {max_dim} columns, got {d}.")
    if n < 2:
        raise ValueError("data must contain at least two observations.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("data contains NaN or infinite values.")
    return arr


def chunk_size(n: int) -> int:
    """Rows per chunk such that a ``(rows, n)`` float array has ~4e6 entries."""
    return max(1, _CHUNK_ELEMS // max(int(n), 1))


def dominance_counts(x: np.ndarray, y: np.ndarray, ties: str = "strict") -> np.ndarray:
    r"""Bivariate dominance counts of the sample points.

    For every observation ``i`` returns

    * ``ties="strict"``: :math:`\#\{j : x_j < x_i,\ y_j < y_i\}`;
    * ``ties="weak"``: :math:`\#\{j : x_j \le x_i,\ y_j \le y_i\}` (including
      ``j = i``), i.e. :math:`n\,C_n(x_i, y_i)` for the empirical cdf;
    * ``ties="hoeffding"``: Hoeffding's (1948) tie-corrected count
      :math:`\#\{x_j<x_i, y_j<y_i\} + \tfrac14\#\{j\ne i: x_j=x_i, y_j=y_i\}
      + \tfrac12\#\{x_j=x_i, y_j<y_i\} + \tfrac12\#\{x_j<x_i, y_j=y_i\}`.

    Computed by chunked vectorized comparisons (:math:`O(n^2)` time,
    :math:`O(n)` extra memory per chunk row).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    out = np.empty(n, dtype=float)
    step = chunk_size(n)
    for s in range(0, n, step):
        xi = x[s : s + step, None]
        yi = y[s : s + step, None]
        if ties == "strict":
            out[s : s + step] = np.count_nonzero((x[None, :] < xi) & (y[None, :] < yi), axis=1)
        elif ties == "weak":
            out[s : s + step] = np.count_nonzero((x[None, :] <= xi) & (y[None, :] <= yi), axis=1)
        elif ties == "hoeffding":
            lx, ex = x[None, :] < xi, x[None, :] == xi
            ly, ey = y[None, :] < yi, y[None, :] == yi
            both_eq = np.count_nonzero(ex & ey, axis=1) - 1.0  # exclude j == i
            out[s : s + step] = (
                np.count_nonzero(lx & ly, axis=1)
                + 0.25 * both_eq
                + 0.5 * np.count_nonzero(ex & ly, axis=1)
                + 0.5 * np.count_nonzero(lx & ey, axis=1)
            )
        else:  # pragma: no cover - internal
            raise ValueError(ties)
    return out


def grid_counts_blocks(r: np.ndarray, s: np.ndarray):
    r"""Yield row blocks of the rank count matrix.

    For ordinal ranks ``r, s`` (permutations of ``1..n``) the matrix
    :math:`N_{ij} = \#\{k : r_k \le i,\ s_k \le j\}`, :math:`i, j = 1..n`,
    i.e. :math:`n\,C_n(i/n, j/n)`, is generated in blocks ``(i0, block)``
    with ``block`` of shape ``(b, n)`` holding rows ``i0+1 .. i0+b`` -- the
    full :math:`n\times n` matrix is never stored.
    """
    r = np.asarray(r, dtype=np.int64).ravel()
    s = np.asarray(s, dtype=np.int64).ravel()
    n = r.size
    col_of_row = np.empty(n, dtype=np.int64)  # s-rank of the point with r-rank i+1
    col_of_row[r - 1] = s - 1
    step = chunk_size(n)
    carry = np.zeros(n, dtype=np.int64)
    for i0 in range(0, n, step):
        rows = np.arange(i0, min(i0 + step, n))
        inc = np.zeros((rows.size, n), dtype=np.int64)
        inc[np.arange(rows.size), col_of_row[rows]] = 1
        np.cumsum(inc, axis=1, out=inc)  # counts s <= j within each row
        np.cumsum(inc, axis=0, out=inc)  # counts r <= i within the block
        inc += carry[None, :]
        carry = inc[-1].copy()
        yield i0, inc
