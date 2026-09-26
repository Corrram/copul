r"""
Random checkerboard copulas for stress-testing inequalities and regions.

:func:`random_checkerboards` yields checkerboard copulas whose mass matrices
are drawn with the diverse strategies of
:meth:`~copul.checkerboard.biv_check_pi.BivCheckPi.random_bistochastic_matrix`
(permutations, sparse/dense Birkhoff mixtures, Sinkhorn-normalised random
matrices, band matrices, ...), optionally transformed to satisfy a structural
condition:

``"si"``
    the stochastically increasing rearrangement of Strothmann, Dette &
    Siburg (2022): sort the cumulative row sums
    :math:`B_{kj}=\sum_{l\le j}P_{kl}` of every column in decreasing order and
    difference again (exactly SI for ``kind="pi"``);
``"sd"``
    the rows of the SI rearrangement in reverse order;
``"exchangeable"``, ``"radially_symmetric"``
    symmetrisation :math:`(P+P^\top)/2` resp. :math:`(P+JPJ)/2`;
callable
    rejection sampling with the predicate ``condition(copula) -> bool``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from copul.optim.checkerboard_formulas import checkerboard_copula, normalize_kind

__all__ = ["random_checkerboards", "random_mass_matrix", "si_rearrangement"]


def si_rearrangement(P: np.ndarray) -> np.ndarray:
    """Stochastically increasing rearrangement of a mass matrix.

    Parameters
    ----------
    P : numpy.ndarray
        Mass matrix with equal row sums.

    Returns
    -------
    numpy.ndarray
        A mass matrix with the same column sums whose checkerboard (``"pi"``)
        is stochastically increasing.
    """
    P = np.asarray(P, dtype=float)
    B = np.zeros((P.shape[0], P.shape[1] + 1))
    B[:, 1:] = np.cumsum(P, axis=1)
    B = -np.sort(-B, axis=0)
    return np.clip(np.diff(B, axis=1), 0.0, None)


def random_mass_matrix(
    n: int, m: int | None = None, rng=None, strategy: str | None = None
) -> np.ndarray:
    """A random mass matrix (row sums ``1/m``, column sums ``1/n``).

    Square matrices use
    :meth:`BivCheckPi.random_bistochastic_matrix`; rectangular ones use
    Sinkhorn balancing of a random positive matrix.
    """
    rng = np.random.default_rng(rng)
    m = n if m is None else m
    if m == n:
        from copul.checkerboard.biv_check_pi import BivCheckPi

        P = BivCheckPi.random_bistochastic_matrix(n, rng=rng, strategy=strategy)
        return P / P.sum()
    from copul.optim.problem import balance

    A = rng.random((m, n)) ** rng.uniform(1.0, 4.0) + 1e-4
    return balance(A)


def _apply_condition(P: np.ndarray, condition: str) -> np.ndarray:
    c = condition.lower()
    if c in ("si", "ci", "cis"):
        return si_rearrangement(P)
    if c in ("sd", "cd"):
        return si_rearrangement(P)[::-1].copy()
    if c in ("exchangeable", "symmetric"):
        return 0.5 * (P + P.T)
    if c in ("radially_symmetric", "radial"):
        return 0.5 * (P + P[::-1, ::-1])
    raise ValueError(
        f"Unknown condition {condition!r}; use 'si', 'sd', 'exchangeable', "
        "'radially_symmetric' or a callable."
    )


def random_checkerboards(
    n_samples: int | None = None,
    grid: int | tuple[int, int] = (2, 50),
    kind: str = "pi",
    condition: str | Callable | None = None,
    rng=None,
    strategy: str | None = None,
    max_tries: int = 1000,
) -> Iterator:
    """Iterate over random checkerboard copulas.

    Parameters
    ----------
    n_samples : int, optional
        Number of copulas to yield (infinite iterator if ``None``).
    grid : int or (int, int)
        Fixed grid size or inclusive range from which it is drawn.
    kind : {"pi", "min", "w"}
        Checkerboard class (``BivCheckPi``, ``BivCheckMin``, ``BivCheckW``).
    condition : str or callable, optional
        ``"si"``, ``"sd"``, ``"exchangeable"``, ``"radially_symmetric"`` or a
        predicate ``copula -> bool`` (rejection sampling).
    rng : numpy.random.Generator or int, optional
        Random source (seed for reproducibility).
    strategy : str, optional
        Fixed strategy of :meth:`BivCheckPi.random_bistochastic_matrix`.
    max_tries : int
        Rejection-sampling budget per sample for callable conditions.

    Yields
    ------
    BivCheckPi, BivCheckMin or BivCheckW

    Examples
    --------
    >>> from copul.search import random_checkerboards
    >>> cops = list(random_checkerboards(3, grid=4, condition="si", rng=0))
    >>> all(c.is_cis() for c in cops)
    True
    """
    rng = np.random.default_rng(rng)
    kind = normalize_kind(kind)
    count = 0
    while n_samples is None or count < n_samples:
        for _ in range(max_tries):
            if isinstance(grid, (tuple, list)):
                n = int(rng.integers(int(grid[0]), int(grid[1]) + 1))
            else:
                n = int(grid)
            P = random_mass_matrix(n, rng=rng, strategy=strategy)
            if isinstance(condition, str):
                P = _apply_condition(P, condition)
            cop = checkerboard_copula(P, kind)
            if callable(condition) and not condition(cop):
                continue
            break
        else:
            raise RuntimeError("rejection sampling exhausted max_tries")
        count += 1
        yield cop
