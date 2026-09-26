r"""
Markov (``*``-) product of bivariate copulas.

.. math::

   (A * B)(u, v) = \int_0^1 \partial_2 A(u, t)\, \partial_1 B(t, v)\, dt .

Exact checkerboard case
-----------------------
For two independence-kernel checkerboards (:class:`BivCheckPi`) with mass
matrices :math:`P` (``m x k``) and :math:`Q` (``k' x n``), write
:math:`A(u,t) = \sum_{i,l} P_{il}\,\alpha_i(u)\beta_l(t)` with the cell
fractions :math:`\alpha_i(u) = (mu - i)^+ \wedge 1` etc.  Then
:math:`\partial_2 A(u,t) = k\sum_i P_{il}\alpha_i(u)` for ``t`` in column
cell ``l`` and analogously for :math:`\partial_1 B`, so

.. math::

   A * B = \text{BivCheckPi}\bigl(P\, M\, Q\bigr),\qquad
   M_{ll'} = k\,k'\,\bigl|[\tfrac{l}{k},\tfrac{l+1}{k}]\cap
             [\tfrac{l'}{k'},\tfrac{l'+1}{k'}]\bigr| .

For compatible grids (``k = k'``) this is simply ``k * P @ Q``.  The product
is associative, ``Pi`` is absorbing and the checkerboard ``M`` of order ``k``
is neutral for order-``k`` checkerboards.

General copulas
---------------
For any other pair the product is approximated by a :class:`BivCheckPi` on an
``n_grid x n_grid`` grid: the cdf of ``A * B`` is computed at all grid nodes by
vectorised midpoint quadrature over ``t`` (``n_quad`` nodes),

.. math::

   G_{ij} \approx \sum_q w_q\, \partial_2 A(u_i, t_q)\, \partial_1 B(t_q, v_j)
   \quad (\text{a matrix product}),

and the cell masses are the rectangle increments of ``G``.  The partial
derivatives are taken from vectorised ``cond_distr`` methods (checkerboards,
Bernstein, shuffles), from the lambdified symbolic cdf of parametric
families, or from central differences of the cdf as a last resort.
"""

from __future__ import annotations

import warnings

import numpy as np
import sympy as sp


def _is_exact_pi_checkerboard(C) -> bool:
    from copul.checkerboard._biv_mixin import BivCheckerboardMixin

    if not isinstance(C, BivCheckerboardMixin):
        return False
    signs = C._kernel_signs()
    if signs is None:
        return True
    signs = np.broadcast_to(np.asarray(signs), np.shape(C.matr))
    return not np.any(signs[np.asarray(C.matr) > 0])


def _overlap_matrix(k: int, kp: int) -> np.ndarray:
    """``M[l, l'] = k k' |cell_l(k) intersected with cell_l'(k')|``."""
    lo = np.maximum(np.arange(k)[:, None] / k, np.arange(kp)[None, :] / kp)
    hi = np.minimum((np.arange(k)[:, None] + 1) / k, (np.arange(kp)[None, :] + 1) / kp)
    return k * kp * np.clip(hi - lo, 0.0, None)


def _checkerboard_product(A, B):
    from copul.checkerboard.biv_check_pi import BivCheckPi

    P = np.asarray(A.matr, dtype=float)
    Q = np.asarray(B.matr, dtype=float)
    M = _overlap_matrix(P.shape[1], Q.shape[0])
    return BivCheckPi(P @ M @ Q)


def _partial_on_grid(C, which: int, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Vectorised :math:`\\partial_{which} C(U, V)` for arrays of equal shape."""
    # 1) numerical copulas with vectorised conditional distributions
    try:
        out = np.asarray(C.cond_distr(which, U.ravel(), V.ravel()), dtype=float)
        if out.shape == (U.size,) and np.all(np.isfinite(out)):
            return out.reshape(U.shape)
    except Exception:
        pass
    # 2) symbolic families: lambdify the derivative of the cdf
    try:
        expr = C.cdf().func
        sym = C.u if which == 1 else C.v
        f = sp.lambdify((C.u, C.v), sp.diff(expr, sym), "numpy")
        with np.errstate(all="ignore"):
            out = np.broadcast_to(np.asarray(f(U, V), dtype=float), U.shape)
        if np.all(np.isfinite(out)):
            return np.array(out)
    except Exception:
        pass
    # 3) central differences of a vectorised cdf
    h = 1e-6
    try:
        cdf = C.cdf_vectorized
    except AttributeError:

        def cdf(a, b):
            return np.vectorize(lambda x, y: float(C.cdf(float(x), float(y))))(a, b)

    if which == 1:
        lo, hi = np.clip(U - h, 0, 1), np.clip(U + h, 0, 1)
        return (np.asarray(cdf(hi, V)) - np.asarray(cdf(lo, V))) / (hi - lo)
    lo, hi = np.clip(V - h, 0, 1), np.clip(V + h, 0, 1)
    return (np.asarray(cdf(U, hi)) - np.asarray(cdf(U, lo))) / (hi - lo)


def _numeric_product(A, B, n_grid: int, n_quad: int):
    from copul.checkerboard.biv_check_pi import BivCheckPi

    grid = np.linspace(0.0, 1.0, n_grid + 1)
    t = (np.arange(n_quad) + 0.5) / n_quad
    Ug, Tg = np.meshgrid(grid, t, indexing="ij")  # (n_grid+1, n_quad)
    dA = _partial_on_grid(A, 2, Ug, Tg)  # d2 A(u_i, t_q)
    Tg2, Vg = np.meshgrid(t, grid, indexing="ij")  # (n_quad, n_grid+1)
    dB = _partial_on_grid(B, 1, Tg2, Vg)  # d1 B(t_q, v_j)
    G = (dA @ dB) / n_quad
    # boundary conditions of a copula are known exactly
    G[0, :] = 0.0
    G[:, 0] = 0.0
    G[-1, :] = grid
    G[:, -1] = grid
    mass = G[1:, 1:] - G[:-1, 1:] - G[1:, :-1] + G[:-1, :-1]
    mass = np.clip(mass, 0.0, None)
    return BivCheckPi(mass)


def markov_product(
    A,
    B,
    *,
    n_grid: int = 100,
    n_quad: int = 2000,
    checkerboard=None,
):
    r"""
    Markov product :math:`(A * B)(u,v) = \int_0^1 \partial_2 A(u,t)\,\partial_1 B(t,v)\,dt`.

    Parameters
    ----------
    A, B : bivariate copulas
        Any bivariate copulas of the package (checkerboards, Bernstein,
        shuffles, symbolic families, ...).
    n_grid : int, default 100
        Grid size of the returned checkerboard approximation (general case).
    n_quad : int, default 2000
        Number of midpoint quadrature nodes in ``t`` (general case).
    checkerboard : ignored
        Deprecated; the general result is always a :class:`BivCheckPi`.

    Returns
    -------
    BivCheckPi
        Exactly ``A * B`` if both factors are independence-kernel
        checkerboards (matrix ``P M Q``, see module docstring; ``k P @ Q`` for
        compatible ``k``-grids), otherwise an ``n_grid x n_grid`` checkerboard
        whose cdf matches ``A * B`` at the grid nodes up to quadrature error.
    """
    if checkerboard is not None:
        warnings.warn(
            "markov_product(checkerboard=...) is deprecated and ignored; the "
            "result is always a BivCheckPi.",
            DeprecationWarning,
            stacklevel=2,
        )
    if _is_exact_pi_checkerboard(A) and _is_exact_pi_checkerboard(B):
        return _checkerboard_product(A, B)
    return _numeric_product(A, B, int(n_grid), int(n_quad))


star_product = markov_product
