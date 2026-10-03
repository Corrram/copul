import logging

import numpy as np

log = logging.getLogger(__name__)


class CornerSetVerifier:
    r"""Verifier for corner set monotonicity properties of copulas.

    A copula :math:`C` is *left corner set decreasing* (LCSD) if
    :math:`P(U\le u, V\le v \mid U\le u', V\le v')` is decreasing in
    :math:`u'` and :math:`v'` for all :math:`(u,v)`. By
    [Nelsen, An Introduction to Copulas, Cor. 5.2.17], this holds if and
    only if the *function* :math:`C` is totally positive of order 2, i.e.

    .. math::

       C(u_1,v_1)\,C(u_2,v_2) \;\ge\; C(u_1,v_2)\,C(u_2,v_1)
       \qquad \text{for all } u_1\le u_2,\; v_1\le v_2.

    Similarly, :math:`C` is *right corner set increasing* (RCSI) if and
    only if the survival function
    :math:`\bar C(u,v) = 1-u-v+C(u,v)` is TP2.

    Note that this is TP2-ness of the cdf/survival function, *not* of the
    density (the latter is the strictly stronger ``TP2`` property).
    """

    def __init__(self, n_grid: int = 40):
        self.n_grid = n_grid

    # ------------------------------------------------------------------
    # instance-level checks
    # ------------------------------------------------------------------

    def is_lcsd(self, copula) -> bool:
        r"""Check whether the copula instance is LCSD (i.e. :math:`C` is TP2)."""
        engine = self._engine(copula, "LCSD")
        if engine is not None:
            return engine
        return self._is_tp2_function(copula, survival=False)

    def is_rcsi(self, copula) -> bool:
        r"""Check whether the copula instance is RCSI (i.e. :math:`\bar C` is TP2)."""
        engine = self._engine(copula, "RCSI")
        if engine is not None:
            return engine
        return self._is_tp2_function(copula, survival=True)

    def _engine(self, copula, key):
        """:func:`copul.theory.dependence.check_property` (exact or refined grid)."""
        from copul.theory.dependence import _is_specified_bivariate, check_property

        if _is_specified_bivariate(copula):
            return bool(check_property(copula, key, n_grid=max(int(self.n_grid), 9)))
        return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _cdf_matrix(self, copula, grid):
        cdf = copula.cdf
        if hasattr(copula, "cdf_vectorized"):
            try:
                uu, vv = np.meshgrid(grid, grid, indexing="ij")
                H = np.asarray(copula.cdf_vectorized(uu, vv), dtype=float)
                # sanity check against scalar cdf at a few interior points
                k = len(grid) // 2
                for idx in {(k, k), (1, k), (k, len(grid) - 2)}:
                    ref = float(cdf(u=grid[idx[0]], v=grid[idx[1]]))
                    if abs(H[idx] - ref) > 1e-6 * max(1.0, abs(ref)):
                        raise ValueError("cdf_vectorized inconsistent")
                return H
            except Exception:  # pragma: no cover - fall back to scalar cdf
                pass
        return np.array([[float(cdf(u=u, v=v)) for v in grid] for u in grid])

    def _is_tp2_function(self, copula, survival: bool, tol: float = 1e-12) -> bool:
        grid = np.linspace(0.001, 0.999, self.n_grid)
        H = self._cdf_matrix(copula, grid)
        if survival:
            uu, vv = np.meshgrid(grid, grid, indexing="ij")
            H = 1.0 - uu - vv + H
        H = np.clip(H, 0.0, None)

        if np.all(H > 0):
            # For strictly positive matrices, TP2 for all pairs reduces to
            # TP2 for adjacent grid cells.
            a = H[:-1, :-1] * H[1:, 1:]
            b = H[:-1, 1:] * H[1:, :-1]
            return bool(np.all(a - b >= -tol * np.maximum(a, 1e-300)))

        # With zeros present the adjacent-cell reduction is not valid; check
        # all pairs i<i', j<j' by broadcasting.
        n = len(grid)
        a = H[:, None, :, None] * H[None, :, None, :]  # H[i,j] * H[i',j']
        b = H[:, None, None, :] * H[None, :, :, None]  # H[i,j'] * H[i',j]
        iu = np.triu_indices(n, k=1)
        diff = (a - b + tol * np.maximum(a, 1e-300))[iu[0], iu[1], :, :]
        diff = diff[:, iu[0], iu[1]]
        return bool(np.all(diff >= 0))
