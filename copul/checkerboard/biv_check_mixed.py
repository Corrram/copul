from __future__ import annotations

import numpy as np

from copul.checkerboard._biv_mixin import BivCheckerboardMixin
from copul.family.core.biv_core_copula import BivCoreCopula
from copul.family.core.copula_approximator_mixin import CopulaApproximatorMixin
from copul.family.core.copula_plotting_mixin import CopulaPlottingMixin


class BivCheckMixed(
    BivCheckerboardMixin, BivCoreCopula, CopulaPlottingMixin, CopulaApproximatorMixin
):
    r"""
    Mixed checkerboard copula (per–cell choice of :math:`\Pi` / ↗ / ↘).

    A sign matrix :math:`S` with entries :math:`\{0,+1,-1\}` selects, in every
    checkerboard rectangle, which base copula to use:

      * :math:`0`  → independence  (:math:`\Pi`)
      * :math:`+1` → perfect positive dependence (check-min, ↗)
      * :math:`-1` → perfect negative dependence (check-w, ↘)

    The probability matrix :math:`\Delta` (argument ``matr``) is shared across
    all three modes.  All numerics are exact and vectorised (see
    :mod:`copul.checkerboard._biv_engine`), e.g.

    .. math::

       \tau = 1 - \operatorname{tr}(\Xi_m\Delta\Xi_n\Delta^\top)
              + \sum_{ij} S_{ij}\Delta_{ij}^2,\qquad
       \xi = \xi_\Pi + \tfrac{m}{n}\sum_{ij}|S_{ij}|\Delta_{ij}^2,\qquad
       \rho = \rho_\Pi + \tfrac{1}{mn}\sum_{ij} S_{ij}\Delta_{ij}.
    """

    params: list = []
    intervals: dict = {}

    def __init__(
        self,
        matr: np.ndarray | list[list[float]],
        sign: np.ndarray | list[list[int]] | None = None,
        **kwargs,
    ):
        matr = np.array(matr, dtype=float)
        if matr.ndim != 2:
            raise ValueError("`matr` must be a 2-D array")
        if np.any(matr < 0):
            raise ValueError("`matr` must be non-negative")
        total = matr.sum()
        if total <= 0:
            raise ValueError("`matr` must have positive total mass")
        matr = matr / total

        self.m, self.n = matr.shape

        # --- S --------------------------------------------------------- #
        if sign is None:
            sign = np.zeros_like(matr, dtype=int)
        sign = np.array(sign, dtype=int)
        if sign.shape != matr.shape:
            raise ValueError("`sign` must have the same shape as `matr`")
        if not np.isin(sign, (-1, 0, 1)).all():
            raise ValueError("`sign` entries must be −1, 0, or +1")

        self.matr = matr  # probability matrix  Δ
        self.sign = sign  # sign selector      S

        super().__init__()

    def _kernel_signs(self):
        return self.sign

    # ------------------------------------------------------------------ #
    #                       basic properties                             #
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        return f"BivCheckMixed(m={self.m}, n={self.n})"

    __repr__ = __str__

    @property
    def is_absolutely_continuous(self) -> bool:
        return bool(np.all(self.sign[self.matr > 0] == 0))

    @property
    def is_symmetric(self) -> bool:
        return (
            self.m == self.n
            and np.allclose(self.matr, self.matr.T)
            and np.array_equal(self.sign, self.sign.T)
        )

    def transpose(self):
        """Swap the roles of U and V."""
        return BivCheckMixed(self.matr.T, sign=self.sign.T)

    def _cell_indices(self, u: float, v: float) -> tuple[int, int]:
        """Cell ``(i, j)`` containing the point ``(u, v)``."""
        i = min(int(np.floor(u * self.m)), self.m - 1)
        j = min(int(np.floor(v * self.n)), self.n - 1)
        return i, j

    def pdf(self, *args, **kwargs):
        """Density of the absolutely continuous part (cells with sign 0)."""
        if not self.is_absolutely_continuous:
            from copul.exceptions import PropertyUnavailableException

            raise PropertyUnavailableException(
                "PDF does not exist for a mixed checkerboard with singular cells."
            )
        from copul.checkerboard import _biv_engine as eng

        u, v, scalar = eng.parse_uv(args, kwargs)
        return eng.finish(eng.pdf_pi(self.matr, u, v), scalar)


# -------------------------------------------------------------------------- #
# quick manual check
# -------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    Delta = [[1, 0], [0, 1]]
    S = np.array([[0, 0], [0, 1]])
    cop = BivCheckMixed(Delta, sign=S)
    print("τ:", cop.kendalls_tau(), "ρ:", cop.spearmans_rho(), "ξ:", cop.chatterjees_xi())
