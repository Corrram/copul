"""
Mixed checkerboard copula (per–cell choice of Π / ↗ / ↘).

A sign-matrix S with entries  0, +1, –1  selects in every rectangle
      0  → independence          (checkerboard Π)
     +1  → perfect positive dep. (check-min  ↗)
     –1  → perfect negative dep. (check-w    ↘)
The probability matrix Δ (= `matr`) is shared across all three modes.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_w import BivCheckW
from copul.families.core.biv_core_copula import BivCoreCopula
from copul.families.core.copula_plotting_mixin import CopulaPlottingMixin
from copul.schur_order.cis_verifier import CISVerifier


class BivCheckMixed(BivCoreCopula, CopulaPlottingMixin):
    """Bivariate mixed checkerboard copula (see module doc-string)."""

    params: List = []
    intervals: dict = {}

    # ------------------------------------------------------------------ #
    #                               init                                 #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        matr: Union[np.ndarray, List[List[float]]],
        sign: Union[np.ndarray, List[List[int]], None] = None,
        **kwargs,
    ):
        # --- Δ --------------------------------------------------------- #
        matr = np.asarray(matr, dtype=float)
        if matr.ndim != 2:
            raise ValueError("`matr` must be a 2-D array")
        if np.any(matr < 0):
            raise ValueError("`matr` must be non-negative")
        if not np.isclose(matr.sum(), 1.0):
            matr = matr / matr.sum()

        self.m, self.n = matr.shape

        # --- S --------------------------------------------------------- #
        if sign is None:
            sign = np.zeros_like(matr, dtype=int)
        sign = np.asarray(sign, dtype=int)
        if sign.shape != matr.shape:
            raise ValueError("`sign` must have the same shape as `matr`")
        if not np.isin(sign, (-1, 0, 1)).all():
            raise ValueError("`sign` entries must be −1, 0, or +1")

        self.matr = matr  # probability matrix  Δ
        self.sign = sign  # sign selector      S

        # instantiate base copulas once
        self._pi = BivCheckPi(matr, **kwargs)
        self._min = BivCheckMin(matr, **kwargs)
        self._w = BivCheckW(matr, **kwargs)

        super().__init__()

    # ------------------------------------------------------------------ #
    #                       basic properties                             #
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        return f"BivCheckMixed(m={self.m}, n={self.n})"

    __repr__ = __str__

    @property
    def is_absolutely_continuous(self) -> bool:
        return np.all(self.sign == 0)

    @property
    def is_symmetric(self) -> bool:
        return (
            self.m == self.n
            and np.allclose(self.matr, self.matr.T)
            and np.array_equal(self.sign, self.sign.T)
        )

    # ------------------------------------------------------------------ #
    #                       helper (cell localisation)                   #
    # ------------------------------------------------------------------ #
    def _cell_indices(self, u: float, v: float) -> tuple[int, int]:
        """Return 0-based cell indices (i,j) for point (u,v)."""
        i = min(int(np.floor(u * self.m)), self.m - 1)
        j = min(int(np.floor(v * self.n)), self.n - 1)
        return i, j

    # ------------------------------------------------------------------ #
    #                               CDF                                  #
    # ------------------------------------------------------------------ #
    def cdf(self, u: float | np.ndarray, v: float | np.ndarray):
        """Piecewise delegates to Π, ↗ or ↘ depending on S."""
        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        if u_arr.shape != v_arr.shape:
            raise ValueError("u and v must have the same shape")

        out = np.empty_like(u_arr, dtype=float)

        it = np.nditer(
            [u_arr, v_arr, out],
            flags=["refs_ok", "multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for uu, vv, dest in it:
            i, j = self._cell_indices(float(uu), float(vv))
            s = self.sign[i, j]
            if s == 0:
                dest[...] = self._pi.cdf(float(uu), float(vv))
            elif s == 1:
                dest[...] = self._min.cdf(float(uu), float(vv))
            else:  # s == −1
                dest[...] = self._w.cdf(float(uu), float(vv))

        return out if out.shape else float(out)

    # ------------------------------------------------------------------ #
    #                matrices for closed-form measures                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _xi_matrix(size: int) -> np.ndarray:
        """Matrix 2·tri − I (strict lower-triangular coefficient)."""
        return 2 * np.tri(size) - np.eye(size)

    def _omega(self) -> np.ndarray:
        """
        Ω-matrix with
            Ω_{i,j} = ((2m-2i-1)(2n-2j-1)) / (m n)
        for **zero-based** indices i,j = 0,… .
        """
        i = np.arange(self.m).reshape(-1, 1)  # 0 … m-1
        j = np.arange(self.n).reshape(1, -1)
        num = (2 * self.m - 2 * i - 1) * (2 * self.n - 2 * j - 1)
        return num / (self.m * self.n)

    # ------------------------------------------------------------------ #
    #                    exact dependence measures                       #
    # ------------------------------------------------------------------ #
    def tau(self) -> float:
        Xi_m = self._xi_matrix(self.m)
        Xi_n = self._xi_matrix(self.n)
        core = np.trace(Xi_m @ self.matr @ Xi_n @ self.matr.T)
        extra = np.sum(self.sign * (self.matr**2))
        return 1.0 - core + extra

    def xi(self, *, condition_on_y: bool = False) -> float:
        # base value from plain checkerboard
        base = self._pi.xi(condition_on_y)

        # scaling m/n depends on conditioning
        m, n = (self.n, self.m) if condition_on_y else (self.m, self.n)
        extra = (m / n) * np.sum(np.abs(self.sign) * (self.matr**2))
        return base + extra

    def rho(self) -> float:
        Omega = self._omega()
        core = 3.0 * np.trace(Omega.T @ self.matr) - 3.0
        extra = np.sum(self.sign * self.matr) / (self.m * self.n)
        return core + extra

    # ------------------------------------------------------------------ #
    #         simple numeric conditional distributions (plots)           #
    # ------------------------------------------------------------------ #
    def cond_distr(self, i: int, u: float, v: float):
        """A quick numeric version (uses Π-behaviour).  Good enough for plots."""
        return self._pi.cond_distr(i, u, v)

    def cond_distr_1(self, u: float, v: float):
        return self.cond_distr(1, u, v)

    def cond_distr_2(self, u: float, v: float):
        return self.cond_distr(2, u, v)

    # ------------------------------------------------------------------ #
    # misc                                                               #
    # ------------------------------------------------------------------ #
    def is_cis(self, i: int = 1) -> bool:
        return CISVerifier(i).is_cis(self)

    # ------------------------------------------------------------------ #
    #                            sampling                                #
    # ------------------------------------------------------------------ #
    def rvs(
        self, n: int = 1, *, random_state: int | None = None, **kwargs
    ) -> np.ndarray:
        """
        Draw *n* i.i.d. samples from the mixed checkerboard copula.

        Parameters
        ----------
        n : int, default 1
            Number of samples to generate.
        random_state : int, optional
            Seed for the RNG (for reproducibility).

        Returns
        -------
        ndarray, shape (n, 2)
            Samples ``(U, V)`` in ``[0,1]²``.
        """
        rng = np.random.default_rng(random_state)

        # ------ 1) pick the cell for every sample  -------------------- #
        flat_Δ = self.matr.ravel()
        probs = flat_Δ / flat_Δ.sum()  # should already sum to 1
        flat_idx = rng.choice(flat_Δ.size, size=n, p=probs)

        i_idx, j_idx = np.unravel_index(flat_idx, self.matr.shape)  # (n,)

        # ------ 2) draw location *inside* the selected cell ----------- #
        u = np.empty(n)
        v = np.empty(n)

        # split into the three regimes once; then fill the vectors
        for s_val, mask in (
            (0, self.sign[i_idx, j_idx] == 0),  # Π   (independence)
            (1, self.sign[i_idx, j_idx] == 1),  # ↗   (check-min)
            (-1, self.sign[i_idx, j_idx] == -1),  # ↘   (check-w)
        ):
            if not mask.any():
                continue
            ii = i_idx[mask]
            jj = j_idx[mask]

            if s_val == 0:  # uniform on the rectangle
                u[mask] = (ii + rng.random(mask.sum())) / self.m
                v[mask] = (jj + rng.random(mask.sum())) / self.n

            elif s_val == 1:  # perfect + dependence ↗
                t = rng.random(mask.sum())
                u[mask] = (ii + t) / self.m
                v[mask] = (jj + t) / self.n

            else:  # perfect – dependence ↘
                t = rng.random(mask.sum())
                u[mask] = (ii + t) / self.m
                v[mask] = (jj + 1 - t) / self.n  # descending line

        return np.column_stack((u, v))


# -------------------------------------------------------------------------- #
# quick manual check
# -------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    Δ = np.full((2, 2), 0.25)
    S = np.array([[0, 1], [-1, 1]])
    cop = BivCheckMixed(Δ, sign=S)
    print("τ:", cop.tau(), "ρ:", cop.rho(), "ξ:", cop.xi())
