"""
Bivariate Checkerboard Copula module.

This module provides a bivariate checkerboard copula implementation
that combines properties of both CheckPi and BivCopula classes.
"""

import numpy as np
from typing import Union, List
import warnings

import sympy
from copul.checkerboard.check_pi import CheckPi
from copul.families.core.biv_core_copula import BivCoreCopula
from copul.schur_order.cis_verifier import CISVerifier


class BivCheckPi(CheckPi, BivCoreCopula):
    """
    Bivariate Checkerboard Copula class.

    This class implements a bivariate checkerboard copula, which is defined by
    a matrix of values that determine the copula's distribution.

    Attributes:
        params (List): Empty list as checkerboard copulas are non-parametric.
        intervals (dict): Empty dictionary as there are no parameters to bound.
        m (int): Number of rows in the checkerboard matrix.
        n (int): Number of columns in the checkerboard matrix.
    """

    params: List = []
    intervals: dict = {}

    def __init__(self, matr: Union[List[List[float]], np.ndarray], **kwargs):
        """
        Initialize a bivariate checkerboard copula.

        Args:
            matr: A matrix (2D array) defining the checkerboard distribution.
            **kwargs: Additional parameters passed to BivCopula.

        Raises:
            ValueError: If matrix dimensions are invalid or matrix contains negative values.
        """
        # Convert input to numpy array if it's a list
        if isinstance(matr, list):
            matr = np.array(matr, dtype=float)
        if isinstance(matr, sympy.Matrix):
            matr = np.array(matr).astype(float)

        # Input validation
        if not hasattr(matr, "ndim"):
            raise ValueError("Input matrix must be a 2D array or list")
        if matr.ndim != 2:
            raise ValueError(
                f"Input matrix must be 2-dimensional, got {matr.ndim} dimensions"
            )
        if np.any(matr < 0):
            raise ValueError("All matrix values must be non-negative")

        CheckPi.__init__(self, matr)
        BivCoreCopula.__init__(self, **kwargs)

        self.m = self.matr.shape[0]
        self.n = self.matr.shape[1]

        # Normalize matrix if not already normalized
        if not np.isclose(np.sum(self.matr), 1.0):
            warnings.warn(
                "Matrix not normalized. Normalizing to ensure proper density.",
                UserWarning,
            )
            self.matr = self.matr / np.sum(self.matr)

    def __str__(self) -> str:
        """
        Return a string representation of the copula.

        Returns:
            str: String representation showing dimensions of the checkerboard.
        """
        return f"BivCheckPi(m={self.m}, n={self.n})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        If the matrix is larger than 5x5, only the top-left 5x5 block is shown.

        Returns:
            str: A string representation of the object, including matrix info.
        """
        rows, cols = self.matr.shape
        if rows > 5 and cols > 5:
            matr_preview = np.array2string(
                self.matr[:5, :5], max_line_width=80, suppress_small=True
            ).replace("\n", " ")
            matr_str = f"{matr_preview} (top-left 5x5 block)"
        else:
            matr_str = self.matr.tolist()

        return f"BivCheckPi(matr={matr_str}, m={self.m}, n={self.n})"

    @property
    def is_symmetric(self) -> bool:
        """
        Check if the copula is symmetric (C(u,v) = C(v,u)).

        Returns:
            bool: True if the copula is symmetric, False otherwise.
        """
        if self.matr.shape[0] != self.matr.shape[1]:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        For checkerboard copulas, this property is always True.

        Returns:
            bool: Always True for checkerboard copulas.
        """
        return True

    def is_cis(self, i=1) -> bool:
        """
        Check if the copula is cis.
        """
        return CISVerifier(i).is_cis(self)

    def transpose(self):
        """
        Transpose the checkerboard matrix.
        """
        return BivCheckPi(self.matr.T)

    def cond_distr_1(self, *args):
        return self.cond_distr(1, *args)

    def cond_distr_2(self, *args):
        return self.cond_distr(2, *args)

    def rho(self) -> float:
        """
        Compute Spearman's rho for a bivariate checkerboard copula.
        """
        p = np.asarray(self.matr, dtype=float)
        m, n = p.shape
        # Compute the factors (2*(i+1)-1)=2i+1 for rows and columns:
        i = np.arange(m).reshape(-1, 1)  # Column vector (i from 0 to m-1)
        j = np.arange(n).reshape(1, -1)  # Row vector (j from 0 to n-1)

        numerator = (2 * m - 2 * i - 1) * (2 * n - 2 * j - 1)
        denominator = m * n
        omega = numerator / denominator
        trace = np.trace(omega.T @ p)
        return 3 * trace - 3

    def tau(self) -> float:
        """
        Calculate the tau coefficient more efficiently using numpy's vectorized operations.

        Returns:
            float: The calculated tau coefficient.
        """
        Xi_m = 2 * np.tri(self.m) - np.eye(self.m)
        Xi_n = 2 * np.tri(self.n) - np.eye(self.n)
        return 1 - np.trace(Xi_m @ self.matr @ Xi_n @ self.matr.T)

    def xi(self, condition_on_y: bool = False) -> float:
        if condition_on_y:
            delta = self.matr.T
            m = self.n
            n = self.m
        else:
            delta = self.matr
            m = self.m
            n = self.n
        T = np.ones(n) - np.tri(n)
        M = T @ T.T + T.T + 1 / 3 * np.eye(n)
        trace = np.trace(delta.T @ delta @ M)
        xi = 6 * m / n * trace - 2
        return xi


if __name__ == "__main__":
    matr3 = [[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]]

    def make_adjacent_ones(n: int) -> np.ndarray:
        """
        Return an n×n 0/1‐matrix where each row has exactly two consecutive 1’s
        and each column also sums to 2.
        """
        assert n % 2 == 0, "n must be even"
        mat = np.zeros((n, n), dtype=int)
        distinct_positions = n // 2  # number of possible start‐positions
        for i in range(n):
            start = (i % distinct_positions) * 2
            mat[i, start : start + 2] = 1
        return mat

    def make_16x16_adjacent_ones() -> np.ndarray:
        return make_adjacent_ones(16)

    def make_32x32_adjacent_ones() -> np.ndarray:
        return make_adjacent_ones(32)

    def make_64x64_adjacent_ones() -> np.ndarray:
        return make_adjacent_ones(64)

    def make_9x9_subblock_matrix():
        D = np.diag([1, 1, 1])  # 3×3 diagonal
        OO = np.ones((3, 3), dtype=int) / 3  # 3×3 ones
        Z = np.zeros((3, 3), dtype=int)  # 3×3 zeros

        top = np.hstack([D, Z, Z])
        middle = np.hstack([Z, OO, Z])
        bottom = np.hstack([Z, Z, D])

        return np.vstack([top, middle, bottom])

    def make_6x6_subblock_matrix():
        D = np.diag([1, 1])  # 2×2 diagonal
        OO = np.ones((2, 2), dtype=int) / 2  # 2×2 ones
        Z = np.zeros((2, 2), dtype=int)  # 2×2 zeros

        top = np.hstack([D, Z, Z])
        middle = np.hstack([Z, OO, Z])
        bottom = np.hstack([Z, Z, D])

        return np.vstack([top, middle, bottom])

    M = make_6x6_subblock_matrix().T / 6
    n = 6
    T = np.ones(n) - np.tri(n)
    M_xi = T @ T.T + T.T + 1 / 3 * np.eye(n)
    print(np.trace(M @ M.T @ M_xi))
    print("Row sums:", np.unique(M.sum(axis=1)))
    print("Col sums:", np.unique(M.sum(axis=0)))
    M = [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]]
    ccop = BivCheckPi(M)
    xi = ccop.xi()
    ccop.plot_cdf()
