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

    @classmethod
    def generate_randomly(cls, grid_size: int | list | None = None, n: int = 1):
        if grid_size is None:
            grid_size = [2, 50]
        generated_copulas = []
        for i in range(n):
            if isinstance(grid_size, list):
                grid_size = np.random.randint(*grid_size)
            # 1) draw n permutations all at once via argsort of uniforms
            perms = np.argsort(
                np.random.rand(grid_size, grid_size), axis=1
            )  # shape (n,n)
            # 2) draw n cauchy random variables
            a = np.abs(np.random.standard_cauchy(size=grid_size))  # shape (n,)
            # a**1.5
            # a = a**1.5

            # 3) build weighted sum of permuted identity matrices:
            #    M[j,k] = sum_i a[i] * 1{perms[i,j] == k}
            #    -> we can do this in one np.add.at call
            rows = np.repeat(
                np.arange(grid_size)[None, :], grid_size, axis=0
            )  # shape (n,n)
            cols = perms  # shape (n,n)
            weights = np.broadcast_to(a[:, None], (grid_size, grid_size))  # shape (n,n)
            M = np.zeros((grid_size, grid_size), float)
            np.add.at(M, (rows.ravel(), cols.ravel()), weights.ravel())

            # 4) feed into copul
            generated_copulas.extend([cls(M)])
        if n == 1:
            return generated_copulas[0]
        return generated_copulas

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

    def footrule(self) -> float:
        """
        Compute Spearman's Footrule (psi) for a bivariate checkerboard copula.

        This method correctly implements the analytical integral of C(u,u).
        It is implemented for square checkerboard matrices.

        Returns:
            float: The value of Spearman's Footrule.
        """
        if self.m != self.n:
            warnings.warn(
                "Footrule analytical formula is implemented for square matrices only."
            )
            return np.nan

        p = np.asarray(self.matr, dtype=float)
        n = self.m

        total_integral_diag = 0
        # Loop through each diagonal cell I_kk (using 0-based index k_0)
        for k_0 in range(n):
            # Cumulative sum of the box to the top-left of the current diagonal cell
            sum_box = np.sum(p[:k_0, :k_0]) if k_0 > 0 else 0

            # Sum of the column elements above the diagonal element in the same column
            sum_col_above = np.sum(p[:k_0, k_0]) if k_0 > 0 else 0

            # Sum of the row elements left of the diagonal element in the same row
            sum_row_left = np.sum(p[k_0, :k_0]) if k_0 > 0 else 0

            # The diagonal element itself
            diag_element = p[k_0, k_0]

            # Integral over the diagonal segment within cell (k_0, k_0)
            cell_integral = (1 / n) * (
                sum_box + 0.5 * (sum_col_above + sum_row_left) + (1 / 3) * diag_element
            )
            total_integral_diag += cell_integral

        return 6 * total_integral_diag - 2

    def ginis_gamma(self) -> float:
        """
        Compute Gini's Gamma for a bivariate checkerboard copula.

        This method correctly implements the analytical integrals of C(u,u) and C(u,1-u).
        It is implemented for square checkerboard matrices.

        Returns:
            float: The value of Gini's Gamma.
        """
        if self.m != self.n:
            warnings.warn(
                "Gini's Gamma analytical formula is implemented for square matrices only."
            )
            return np.nan

        p = np.asarray(self.matr, dtype=float)
        n = self.m

        # 1. Calculate the integral over the main diagonal C(u,u)
        total_integral_diag = (self.footrule() + 2) / 6

        # 2. Calculate the integral over the anti-diagonal C(u, 1-u)
        total_integral_antidiag = 0
        # Loop through each cell (k, n-1-k) that the anti-diagonal passes through
        for k_0 in range(n):
            # j_0 = n - 1 - k_0 is the column index for the anti-diagonal cell
            j_0 = n - 1 - k_0

            # Cumulative sum of the box top-left of the anti-diagonal cell
            sum_box = np.sum(p[:k_0, :j_0]) if (k_0 > 0 and j_0 > 0) else 0

            # Sum of elements in the same column, above the anti-diagonal cell
            sum_col_above = np.sum(p[:k_0, j_0]) if k_0 > 0 else 0

            # Sum of elements in the same row, left of the anti-diagonal cell
            sum_row_left = np.sum(p[k_0, :j_0]) if j_0 > 0 else 0

            # The anti-diagonal element itself
            antidiag_element = p[k_0, j_0]

            # Integral over the anti-diagonal segment within cell (k_0, j_0)
            cell_integral = (1 / n) * (
                sum_box
                + 0.5 * (sum_col_above + sum_row_left)
                + (1 / 3) * antidiag_element
            )
            total_integral_antidiag += cell_integral

        return 4 * total_integral_diag + 4 * total_integral_antidiag - 2


if __name__ == "__main__":
    matr = [[4, 0, 0], [0, 1, 3], [0, 3, 1]]
    # matr = [[1,0], [0, 1]]
    ccop = BivCheckPi(matr)
    # ccop.plot_c_over_u()
    # ccop.plot_cond_distr_1()
    xi = ccop.xi()
    rho = ccop.rho()
    footrule = ccop.footrule()
    gini = ccop.ginis_gamma()
    beta = ccop.blomqvists_beta()
    # ccop.plot_cdf()
    # ccop.plot_pdf()
    print(
        f"xi = {xi:.3f}, rho = {rho:.3f}, footrule = {footrule:.3f}, gini = {gini:.3f}, beta = {beta:.3f}"
    )
