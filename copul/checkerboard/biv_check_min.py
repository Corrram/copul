import numpy as np

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.check_min import CheckMin
from copul.exceptions import PropertyUnavailableException

from typing import Any

class BivCheckMin(CheckMin, BivCheckPi):
    """Bivariate Checkerboard Minimum class.

    A class that implements bivariate checkerboard minimum operations.
    """

    def __new__(cls, matr, *args, **kwargs):
        """
        Create a new BivCheckMin instance.

        Parameters
        ----------
        matr : array-like
            Matrix of values that determine the copula's distribution.
        *args, **kwargs
            Additional arguments passed to the constructor.

        Returns
        -------
        BivCheckMin
            A BivCheckMin instance.
        """
        # Skip intermediate classes and directly use Check.__new__
        # This avoids Method Resolution Order (MRO) issues with multiple inheritance
        from copul.checkerboard.check import Check

        instance = Check.__new__(cls)
        return instance

    def __init__(self, matr: np.ndarray, **kwargs) -> None:
        """Initialize the BivCheckMin instance.

        Args:
            matr: Input matrix
            **kwargs: Additional keyword arguments
        """
        CheckMin.__init__(self, matr, **kwargs)
        BivCheckPi.__init__(self, matr, **kwargs)

    def __str__(self) -> str:
        """Return string representation of the instance."""
        return f"CheckMin(m={self.m}, n={self.n})"
    
    def __repr__(self) -> str:
        """Return string representation of the instance."""
        return f"CheckMin(m={self.m}, n={self.n})"

    @property
    def is_symmetric(self) -> bool:
        """Check if the matrix is symmetric.

        Returns:
            bool: True if matrix is symmetric, False otherwise
        """
        if self.matr.shape[0] != self.matr.shape[1]:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def is_absolutely_continuous(self) -> bool:
        """Check if the distribution is absolutely continuous.

        Returns:
            bool: Always returns False for checkerboard distributions
        """
        return False

    @property
    def pdf(self):
        """PDF is not available for BivCheckMin.

        Raises:
            PropertyUnavailableException: Always raised, since PDF does not exist for BivCheckMin.
        """
        raise PropertyUnavailableException("PDF does not exist for BivCheckMin.")

    def rho(self) -> float:
        """
        Compute Spearman's rho under the assumption that each block
        (cell) is perfectly positively correlated (comonotonic) rather 
        than uniformly distributed in 2D.
        
        In a cell (i,j) (with i, j = 0, ..., m-1 or n-1) we assume
            U = (i + t)/m   and   V = (j + t)/n,  for t in [0,1].
        Hence,
            E[U*V | cell (i,j)] = [ i*j + (i+j)/2 + 1/3 ]/(m*n).
        The overall expectation is the weighted sum of these over the cells.
        Finally, ρ = 12 * E[U*V] - 3.
        This implementation is fully vectorized.
        """
        matr = np.asarray(self.matr, dtype=float)
        total_mass = matr.sum()
        if total_mass <= 0:
            return 0.0  # or raise an error
        
        # Normalize to obtain cell probabilities p[i,j]
        p = matr / total_mass
        m, n = p.shape
        # Create index arrays for rows and columns (0-based)
        I = np.arange(m).reshape(m, 1)  # shape (m,1)
        J = np.arange(n).reshape(1, n)  # shape (1,n)
        # Compute the closed-form for each cell:
        # E[U*V | cell (i,j)] = [ i*j + (i+j)/2 + 1/3 ]/(m*n)
        cell_val = (I * J) + 0.5 * (I + J) + (1.0 / 3.0)
        E_UV = np.sum(p * (cell_val / (m * n)))
        return 12.0 * E_UV - 3.0
          
    def tau(self) -> float:
        return super().tau() + np.sum(self.matr**2)

    def xi(
        self,
        condition_on_y: bool = False,
    ) -> float:
        m, n = (self.n, self.m) if condition_on_y else (self.m, self.n)
        return super().xi(condition_on_y) + m*np.sum(self.matr**2)/n

if __name__ == "__main__":
    ccop = BivCheckMin([[1, 2], [2, 1]])
    ccop.plot_cdf()
