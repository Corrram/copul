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

        The probability mass in cell (i, j) is placed on the line
            U = (i + t)/m,  V = (j + t)/n   for t in [0,1].
        """
        matr = np.asarray(self.matr, dtype=float)
        total_mass = matr.sum()
        if total_mass <= 0:
            return 0.0  # or raise an error if needed

        # Normalize to get p_{i,j}
        p = matr / total_mass
        m, n = p.shape

        # Compute E[U V]
        E_UV = 0.0
        for i in range(m):
            for j in range(n):
                pij = p[i, j]
                if pij > 0:
                    # Integral in the (comonotonic) block (i,j)
                    val_ij = i*j + 0.5*(i + j) + 1.0/3.0
                    E_UV += pij * (val_ij / (m * n))

        # Spearman's rho
        return 12.0 * E_UV - 3.0
    
    def tau(self) -> float:
        tau_checkpi = super().tau()
        extra = np.sum(self.matr**2)
        tau = tau_checkpi + extra
        return tau

    def chatterjees_xi(
        self,
        condition_on_y: bool = False,
    ) -> float:
        p = self.matr
        m, n = p.shape
        uv_sum = 0.0
        for i in range(m):
            for j in range(n):
                if condition_on_y:
                    prior = sum(p[:i, j])/sum(p[:, j])
                    current = p[i, j]/sum(p[:, j])
                else:
                    prior = sum(p[i, :j])/sum(p[i, :])
                    current = p[i, j]/sum(p[i, :])
                uv_sum += prior + current/2

        xi = 6* uv_sum/(m*n) - 2
        return xi


if __name__ == "__main__":
    ccop = BivCheckMin([[1, 2], [2, 1]])
    ccop.plot_cdf()
