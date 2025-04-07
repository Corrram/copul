import numpy as np

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.check_min import CheckMin
from copul.exceptions import PropertyUnavailableException


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
        return BivCheckPi.rho(self) + 1 / (self.m * self.n)

    def tau(self) -> float:
        return BivCheckPi.tau(self) + np.trace(self.matr.T @ self.matr)

    def xi(
        self,
        condition_on_y: bool = False,
    ) -> float:
        m, n = (self.n, self.m) if condition_on_y else (self.m, self.n)
        return super().xi(condition_on_y) + m * np.trace(self.matr.T @ self.matr) / n

        
    def lambda_L(self):
        return self.matr[0, 0] * np.min(self.m, self.n)
    
    def lambda_U(self):
        return self.matr[-1, -1] * np.min(self.m, self.n)
    
if __name__ == "__main__":
    ccop = BivCheckMin([[1, 2], [2, 1]])
    ccop.plot_cdf()
