import numpy as np

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.check_min import CheckMin


class BivCheckMin(BivCheckPi, CheckMin):
    """Bivariate Checkerboard Minimum class.

    A class that implements bivariate checkerboard minimum operations.
    """

    def __init__(self, matr: np.ndarray, mc_size: int = 200_000, **kwargs) -> None:
        """Initialize the BivCheckMin instance.

        Args:
            matr: Input matrix
            mc_size: Monte Carlo simulation size
            **kwargs: Additional keyword arguments
        """
        super().__init__(matr, **kwargs)
        self.m: int = self.matr.shape[0]
        self.n: int = self.matr.shape[1]
        self.n_samples: int = mc_size

    def __str__(self) -> str:
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
