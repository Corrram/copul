import numpy as np

from copul.checkerboard._biv_mixin import BivCheckerboardMixin
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.check_min import CheckMin
from copul.exceptions import PropertyUnavailableException


class BivCheckMin(CheckMin, BivCheckPi):
    """Bivariate checkerboard copula with comonotone (Min) cell kernels.

    Inside every cell the mass is placed on the rising cell diagonal.  All
    numerics (cdf, conditional distributions, sampling and the closed-form
    dependence measures) are exact, see :mod:`copul.checkerboard._biv_engine`.
    Relative to :class:`BivCheckPi` with the same matrix ``Delta``::

        rho  += 1 / (m n)
        tau  += sum Delta_ij^2
        xi   += (m / n) sum Delta_ij^2
        nu   += sum_i r_i (2m - 2i - 1) / (m^2 n),   r_i = row sums (0-based i)
        lambda_L = Delta_00 min(m, n),  lambda_U = Delta_{m-1,n-1} min(m, n)
    """

    def __new__(cls, matr, *args, **kwargs):
        # Skip intermediate classes and directly use Check.__new__
        # This avoids Method Resolution Order (MRO) issues with multiple inheritance
        from copul.checkerboard.check import Check

        return Check.__new__(cls)

    def __init__(self, matr: list[list[float]] | np.ndarray, **kwargs) -> None:
        """Initialize the BivCheckMin instance.

        Args:
            matr: Input matrix (or another checkerboard copula).
            **kwargs: Additional keyword arguments (ignored).
        """
        BivCheckPi.__init__(self, matr, **kwargs)

    # the bivariate exact engine overrides the d-dimensional CheckMin code
    cdf = BivCheckerboardMixin.cdf
    cond_distr = BivCheckerboardMixin.cond_distr
    rvs = BivCheckerboardMixin.rvs

    def _kernel_signs(self):
        return 1

    def __str__(self) -> str:
        """Return string representation of the instance."""
        return f"CheckMin(m={self.m}, n={self.n})"

    def __repr__(self) -> str:
        """Return string representation of the instance."""
        return f"CheckMin(m={self.m}, n={self.n})"

    @property
    def is_symmetric(self) -> bool:
        """Check if the matrix is symmetric."""
        if self.matr.shape[0] != self.matr.shape[1]:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def is_absolutely_continuous(self) -> bool:
        """Always False: the mass lives on line segments."""
        return False

    @property
    def pdf(self):
        """PDF is not available for BivCheckMin.

        Raises:
            PropertyUnavailableException: Always raised, since PDF does not exist for BivCheckMin.
        """
        raise PropertyUnavailableException("PDF does not exist for BivCheckMin.")


if __name__ == "__main__":
    ccop = BivCheckMin([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(ccop.spearmans_footrule(), ccop.ginis_gamma(), ccop.chatterjees_xi())
