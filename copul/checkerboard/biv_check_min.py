import warnings
from typing import List, Union

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

    def __init__(self, matr: Union[List[List[float]], np.ndarray], **kwargs) -> None:
        """Initialize the BivCheckMin instance.

        Args:
            matr: Input matrix
            **kwargs: Additional keyword arguments
        """
        if isinstance(matr, BivCheckPi):
            matr = matr.matr
        CheckMin.__init__(self, matr, **kwargs)
        BivCheckPi.__init__(self, matr, **kwargs)

    def __str__(self) -> str:
        """Return string representation of the instance."""
        return f"CheckMin(m={self.m}, n={self.n})"

    def __repr__(self) -> str:
        """Return string representation of the instance."""
        return f"CheckMin(m={self.m}, n={self.n})"

    def transpose(self):
        """
        Transpose the checkerboard matrix.
        """
        return BivCheckMin(self.matr.T)

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

    @classmethod
    def generate_randomly(cls, grid_size: int | list | None = None, n=1):
        generated_copulas = BivCheckPi.generate_randomly(grid_size, n)
        if n == 1:
            return cls(generated_copulas)
        else:
            return [cls(copula) for copula in generated_copulas]

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
        check_pi_xi = super().xi(condition_on_y)
        add_on = m * np.trace(self.matr.T @ self.matr) / n
        return check_pi_xi + add_on

    def lambda_L(self):
        return self.matr[0, 0] * np.min(self.m, self.n)

    def lambda_U(self):
        return self.matr[-1, -1] * np.min(self.m, self.n)

    def footrule(self) -> float:
        """
        Compute Spearman's Footrule (psi) for a BivCheckMin copula.

        The value is the footrule of the underlying CheckPi copula plus an
        add-on term accounting for the singular part of the distribution.
        Implemented for square checkerboard matrices.

        Returns:
            float: The value of Spearman's Footrule.
        """
        if self.m != self.n:
            warnings.warn(
                "Footrule analytical formula is implemented for square matrices only."
            )
            return np.nan

        # Calculate footrule for the absolutely continuous part (CheckPi)
        check_pi_footrule = super().footrule()

        # Add-on term from the singular part of the copula
        # Add-on = (1/n) * trace(P)
        trace = np.trace(self.matr)
        add_on = trace / self.m

        return check_pi_footrule + add_on

    def ginis_gamma(self) -> float:
        """
        Compute Gini's Gamma for a BivCheckMin copula.

        This method corrects the value from the parent BivCheckPi class. The
        parent method incorrectly uses the overridden `footrule` method from
        this child class, leading to a "contaminated" result that already
        includes the add-on for the main diagonal integral. We correct this
        by adding only the missing component from the anti-diagonal integral.
        Implemented for square checkerboard matrices.

        Returns:
            float: The value of Gini's Gamma.
        """
        if self.m != self.n:
            warnings.warn(
                "Gini's Gamma analytical formula is implemented for square matrices only."
            )
            return np.nan

        # The super() call returns a value that has incorrectly incorporated the
        # diagonal add-on but not the anti-diagonal add-on.
        contaminated_gamma_pi = super().ginis_gamma()

        # We add only the part that was missing: the add-on for the
        # anti-diagonal integral C(u, 1-u).
        # Add-on = 4 * (Trace(Anti-Diagonal(P)) / (12n))
        anti_diag_trace = np.trace(np.fliplr(self.matr))

        add_on = anti_diag_trace / (3 * self.m)

        return contaminated_gamma_pi + add_on


if __name__ == "__main__":
    matr1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    matr2 = [[5, 1, 5, 1], [5, 1, 5, 1], [1, 5, 1, 5], [1, 5, 1, 5]]
    matr = [[1, 0], [0, 1]]
    matr = [[1, 1]]
    ccop = BivCheckMin(matr).to_checkerboard()
    footrule = ccop.footrule()
    rho = ccop.rho()
    ginis_gamma = ccop.ginis_gamma()
    xi = ccop.xi()
    # ccop.plot_cond_distr_1()
    # ccop.transpose().plot_cond_distr_1()
    is_cis, is_cds = ccop.is_cis()
    transpose_is_cis, transpose_is_cds = ccop.transpose().is_cis()
    print(f"Is cis: {is_cis}, Is cds: {is_cds}")
    print(f"Is cis: {transpose_is_cis}, Is cds: {transpose_is_cds}")
    print(f"Footrule: {footrule}, Gini's Gamma: {ginis_gamma}, xi: {xi}")
