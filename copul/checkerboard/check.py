import mfoci

import numpy as np
import sympy


class Check:
    params = []
    intervals = {}

    def __init__(self, matr):
        """
        Initialize a checkerboard (piecewise-uniform) copula with the given weight matrix.

        Parameters
        ----------
        matr : array-like
            d-dimensional array of nonnegative weights (one per cell).
            They will be normalized so that the total sum is 1.
        """
        if isinstance(matr, list):
            matr = np.array(matr)

        # Normalize the matrix so it sums to 1
        matr_sum = sum(matr) if isinstance(matr, sympy.Matrix) else matr.sum()
        self.matr = matr / matr_sum

        # Store shape and dimension
        self.dim = len(self.matr.shape)

    def __str__(self):
        return f"CheckerboardCopula({self.matr.shape})"

    def lambda_L(self):
        """Lower tail dependence (usually 0 for a checkerboard copula)."""
        return 0

    def lambda_U(self):
        """Upper tail dependence (usually 0 for a checkerboard copula)."""
        return 0

    def chatterjees_xi(self, n=100_000, seed=None):
        samples = self.rvs(n, random_state=seed)
        x = samples[:, 0]
        z = samples[:, 1:3]
        xi = mfoci.codec(x, z)
        return xi
