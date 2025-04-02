import math
import numpy as np
from copul.checkerboard.bernstein import BernsteinCopula
from copul.families.core.copula_sampling_mixin import CopulaSamplingMixin
from copul.families.core.biv_core_copula import BivCoreCopula
from typing import TypeAlias


class BivBernsteinCopula(BernsteinCopula, BivCoreCopula, CopulaSamplingMixin):
    params: list = []
    intervals: dict = {}

    def __init__(self, theta, check_theta=True):
        BernsteinCopula.__init__(self, theta, check_theta)
        BivCoreCopula.__init__(self)
        self.m = self.matr.shape[0]
        self.n = self.matr.shape[1]

    def rho(self) -> float:
        """
        Compute Spearman's rho for a bivariate checkerboard copula.
        """
        return 12/(self.m*self.n) * np.trace(self.theta) - 3

    def tau(self) -> float:
        """
        Calculate the tau coefficient more efficiently using numpy's vectorized operations.

        Returns:
            float: The calculated tau coefficient.
        """
        d = self._cumsum_theta(with_zeros=True)
        theta_m = self._construct_theta(self.m)
        theta_n = self._construct_theta(self.n)
        return 1 - np.trace(theta_m @ d @ theta_n @ d.T)

    @staticmethod
    def _construct_theta(m):
        """
        Construct the m x m matrix Theta^(m) with entries:
        Theta[i,j] = ( (i+1) - (j+1) ) * C(m, i+1) * C(m, j+1 )
                    / [ (2m - (i+1) - (j+1)) * C(2m-1, (i+1) + (j+1) - 1 ) ]
        where i, j go from 0 to m-1 internally (which corresponds to 1..m in the formula).
        """
        Theta = np.zeros((m+1, m+1), dtype=float)
        for i in range(m+1):      # i = 0..m-1 corresponds to i+1 in {1..m}
            for j in range(m+1):  # j = 0..m-1 corresponds to j+1 in {1..m}
                numerator = (i - j) * math.comb(m, i) * math.comb(m, j)
                if i + j == 0:
                    denom = 0
                else:
                    denom = (2*m - i - j) * math.comb(2*m - 1, i + j - 1)
                
                # Check for zero in the denominator just in case
                if denom == 0:
                    # Decide how you want to handle this case; 
                    # here we set to 0 if numerator is also 0, else np.nan
                    Theta[i, j] = 0.0 if numerator == 0 else np.nan
                else:
                    Theta[i, j] = numerator / denom
        
        return Theta

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

BivBernstein: TypeAlias = BivBernsteinCopula