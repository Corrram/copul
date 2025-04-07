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
        Compute Spearman's rho for a bivariate checkerboard (Bernstein) copula.
        Formula:  rho = 12/( (m+1)*(n+1) ) * sum_{i,j} D_{i,j} - 3
        where D = self._cumsum_theta() is the cumulated checkerboard matrix.
        """
        # D is the m x n "cumulated" checkerboard matrix
        d = self._cumsum_theta()
        trace_sum = np.sum(d)  # sum of all D_{i,j}
        factor = 12.0 / ((self.m+1) * (self.n+1))
        rho_val = factor * trace_sum - 3.0
        return rho_val

    def tau(self) -> float:
        """
        Calculate Kendall's tau using the matrix formula:
            tau = 1 - trace( Theta^(m) * D * Theta^(n) * D^T ).
        """
        d = self._cumsum_theta()
        theta_m = self._construct_theta(self.m)
        theta_n = self._construct_theta(self.n)
        tau_val = 1.0 - np.trace(theta_m @ d @ theta_n @ d.T)
        return tau_val

    def xi(self, condition_on_y: bool = False) -> float:
        """
        Calculate Chatterjee's xi using the matrix-trace formula:
            xi = 6 * trace( Omega^(m) * D * Lambda^(n) * D^T ) - 2,
        where:
            D = self._cumsum_theta(),
            Omega^(m) captures integrals of partial derivatives of Bernstein polynomials,
            Lambda^(n) captures integrals of Bernstein polynomials themselves.
        """
        d = self._cumsum_theta()  # shape (m, n)
        Omega = self._construct_omega(self.m)  # shape (m, m)
        Lambda = self._construct_lambda(self.n)  # shape (n, n)
        xi_val = 6.0 * np.trace(Omega @ d @ Lambda @ d.T) - 2.0
        return xi_val

    @staticmethod
    def _construct_theta(m: int) -> np.ndarray:
        """
        Construct the m x m matrix Theta^(m) with entries:
          Theta[i,j] = ( (i+1) - (j+1) ) * C(m, i+1) * C(m, j+1)
                       / [ (2m - (i+1) - (j+1)) * C(2m-1, (i+1) + (j+1) - 1 ) ]
        for i,j in {0,...,m-1}.
        """
        Theta = np.zeros((m, m), dtype=float)
        for i in range(1, m+1):      # i from 1..m
            for j in range(1, m+1):  # j from 1..m
                numerator = (i - j) * math.comb(m, i) * math.comb(m, j)
                denom = (2*m - i - j) * math.comb(2*m - 1, i + j - 1)
                if denom == 0:
                    # By convention 0/0 = 1, or handle as needed
                    Theta[i-1, j-1] = 0.0 if (numerator != 0) else 1.0
                else:
                    Theta[i-1, j-1] = numerator / denom
        return Theta

    @staticmethod
    def _construct_omega(m: int) -> np.ndarray:
        """
        Construct the m x m matrix Omega^(m) via the final, piecewise closed-form:

        Omega_{i,r} =
        1) if 1 <= i < m and 1 <= r < m:
           binom(m,i)*binom(m,r)
           * [ ( (i+r-2)! * (2m-(i+r)-2)! ) / ( (2m-1)! ) ]
           * [ (2m-1)(2m-2)*i*r  -  m(m-1)*(i+r)*((i+r)-1) ]

        2) if 1 <= i < m, r = m:
           m * binom(m,i)
           * [ ( (m + i -2)! * (m - i -1)! ) / ( (2m-1)! ) ]
           * [ (m-1)*( i - m ) ]

        3) if i = m, 1 <= r < m:
           m * binom(m,r)
           * [ ( (m + r -2)! * (m - r -1)! ) / ( (2m-1)! ) ]
           * [ (m-1)*( r - m ) ]

        4) if i = m, r = m:
           m^2 / (2m - 1)

        where i,r = 1..m in mathematical notation (i.e. Python indices i-1, r-1).
        """

        Omega = np.zeros((m, m), dtype=float)

        # Helper for binomial and factorial
        def binom(n, k):
            return math.comb(n, k)

        def factorial_nonneg(x):
            # Return 0 if x < 0; otherwise factorial(x).
            if x < 0:
                return 0
            return math.factorial(x)

        for i in range(1, m+1):      # i = 1..m
            for r in range(1, m+1):  # r = 1..m

                if i < m and r < m:
                    # CASE 1
                    # (i+r-2)! * (2m-(i+r)-2)!
                    top_fact = factorial_nonneg(i + r - 2) * factorial_nonneg(2*m - (i + r) - 2)
                    denom = factorial_nonneg(2*m - 1)
                    bracket = ((2*m - 1)*(2*m - 2)*i*r) - (m*(m - 1)*(i + r)*((i + r) - 1))
                    val = (
                        binom(m, i)
                        * binom(m, r)
                        * (top_fact / denom if denom != 0 else 0.0)
                        * bracket
                    )

                elif i < m and r == m:
                    # CASE 2
                    # (m + i -2)! * (m - i -1)! / (2m-1)!
                    top_fact = factorial_nonneg(m + i - 2) * factorial_nonneg(m - i - 1)
                    denom = factorial_nonneg(2*m - 1)
                    bracket = (m - 1)*(i - m)
                    val = (
                        m
                        * binom(m, i)
                        * (top_fact / denom if denom != 0 else 0.0)
                        * bracket
                    )

                elif i == m and r < m:
                    # CASE 3
                    # (m + r -2)! * (m - r -1)! / (2m-1)!
                    top_fact = factorial_nonneg(m + r - 2) * factorial_nonneg(m - r - 1)
                    denom = factorial_nonneg(2*m - 1)
                    bracket = (m - 1)*(r - m)
                    val = (
                        m
                        * binom(m, r)
                        * (top_fact / denom if denom != 0 else 0.0)
                        * bracket
                    )

                else:
                    # CASE 4: i = m, r = m
                    val = (m**2) / float(2*m - 1)

                Omega[i-1, r-1] = val

        return Omega

    @staticmethod
    def _construct_lambda(n: int) -> np.ndarray:
        """
        Construct the n x n matrix Lambda^(n), where
          Lambda_{j,s} = int_0^1 B_{j+1,n}(v)*B_{s+1,n}(v) dv
        and using the known Beta-function formula:
          B_{j,n}(v) = C(n,j) v^j(1-v)^{n-j},
          => Lambda^(n)_{j,s} = C(n,j+1)*C(n,s+1)*[(j+1 + s+1)! * (2n-(j+1+s+1))!] / (2n+1)!
        for j,s in {0,...,n-1}.
        """
        Lambda = np.zeros((n, n), dtype=float)
        for j in range(1, n+1):      # j from 1..n
            for s in range(1, n+1):  # s from 1..n
                bin_j = math.comb(n, j)
                bin_s = math.comb(n, s)
                top = math.factorial(j + s) * math.factorial(2*n - (j + s))
                bottom = math.factorial(2*n + 1)
                val = bin_j * bin_s * (top / bottom)
                Lambda[j-1, s-1] = val
        return Lambda
    
    def lambda_L(self):
        """
            Lower tail dependence is zero by 2016 Pfeifer, Tsatedem, Mändle and Girschig - Example 1
        """
        return 0
    
    def lambda_U(self):
        """
            Upper tail dependence is zero by 2016 Pfeifer, Tsatedem, Mändle and Girschig - Example 1
        """
        return 0


BivBernstein: TypeAlias = BivBernsteinCopula
