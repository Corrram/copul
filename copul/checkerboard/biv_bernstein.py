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
        Construct the m x m matrix Omega^(m) by explicitly treating the four subcases:
        (i < m, r < m), (i < m, r = m), (i = m, r < m), (i = m, r = m).
        Indices i, r go from 1..m in mathematical notation, corresponding to 0..(m-1) in Python.

        For 1 <= i < m, we have:
            partial_1 B_{i,m}(u) = comb(m,i) * [i - m*u] * u^(i-1) * (1-u)^(m-i-1)
        For i = m, we have:
            partial_1 B_{m,m}(u) = m * u^(m-1).

        Then,
        Omega_{i,r} = ∫ [∂_1 B_{i,m}(u)] [∂_1 B_{r,m}(u)] du
        is split according to the four subcases.

        This piecewise definition matches the Beta-integral approach from the proof.
        """

        Omega = np.zeros((m, m), dtype=float)

        # Small helper for factorial-based Beta integrals:
        #     Beta(p, q) = ∫_0^1 x^p (1-x)^q dx = p! q! / (p+q+1)!
        # returns 0 if p<0 or q<0.
        def beta_int(p: int, q: int) -> float:
            if p < 0 or q < 0:
                return 0.0
            return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 1)

        # Binomial coefficient comb(m, i)
        def binom(m_: int, k: int) -> int:
            return math.comb(m_, k)

        for i in range(1, m + 1):     # i = 1..m
            for r in range(1, m + 1): # r = 1..m

                if i < m and r < m:
                    # CASE (a): 1 <= i < m, 1 <= r < m
                    # partial_1 B_{i,m}(u) * partial_1 B_{r,m}(u)
                    # = comb(m,i)*comb(m,r)* [ (i - m*u)(r - m*u) ] * u^(i+r-2)*(1-u)^(2m-(i+r)-2)
                    # Expand (i - m*u)(r - m*u) => i*r - m(i+r)u + m^2 u^2
                    # => 3 Beta integrals
                    bin_i = binom(m, i)
                    bin_r = binom(m, r)
                    # Term 1: i*r * Beta(i+r-2, 2m-(i+r)-2)
                    t1 = i * r * beta_int(i + r - 2, 2*m - (i + r) - 2)
                    # Term 2: m*(i+r) * Beta(i+r-1, 2m-(i+r)-2)
                    t2 = m * (i + r) * beta_int(i + r - 1, 2*m - (i + r) - 2)
                    # Term 3: m^2 * Beta(i+r, 2m-(i+r)-2)
                    t3 = (m**2) * beta_int(i + r, 2*m - (i + r) - 2)
                    val = bin_i * bin_r * (t1 - t2 + t3)

                elif i < m and r == m:
                    # CASE (b): 1 <= i < m, r = m
                    # partial_1 B_{i,m}(u) = comb(m,i)*(i - m*u)*u^(i-1)*(1-u)^(m-i-1)
                    # partial_1 B_{m,m}(u) = m * u^(m-1)
                    # => 2-term integral:
                    #
                    #  Omega_{i,m} = m*comb(m,i) * ∫[ (i - m*u)* u^( (i-1)+(m-1) ) (1-u)^( (m-i-1) ) ] du
                    #              = m*comb(m,i)* [ i * Beta(m+i-2, m-i-1) - m * Beta(m+i-1, m-i-1) ]
                    bin_i = binom(m, i)
                    tA = i * beta_int((m + i - 2), (m - i - 1))   # i * Beta(m+i-2, m-i-1)
                    tB = m * beta_int((m + i - 1), (m - i - 1))   # m * Beta(m+i-1, m-i-1)
                    val = m * bin_i * (tA - tB)

                elif i == m and r < m:
                    # CASE (c): i = m, 1 <= r < m
                    # By symmetry, just swap i<->r in the formula of case (b).
                    bin_r = binom(m, r)
                    tA = r * beta_int((m + r - 2), (m - r - 1))
                    tB = m * beta_int((m + r - 1), (m - r - 1))
                    val = m * bin_r * (tA - tB)

                else:
                    # CASE (d): i = m, r = m
                    # partial_1 B_{m,m}(u) = m*u^(m-1)
                    # => Omega_{m,m} = ∫ [m*u^(m-1)]^2 du = m^2 ∫ u^(2m-2) du = m^2/(2m-1).
                    val = (m**2) / float(2*m - 1)

                Omega[i - 1, r - 1] = val

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


BivBernstein: TypeAlias = BivBernsteinCopula
