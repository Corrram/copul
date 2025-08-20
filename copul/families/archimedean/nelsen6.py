from typing import Optional

import numpy as np
import sympy

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.archimedean.heavy_compute_arch import HeavyComputeArch
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Joe(BivArchimedeanCopula):
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)
    special_cases = {1: BivIndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return -sympy.log(1 - (1 - self.t) ** self.theta)

    @property
    def _raw_inv_generator(self):
        return 1 - (1 - sympy.exp(-self.y)) ** (1 / self.theta)

    @property
    def _cdf_expr(self):
        theta = self.theta
        return 1 - (-((1 - self.u) ** theta - 1) * ((1 - self.v) ** theta - 1) + 1) ** (
            1 / theta
        )

    def rvs(
        self, n: int = 1, random_state: Optional[int] = None, approximate: bool = False
    ) -> np.ndarray:
        """
        Generate random samples from the Joe copula using a fast, vectorized algorithm.

        This method overrides the slow, iterative solver from the parent class. It uses a
        numerically stable, closed-form inverse of the conditional distribution, allowing
        for thousands of samples to be generated almost instantly.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        random_state : int, optional
            Seed for the random number generator for reproducibility.
        approximate : bool
            This parameter is ignored as the exact vectorized method is always fast.

        Returns
        -------
        numpy.ndarray
            Array of shape (n, 2) containing the generated samples.
        """
        rng = np.random.default_rng(random_state)
        w = rng.random((n, 2))

        theta_val = float(self.theta)

        # Handle the independence case
        if np.isclose(theta_val, 1):
            return w

        v = w[:, 1]

        # Use the closed-form inverse of the conditional distribution C(u|v)
        # This is a highly efficient and numerically stable algorithm
        term1 = (1 - v) ** (-theta_val) - 1
        term2 = w[:, 0] ** (-theta_val / (theta_val - 1)) - 1
        u = 1 - (1 + term1 * (1 + term2)) ** (-1 / theta_val)

        return np.column_stack((u, v))

    def cdf_vectorized(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Vectorized implementation of the cumulative distribution function for the Joe copula.

        This method uses the explicit mathematical formula for the Joe copula, which is
        significantly faster than the generic generator-based approach.

        Parameters
        ----------
        u : array_like
            First uniform marginal, must be in [0, 1].
        v : array_like
            Second uniform marginal, must be in [0, 1].

        Returns
        -------
        numpy.ndarray
            The CDF values at the specified points.
        """
        theta_val = float(self.theta)

        # Handle the independence case
        if np.isclose(theta_val, 1):
            return np.asarray(u) * np.asarray(v)

        # Convert inputs to numpy arrays for vectorized operations
        u = np.asarray(u)
        v = np.asarray(v)

        # Initialize result array. The default of 0 correctly handles C(0,v) and C(u,0).
        result = np.zeros_like(u, dtype=float)

        # Identify points that require the full computation (not on the boundaries)
        interior_mask = (u > 0) & (u < 1) & (v > 0) & (v < 1)

        if np.any(interior_mask):
            u_int, v_int = u[interior_mask], v[interior_mask]

            # Use the standard formula for the Joe copula for better numerical stability
            # C(u,v) = 1 - [ (1-u)^θ + (1-v)^θ - (1-u)^θ * (1-v)^θ ]^(1/θ)
            term_u = (1 - u_int) ** theta_val
            term_v = (1 - v_int) ** theta_val
            base = term_u + term_v - term_u * term_v
            result[interior_mask] = 1 - base ** (1 / theta_val)

        # Handle the u=1 and v=1 boundary cases using the mask
        result[u == 1] = v[u == 1]
        result[v == 1] = u[v == 1]

        return result

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        cond_distr_1 = (
            -((1 - self.u) ** theta)
            * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1)
            ** (1 / theta)
            * ((1 - self.v) ** theta - 1)
            / (
                (1 - self.u)
                * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1)
            )
        )
        return SymPyFuncWrapper(cond_distr_1)(u, v)

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        cond_distr_2 = (
            (1 - self.v) ** theta
            * (1 - (1 - self.u) ** theta)
            * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1)
            ** (1 / theta)
            / (
                (1 - self.v)
                * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1)
            )
        )
        return SymPyFuncWrapper(cond_distr_2)(u, v)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)


Nelsen6 = Joe

# B5 = Joe


if __name__ == "__main__":
    copula = Nelsen6(theta=2)
    print(copula.rvs(5))
    for i in range(1000):
        copula.rvs(1, approximate=False)
    print(copula.cdf(0.5, 0.5))
    print(copula.cond_distr_1(0.5, 0.5))
    print(copula.cond_distr_2(0.5, 0.5))
    print(copula.lambda_L())
    print(copula.lambda_U())
