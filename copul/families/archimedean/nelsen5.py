import numpy as np
import sympy
from scipy import integrate
from typing import Optional

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Frank(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta_interval = sympy.Interval(-np.inf, np.inf, left_open=True, right_open=True)
    # Define special cases
    special_cases = {0: BivIndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return -sympy.log(
            (sympy.exp(-self.theta * self.t) - 1) / (sympy.exp(-self.theta) - 1)
        )

    @property
    def _raw_inv_generator(self):
        theta = self.theta
        y = self.y
        return (
            theta + y - sympy.log(-sympy.exp(theta) + sympy.exp(theta + y) + 1)
        ) / theta

    @property
    def _cdf_expr(self):
        """Cumulative distribution function of the copula"""
        theta = self.theta
        u = self.u
        v = self.v
        return (
            -1
            / theta
            * sympy.log(
                1
                + (sympy.exp(-theta * u) - 1)
                * (sympy.exp(-theta * v) - 1)
                / (sympy.exp(-theta) - 1)
            )
        )

    def rvs(
        self, n: int = 1, random_state: Optional[int] = None, approximate: bool = False
    ) -> np.ndarray:
        """
        Generate random samples from the Frank copula using a fast, vectorized algorithm.

        This method uses a numerically stable, closed-form inverse of the conditional
        distribution, allowing for thousands of samples to be generated almost instantly.

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

        # Handle the independence case where theta is close to 0
        if np.isclose(theta_val, 0):
            return w

        u = w[:, 0]
        w2 = w[:, 1]

        # Use the closed-form inverse of the conditional distribution C(v|u) = w2
        # v = (-1/θ) * log( [e⁻θu(w₂-1) - w₂e⁻θ] / [e⁻θu(w₂-1) - w₂] )
        exp_thetau = np.exp(-theta_val * u)
        exp_theta = np.exp(-theta_val)

        # Numerator of the log's argument
        num = exp_thetau * (w2 - 1) - w2 * exp_theta
        # Denominator of the log's argument
        den = exp_thetau * (w2 - 1) - w2

        v = (-1 / theta_val) * np.log(num / den)

        return np.column_stack((u, v))

    def cdf_vectorized(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Vectorized implementation of the cumulative distribution function for the Frank copula.

        This method uses the explicit mathematical formula for the Frank copula, which is
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
        u = np.asarray(u)
        v = np.asarray(v)

        # Handle the independence case where theta is close to 0
        if np.isclose(theta_val, 0):
            return u * v

        # Numerator term
        num = (np.exp(-theta_val * u) - 1) * (np.exp(-theta_val * v) - 1)
        # Denominator term
        den = np.exp(-theta_val) - 1

        # Full expression, with protection against division by zero if den is somehow zero
        # although this is handled by the theta_val check.
        if np.isclose(den, 0):
            return u * v  # Fallback to independence

        result = (-1 / theta_val) * np.log(1 + num / den)
        return result

    def cond_distr_1(self, u=None, v=None):
        expr_u = sympy.exp(-self.theta * self.u)
        expr_v = sympy.exp(-self.theta * self.v) - 1
        expr = sympy.exp(-self.theta) - 1
        cond_distr_1 = expr_v * expr_u / (expr + (-1 + expr_u) * expr_v)
        return SymPyFuncWrapper(cond_distr_1)(u, v)

    def _squared_cond_distr_1(self, v, u):
        theta = self.theta
        return (
            (-1 + sympy.exp(-theta * v)) ** 2
            * sympy.exp(-2 * theta * u)
            / (
                (-1 + sympy.exp(-theta * u)) * (-1 + sympy.exp(-theta * v))
                - 1
                + sympy.exp(-theta)
            )
            ** 2
        )

    def _xi_int_1(self, v):
        theta = self.theta
        return (
            theta * v * sympy.exp(2 * theta * v)
            - theta * v * sympy.exp(2 * theta * (v + 1))
            - 2 * theta * v * sympy.exp(theta * (v + 1))
            + 2 * theta * v * sympy.exp(theta * (v + 2))
            + theta * sympy.exp(2 * theta)
            + theta * sympy.exp(2 * theta * (v + 1))
            - 2 * theta * sympy.exp(theta * (v + 2))
            - sympy.exp(3 * theta * v)
            + sympy.exp(2 * theta * v)
            - sympy.exp(2 * theta * (v + 1))
            - sympy.exp(theta * (v + 1))
            + sympy.exp(theta * (v + 2))
            + sympy.exp(theta * (3 * v + 1))
        ) / (
            theta
            * (
                sympy.exp(2 * theta)
                + sympy.exp(4 * theta * v)
                - 2 * sympy.exp(3 * theta * v)
                + sympy.exp(2 * theta * v)
                + sympy.exp(2 * theta * (v + 1))
                - 2 * sympy.exp(theta * (v + 1))
                - 2 * sympy.exp(theta * (v + 2))
                + 4 * sympy.exp(theta * (2 * v + 1))
                - 2 * sympy.exp(theta * (3 * v + 1))
            )
        )

    def rho(self, *args, **kwargs):
        """
        Calculate Spearman's rho for the Frank copula using numerical integration.

        For theta = 0, returns 0 (independence).
        """
        self._set_params(args, kwargs)
        theta = abs(float(self.theta))

        if theta == 0:
            return 0

        # Use numerical integration for calculating Debye functions
        d1 = self._debye_function(1, theta)
        d2 = self._debye_function(2, theta)

        # For Frank copula, rho = 1 - 12(D₁ - D₂)/θ
        rho = 1 - 12 * (d1 - d2) / theta
        return rho * np.sign(self.theta)

    def tau(self, *args, **kwargs):
        """
        Calculate Kendall's tau for the Frank copula using numerical integration.

        For theta = 0, returns 0 (independence).
        """
        self._set_params(args, kwargs)
        theta = abs(float(self.theta))

        if theta == 0:
            return 0

        # Use numerical integration for calculating Debye function
        d1 = self._debye_function(1, theta)

        # For Frank copula, tau = 1 + 4(D₁ - 1)/θ
        tau = 1 + 4 * (d1 - 1) / theta
        return tau * np.sign(self.theta)

    def _debye_function(self, n, x):
        """
        Calculate the Debye function of order n for parameter x.

        Uses numerical integration for accurate results.

        Args:
            n (int): Order of the Debye function
            x (float): Parameter value

        Returns:
            float: Value of the Debye function
        """
        if x == 0:
            return 1.0  # Limit value

        # Make sure we're working with the right sign
        x_abs = abs(x)

        # Define the integrand for the Debye function
        def integrand(t):
            return t**n / (np.exp(t) - 1)

        # Calculate the integral
        result, _ = integrate.quad(integrand, 0, x_abs)

        # Apply the normalization factor
        debye_value = n / x_abs**n * result
        return debye_value

    def _d_1(self):
        """
        Helper function for Debye function of first order.
        Used in Kendall's tau calculation.
        """
        theta = float(self.theta)
        if theta == 0:
            return 1.0
        return self._debye_function(1, theta)

    def _d_2(self):
        """
        Helper function for Debye function of second order.
        Used in Spearman's rho calculation.
        """
        theta = float(self.theta)
        if theta == 0:
            return 1.0
        return self._debye_function(2, theta)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0

    def blomqvist(self):
        return 4 / self.theta * sympy.log(sympy.cosh(self.theta / 4))


Nelsen5 = Frank

# B3 = Frank

if __name__ == "__main__":
    copula = Nelsen5(theta=4.0)
    print("--- Testing Frank Copula with theta = 4.0 ---")
    print(f"Spearman's rho: {copula.rho():.4f}")
    print(f"Kendall's tau: {copula.tau():.4f}")

    # Test new rvs method
    print("\nTesting fast rvs method:")
    samples = copula.rvs(5, random_state=42)
    print(samples)

    # Test new cdf_vectorized method
    print("\nTesting fast cdf_vectorized method:")
    u_test = np.array([0.2, 0.5, 0.8])
    v_test = np.array([0.3, 0.6, 0.7])
    cdf_vals = copula.cdf_vectorized(u_test, v_test)
    print(f"CDF values for u={u_test}, v={v_test}:")
    print(cdf_vals)

    print("\nDone!")
