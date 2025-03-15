import numpy as np
import sympy
from scipy import integrate

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Frank(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta_interval = sympy.Interval(-np.inf, np.inf, left_open=True, right_open=True)
    # Define special cases
    special_cases = {0: IndependenceCopula}

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
    def cdf(self):
        """Cumulative distribution function of the copula"""
        theta = self.theta
        u = self.u
        v = self.v
        cdf = (
            -1
            / theta
            * sympy.log(
                1
                + (sympy.exp(-theta * u) - 1)
                * (sympy.exp(-theta * v) - 1)
                / (sympy.exp(-theta) - 1)
            )
        )
        return SymPyFuncWrapper(cdf)

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

    def spearmans_rho(self, *args, **kwargs):
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

    def kendalls_tau(self, *args, **kwargs):
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


Nelsen5 = Frank

# B3 = Frank
