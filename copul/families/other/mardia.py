import copy

import sympy

from copul.exceptions import PropertyUnavailableException
from copul.families.bivcopula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Mardia(BivCopula):
    """
    Mardia Copula.

    A convex mixture of the Fréchet bounds and the independence copula.

    C(u,v) = theta^2 * (1 + theta) / 2 * min(u,v) +
             (1 - theta^2) * u*v +
             theta^2 * (1 - theta) / 2 * max(u+v-1, 0)

    Parameters:
    -----------
    theta : float, -1 ≤ theta ≤ 1
        Dependence parameter
    """

    @property
    def is_symmetric(self) -> bool:
        return True

    theta = sympy.symbols("theta")
    params = [theta]
    intervals = {"theta": sympy.Interval(-1, 1, left_open=False, right_open=False)}

    def __init__(self, *args, **kwargs):
        """Initialize the Mardia copula with parameter validation."""
        if args and len(args) == 1:
            kwargs["theta"] = args[0]

        if "theta" in kwargs:
            # Validate theta parameter
            theta_val = kwargs["theta"]
            if theta_val < -1 or theta_val > 1:
                raise ValueError(
                    f"Parameter theta must be between -1 and 1, got {theta_val}"
                )

            self.theta = kwargs["theta"]
            self.params = [param for param in self.params if str(param) != "theta"]
            del kwargs["theta"]

        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Handle parameter updates when calling the instance."""
        if args and len(args) == 1:
            kwargs["theta"] = args[0]

        if "theta" in kwargs:
            # Validate theta parameter
            theta_val = kwargs["theta"]
            if theta_val < -1 or theta_val > 1:
                raise ValueError(
                    f"Parameter theta must be between -1 and 1, got {theta_val}"
                )

            new_copula = copy.deepcopy(self)
            new_copula.theta = kwargs["theta"]
            new_copula.params = [
                param for param in new_copula.params if str(param) != "theta"
            ]
            del kwargs["theta"]
            return new_copula.__call__(**kwargs)

        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return self.theta == 0 or self.theta == -1

    @property
    def cdf(self):
        """
        Cumulative distribution function of the copula.

        C(u,v) = theta^2 * (1 + theta) / 2 * min(u,v) +
                 (1 - theta^2) * u*v +
                 theta^2 * (1 - theta) / 2 * max(u+v-1, 0)
        """
        frechet_upper = sympy.Min(self.u, self.v)
        frechet_lower = sympy.Max(self.u + self.v - 1, 0)
        theta = self.theta
        theta_sq = theta**2

        # Handle special cases
        if theta == -1:
            # For theta = -1, the formula simplifies to (u*v + max(u+v-1,0))/2
            cdf = (self.u * self.v + frechet_lower) / 2
            return SymPyFuncWrapper(cdf)

        if theta == 0:
            # For theta = 0, it's the independence copula
            return SymPyFuncWrapper(self.u * self.v)

        if theta == 1:
            # For theta = 1, it's the upper Fréchet bound
            return SymPyFuncWrapper(frechet_upper)

        # General case
        cdf = (
            theta_sq * (1 + theta) / 2 * frechet_upper
            + (1 - theta_sq) * self.u * self.v
            + theta_sq * (1 - theta) / 2 * frechet_lower
        )
        return SymPyFuncWrapper(cdf)

    @property
    def lambda_L(self):
        """
        Lower tail dependence coefficient.

        For Mardia, lambda_L = theta^2 * (1 + theta) / 2

        When theta = 1, this equals 1
        When theta = -1, this equals 0
        """
        theta = self.theta
        # Special case for theta = 1
        if theta == 1:
            return 1

        # Regular formula for 0 <= theta < 1
        return theta**2 * (1 + theta) / 2

    @property
    def lambda_U(self):
        """
        Upper tail dependence coefficient.

        For Mardia, lambda_U = theta^2 * (1 + theta) / 2

        When theta = 1, this equals 1
        When theta = -1, this equals 0
        """
        theta = self.theta
        # Special case for theta = 1
        if theta == 1:
            return 1

        # Regular formula for 0 <= theta < 1
        return theta**2 * (1 + theta) / 2

    def chatterjees_xi(self, *args, **kwargs):
        """
        Calculate Chatterjee's xi for the Mardia copula.

        For Mardia, xi = theta^4 * (3*theta^2 + 1) / 4
        """
        self._set_params(args, kwargs)
        return self.theta**4 * (3 * self.theta**2 + 1) / 4

    def spearmans_rho(self, *args, **kwargs):
        """
        Calculate Spearman's rho for the Mardia copula.

        For Mardia, rho = theta^3
        """
        self._set_params(args, kwargs)
        return self.theta**3

    def kendalls_tau(self, *args, **kwargs):
        """
        Calculate Kendall's tau for the Mardia copula.

        For Mardia, tau = theta^3 * (theta^2 + 2) / 3
        """
        self._set_params(args, kwargs)
        return self.theta**3 * (self.theta**2 + 2) / 3

    @property
    def pdf(self):
        """
        Probability density function of the copula.

        The Mardia copula does not have a PDF due to its singular components.
        """
        raise PropertyUnavailableException("Mardia copula does not have a pdf")
