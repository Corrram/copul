import sympy

from copul.families.bivcopula import BivCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Raftery(BivCopula):
    """
    Raftery Copula.

    This copula has a parameter delta controlling the dependence structure.
    Special cases:
    - delta = 0: Independence copula
    - delta = 1: Upper Fréchet bound (perfect positive dependence)

    Parameters:
    -----------
    delta : float, 0 ≤ delta ≤ 1
        Dependence parameter
    """

    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sympy.symbols("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, 1, left_open=False, right_open=False)}

    def __init__(self, *args, **kwargs):
        """Initialize the Raftery copula with parameter validation."""
        if args and len(args) == 1:
            kwargs["delta"] = args[0]

        if "delta" in kwargs:
            # Validate delta parameter
            delta_val = kwargs["delta"]
            if delta_val < 0 or delta_val > 1:
                raise ValueError(
                    f"Parameter delta must be between 0 and 1, got {delta_val}"
                )

            # Handle special cases before passing to parent class
            if delta_val == 0:
                self._independence = True
            elif delta_val == 1:
                self._upper_frechet = True
            else:
                self._independence = False
                self._upper_frechet = False
        else:
            self._independence = False
            self._upper_frechet = False

        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        """Handle special cases when calling the instance."""
        if "delta" in kwargs:
            # Validate delta parameter
            delta_val = kwargs["delta"]
            if delta_val < 0 or delta_val > 1:
                raise ValueError(
                    f"Parameter delta must be between 0 and 1, got {delta_val}"
                )

            # Special cases
            if delta_val == 0:
                del kwargs["delta"]
                return IndependenceCopula()(**kwargs)
            if delta_val == 1:
                del kwargs["delta"]
                return UpperFrechet()(**kwargs)

        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def cdf(self):
        """
        Cumulative distribution function of the copula.

        The formula has special cases for delta=0 and delta=1 to avoid division by zero.
        """
        u = self.u
        v = self.v
        d = self.delta

        # Handle special cases to avoid division by zero
        if hasattr(self, "_independence") and self._independence:
            # delta = 0: Independence copula
            return SymPyFuncWrapper(u * v)

        if hasattr(self, "_upper_frechet") and self._upper_frechet:
            # delta = 1: Upper Fréchet bound
            return SymPyFuncWrapper(sympy.Min(u, v))

        # Regular case: 0 < delta < 1
        cdf_expr = sympy.Min(u, v) + (1 - d) / (1 + d) * (u * v) ** (1 / (1 - d)) * (
            1 - sympy.Max(u, v) ** (-(1 + d) / (1 - d))
        )

        return SymPyFuncWrapper(cdf_expr)

    @property
    def pdf(self):
        """
        Probability density function of the copula.

        Calculated using the _b function.
        """
        if hasattr(self, "_independence") and self._independence:
            # delta = 0: Independence copula has uniform density = 1
            return SymPyFuncWrapper(1)

        if hasattr(self, "_upper_frechet") and self._upper_frechet:
            # delta = 1: Upper Fréchet bound doesn't have a proper PDF
            # Return a singularity along the diagonal
            return SymPyFuncWrapper(sympy.DiracDelta(self.u - self.v))

        pdf = self._b(sympy.Min(self.u, self.v), sympy.Max(self.u, self.v))
        return SymPyFuncWrapper(pdf)

    def _b(self, u, v):
        """Helper function for calculating the PDF."""
        delta = self.delta
        return (
            (1 - delta**2) ** (-1)
            * u ** (delta / (1 - delta))
            * (delta * v ** (-1 / (1 - delta)) + v ** (delta / (1 - delta)))
        )

    def spearmans_rho(self, *args, **kwargs):
        """
        Calculate Spearman's rho for the Raftery copula.

        For Raftery, rho = delta * (4 - 3*delta) / (2 - delta)^2
        """
        self._set_params(args, kwargs)
        return self.delta * (4 - 3 * self.delta) / (2 - self.delta) ** 2

    def kendalls_tau(self, *args, **kwargs):
        """
        Calculate Kendall's tau for the Raftery copula.

        For Raftery, tau = 2*delta / (3 - delta)
        """
        self._set_params(args, kwargs)
        return 2 * self.delta / (3 - self.delta)

    @property
    def lambda_L(self):
        """
        Lower tail dependence coefficient.

        For Raftery, lambda_L = 2*delta / (1 + delta)
        """
        return 2 * self.delta / (1 + self.delta)

    @property
    def lambda_U(self):
        """
        Upper tail dependence coefficient.

        For Raftery, lambda_U = 0
        """
        return 0

    def _squared_cond_distr_1(self, u, v):
        """Helper method for squared conditional distribution."""
        delta = self.delta

        # Handle special cases
        if delta == 0:
            return 0  # Independence case
        if delta == 1:
            # Upper Fréchet case
            return sympy.Piecewise((1, u <= v), (0, True))

        term1 = (
            u
            * (u * v) ** (1 / (delta - 1))
            * (delta + 1)
            * sympy.Heaviside(-u + v)
            * sympy.Max(u, v)
        )
        term2 = (
            u
            * (delta + 1)
            * sympy.Heaviside(u - v)
            * sympy.Max(u, v) ** ((delta + 1) / (delta - 1))
        )
        term3 = (1 - sympy.Max(u, v) ** ((delta + 1) / (delta - 1))) * sympy.Max(u, v)
        full_expr = (term1 + term2 + term3) ** 2 / (
            u**2
            * (u * v) ** (2 / (delta - 1))
            * (delta + 1) ** 2
            * sympy.Max(u, v) ** 2
        )
        return full_expr

    def _xi_int_1(self, v):
        """Helper method for Chatterjee's xi calculation."""
        delta = self.delta

        # Handle special cases
        if delta == 0:
            return 0  # Independence case
        if delta == 1:
            return 1  # Upper Fréchet case

        u = self.u
        term1 = u * (u * v) ** (1 / (delta - 1)) * (delta + 1) * v
        term3 = (1 - v ** ((delta + 1) / (delta - 1))) * v
        func_u_lower_v = sympy.simplify(
            (term1 + term3) ** 2
            / (u**2 * (u * v) ** (2 / (delta - 1)) * (delta + 1) ** 2 * v**2)
        )
        term2 = u * (delta + 1) * u ** ((delta + 1) / (delta - 1))
        term3 = (1 - u ** ((delta + 1) / (delta - 1))) * u
        func_u_greater_v = sympy.simplify(
            (term2 + term3) ** 2
            / (u**2 * (u * v) ** (2 / (delta - 1)) * (delta + 1) ** 2 * u**2)
        )

        try:
            int2 = sympy.simplify(sympy.integrate(func_u_greater_v, (u, v, 1)))
            int1 = sympy.simplify(sympy.integrate(func_u_lower_v, (u, 0, v)))
            return sympy.simplify(int1 + int2)
        except Exception:
            # If integration fails, return a placeholder
            return sympy.symbols("int_result")
