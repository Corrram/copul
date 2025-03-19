import sympy

from copul.families.core.biv_copula import BivCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.cdf_wrapper import CDFWrapper
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
                return BivIndependenceCopula()(**kwargs)
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
            return CDFWrapper(u * v)

        if hasattr(self, "_upper_frechet") and self._upper_frechet:
            # delta = 1: Upper Fréchet bound
            return CDFWrapper(sympy.Min(u, v))

        # Regular case: 0 < delta < 1
        cdf_expr = sympy.Min(u, v) + (1 - d) / (1 + d) * (u * v) ** (1 / (1 - d)) * (
            1 - sympy.Max(u, v) ** (-(1 + d) / (1 - d))
        )

        return CDFWrapper(cdf_expr)

    def cdf_vectorized(self, u, v):
        """
        Vectorized implementation of the cumulative distribution function for Raftery copula.

        This method evaluates the CDF at multiple points simultaneously, which is more efficient
        than calling the scalar CDF function repeatedly.

        Parameters
        ----------
        u : array_like
            First uniform marginal, should be in [0, 1].
        v : array_like
            Second uniform marginal, should be in [0, 1].

        Returns
        -------
        numpy.ndarray
            The CDF values at the specified points.

        Notes
        -----
        The Raftery copula CDF is:
        C(u,v) = min(u, v) + (1 - delta) / (1 + delta) * (u * v)^(1/(1-delta)) * (1 - max(u, v)^(-(1+delta)/(1-delta)))

        Special cases:
        - When delta = 0, it's the Independence copula (u * v)
        - When delta = 1, it's the Upper Fréchet bound (min(u, v))
        """
        import numpy as np

        # Convert inputs to numpy arrays if they aren't already
        u = np.asarray(u)
        v = np.asarray(v)

        # Ensure inputs are within [0, 1]
        if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
            raise ValueError("Marginals must be in [0, 1]")

        # Handle scalar inputs by broadcasting to the same shape
        if u.ndim == 0 and v.ndim > 0:
            u = np.full_like(v, u.item())
        elif v.ndim == 0 and u.ndim > 0:
            v = np.full_like(u, v.item())

        # Get delta parameter as a float
        delta_val = float(self.delta)

        # Special case for delta = 0 (Independence copula)
        if delta_val == 0 or (hasattr(self, "_independence") and self._independence):
            return u * v

        # Special case for delta = 1 (Upper Fréchet bound)
        if delta_val == 1 or (hasattr(self, "_upper_frechet") and self._upper_frechet):
            return np.minimum(u, v)

        # Regular case: 0 < delta < 1
        # Calculate min(u, v) and max(u, v)
        u_min_v = np.minimum(u, v)
        u_max_v = np.maximum(u, v)

        # Constant terms
        factor = (1 - delta_val) / (1 + delta_val)
        power1 = 1 / (1 - delta_val)
        power2 = -(1 + delta_val) / (1 - delta_val)

        try:
            # Calculate the full expression for all points
            uv_product = u * v
            term1 = np.power(uv_product, power1, where=(uv_product > 0))

            # Handle potential division by zero or negative inputs in max term
            max_term = np.zeros_like(u_max_v)
            valid_max = (u_max_v > 0) & (u_max_v < 1)
            if np.any(valid_max):
                max_term[valid_max] = np.power(u_max_v[valid_max], power2)

            term2 = 1 - max_term

            # Calculate full CDF
            result = u_min_v + factor * term1 * term2

            # Handle edge cases (where u or v is 0 or 1)
            # When u or v is 0, the CDF is 0
            result = np.where((u == 0) | (v == 0), 0, result)

            # When u or v is 1, special case
            # If u = 1, C(u,v) = v
            # If v = 1, C(u,v) = u
            result = np.where(u == 1, v, result)
            result = np.where(v == 1, u, result)

            # Numerical safety: ensure results are in [0, 1]
            result = np.maximum(0, np.minimum(1, result))

            return result

        except Exception as e:
            # Fallback to element-by-element calculation
            import warnings

            warnings.warn(
                f"Error in vectorized CDF calculation: {e}. Using scalar fallback."
            )

            # Initialize result array
            result = np.zeros_like(u)

            for idx in np.ndindex(u.shape):
                try:
                    u_val, v_val = u[idx], v[idx]

                    # Try to calculate exact match to original CDF implementation
                    if u_val == 0 or v_val == 0:
                        result[idx] = 0
                    elif u_val == 1:
                        result[idx] = v_val
                    elif v_val == 1:
                        result[idx] = u_val
                    else:
                        min_val = min(u_val, v_val)
                        max_val = max(u_val, v_val)

                        term1 = (u_val * v_val) ** power1
                        term2 = 1 - max_val**power2

                        result[idx] = min_val + factor * term1 * term2

                except Exception:
                    # If calculation fails, use min(u,v) as safe fallback
                    result[idx] = min(u[idx], v[idx])

            return result

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

    def rho(self, *args, **kwargs):
        """
        Calculate Spearman's rho for the Raftery copula.

        For Raftery, rho = delta * (4 - 3*delta) / (2 - delta)^2
        """
        self._set_params(args, kwargs)
        return self.delta * (4 - 3 * self.delta) / (2 - self.delta) ** 2

    def tau(self, *args, **kwargs):
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
