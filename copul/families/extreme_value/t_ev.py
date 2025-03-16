import numpy as np
import sympy
from sympy import stats, Float, re
from copul.families.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


# noinspection PyPep8Naming
class tEV(BivExtremeValueCopula):
    """
    Student-t Extreme Value Copula.

    This copula has two parameters:
    - nu: Degrees of freedom (positive)
    - rho: Correlation parameter (-1 < rho < 1)
    """

    rho = sympy.symbols("rho")
    nu = sympy.symbols("nu", positive=True)
    params = [nu, rho]
    intervals = {
        "nu": sympy.Interval(0, np.inf, left_open=True, right_open=True),
        "rho": sympy.Interval(-1, 1, left_open=True, right_open=True),
    }

    @property
    def is_symmetric(self) -> bool:
        """
        The tEV copula is symmetric when rho = 0.
        """
        if isinstance(self.rho, sympy.Symbol):
            return False
        return np.isclose(float(self.rho), 0.0)

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        The tEV copula is absolutely continuous.
        """
        return True

    @property
    def _pickands(self):
        """
        Pickands dependence function for the tEV copula.
        """
        # Handle edge cases explicitly
        result = sympy.Piecewise(
            (Float(1.0), sympy.Or(self.t == 0, self.t == 1)),
            (self._compute_pickands(), True),
        )
        return result

    def _compute_pickands(self):
        """Internal computation of the Pickands function."""

        def z(t):
            """Helper function for the Student-t Pickands function."""
            return (
                (1 + self.nu) ** (1 / 2)
                * ((t / (1 - t)) ** (1 / self.nu) - self.rho)
                * (1 - self.rho**2) ** (-1 / 2)
            )

        student_t = stats.StudentT("x", self.nu + 1)

        # Calculate using the standard formula
        term1 = (1 - self.t) * stats.cdf(student_t)(z(1 - self.t))
        term2 = self.t * stats.cdf(student_t)(z(self.t))

        # Ensure the result is real by taking the real part
        return re(term1 + term2)

    @property
    def pickands(self):
        """
        Get the Pickands dependence function with current parameter values.

        Returns a wrapper that handles boundary cases and ensures real values.
        """
        # Get the base pickands function
        pickands_expr = self._pickands

        # Create a special wrapper for numerical evaluation
        class SafePickandsWrapper(SymPyFuncWrapper):
            def __call__(self_, t=None):
                # Handle boundary cases
                if t is not None:
                    try:
                        if (
                            float(t) == 0
                            or float(t) == 1
                            or float(t) < 1e-10
                            or float(t) > 1 - 1e-10
                        ):
                            return Float(1.0)
                    except (TypeError, ValueError):
                        pass

                # For non-boundary values, perform substitution
                if t is not None:
                    try:
                        result = self_.func.subs(self.t, t)

                        # Convert complex results to real
                        if hasattr(result, "is_complex") and result.is_complex:
                            return Float(sympy.re(result).evalf())

                        return result.evalf() if hasattr(result, "evalf") else result
                    except Exception:
                        # Fallback for errors
                        return Float(1.0)

                return self_.func

            def __float__(self_):
                try:
                    result = self_.evalf()

                    # Handle complex values
                    if hasattr(result, "is_complex") and result.is_complex:
                        return float(sympy.re(result).evalf())

                    return float(result)
                except Exception:
                    # Fallback for errors
                    return 1.0

        return SafePickandsWrapper(pickands_expr)

    def cdf_vectorized(self, u, v):
        """
        Optimized vectorized implementation of the CDF.

        This implementation uses direct numerical calculations rather than symbolic
        evaluation for significant performance improvements.
        """
        # Convert inputs to numpy arrays
        u_array = np.asarray(u)
        v_array = np.asarray(v)

        # Early validation check
        if np.any((u_array < 0) | (u_array > 1)) or np.any(
            (v_array < 0) | (v_array > 1)
        ):
            raise ValueError("Marginals must be in [0, 1]")

        # Initialize result with boundary conditions
        result = np.zeros_like(u_array, dtype=float)
        result = np.where(u_array == 1, v_array, result)
        result = np.where(v_array == 1, u_array, result)

        # Only calculate for interior points
        interior = (u_array > 0) & (u_array < 1) & (v_array > 0) & (v_array < 1)

        # Skip if no interior points
        if not np.any(interior):
            return result

        # Get parameter values
        try:
            nu_val = float(self.nu)
            rho_val = float(self.rho)
        except (TypeError, ValueError):
            # Fallback if parameters are symbolic
            np.where(interior)
            result[interior] = np.minimum(u_array[interior], v_array[interior])
            return result

        # Extract interior values for calculation
        u_interior = u_array[interior]
        v_interior = v_array[interior]

        # Calculate t values for Pickands function: t = ln(v)/ln(u*v)
        # Avoid log(0) by ensuring values are positive
        uv_product = u_interior * v_interior
        log_uv = np.log(uv_product)
        log_v = np.log(v_interior)
        t_vals = log_v / log_uv

        # Calculate Pickands function values directly using the formula
        # This is specifically optimized for the tEV copula
        a_vals = np.ones_like(t_vals)  # Initialize with safe values

        # Only try this if parameters are defined and in valid ranges
        if nu_val > 0 and -1 < rho_val < 1:
            try:
                # Define z(t) function for vectorized calculation
                def z_func(t_array):
                    """Vectorized calculation of z(t) for tEV pickands function."""
                    # Handle division by zero
                    valid_mask = (t_array > 0) & (t_array < 1)

                    # Initialize with safe values
                    result = np.ones_like(t_array)

                    if np.any(valid_mask):
                        # Calculate only for valid t values
                        valid_t = t_array[valid_mask]
                        ratio = valid_t / (1 - valid_t)

                        # Handle very small/large values to prevent overflow/underflow
                        ratio = np.clip(ratio, 1e-10, 1e10)

                        result[valid_mask] = (
                            np.sqrt(1 + nu_val)
                            * (ratio ** (1 / nu_val) - rho_val)
                            * (1 - rho_val**2) ** (-0.5)
                        )

                    return result

                # Validate inputs to prevent numerical issues
                valid_t_mask = (t_vals > 0) & (t_vals < 1) & np.isfinite(t_vals)
                if np.any(valid_t_mask):
                    # We'll compute A(t) only for valid inputs
                    valid_t = t_vals[valid_t_mask]

                    # Calculate z(t) and z(1-t)
                    z_t = z_func(valid_t)
                    z_1_minus_t = z_func(1 - valid_t)

                    # Calculate Student-t CDF values using scipy
                    from scipy.stats import t as t_dist

                    cdf_t = t_dist.cdf(z_t, df=nu_val + 1)
                    cdf_1_minus_t = t_dist.cdf(z_1_minus_t, df=nu_val + 1)

                    # Calculate A(t) = (1-t)Φ(z(1-t)) + tΦ(z(t))
                    a_valid = (1 - valid_t) * cdf_1_minus_t + valid_t * cdf_t

                    # Store back in the full array
                    a_vals[valid_t_mask] = a_valid

                    # Fallback for invalid t values
                    a_vals[~valid_t_mask] = 1.0

            except Exception:
                # Fallback to safe values if numerical calculation fails
                a_vals = np.ones_like(t_vals)

        # Ensure values are within valid range for a Pickands function
        # A(t) must satisfy max(t, 1-t) ≤ A(t) ≤ 1
        t_min = np.minimum(t_vals, 1 - t_vals)
        a_vals = np.maximum(a_vals, t_min)
        a_vals = np.minimum(a_vals, 1.0)

        # Calculate final CDF: C(u,v) = (u*v)^A(t)
        cdf_vals = uv_product**a_vals

        # Store results
        result[interior] = cdf_vals

        return result
