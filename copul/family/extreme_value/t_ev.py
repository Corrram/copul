import numpy as np
import sympy
from sympy import stats, Float, re
from scipy.stats import t as t_dist
from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


# noinspection PyPep8Naming
class tEV(BivExtremeValueCopula):
    """
    Student-t Extreme Value Copula.

    Parameters
    ----------
    nu : float
        Degrees of freedom, nu > 0
    rho : float
        Correlation parameter, -1 < rho < 1
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
        if isinstance(self.rho, sympy.Symbol):
            return False
        return np.isclose(float(self.rho), 0.0)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _pickands(self):
        result = sympy.Piecewise(
            (Float(1.0), sympy.Or(self.t == 0, self.t == 1)),
            (self._compute_pickands(), True),
        )
        return result

    def _compute_pickands(self):
        def z(t):
            return (
                (1 + self.nu) ** sympy.Rational(1, 2)
                * ((t / (1 - t)) ** (1 / self.nu) - self.rho)
                * (1 - self.rho**2) ** sympy.Rational(-1, 2)
            )

        student_t = stats.StudentT("x", self.nu + 1)
        term1 = (1 - self.t) * stats.cdf(student_t)(z(1 - self.t))
        term2 = self.t * stats.cdf(student_t)(z(self.t))
        return re(term1 + term2)

    @property
    def pickands(self):
        pickands_expr = self._pickands

        class SafePickandsWrapper(SymPyFuncWrapper):
            def __call__(self_, t=None):
                if t is not None:
                    try:
                        t_float = float(t)
                        if (
                            t_float == 0
                            or t_float == 1
                            or t_float < 1e-10
                            or t_float > 1 - 1e-10
                        ):
                            return Float(1.0)
                    except (TypeError, ValueError):
                        pass

                if t is not None:
                    try:
                        result = self_.func.subs(self.t, t)
                        if hasattr(result, "is_complex") and result.is_complex:
                            return Float(sympy.re(result).evalf())
                        return result.evalf() if hasattr(result, "evalf") else result
                    except Exception:
                        return Float(1.0)

                return self_.func

            def __float__(self_):
                try:
                    result = self_.evalf()
                    if hasattr(result, "is_complex") and result.is_complex:
                        return float(sympy.re(result).evalf())
                    return float(result)
                except Exception:
                    return 1.0

        return SafePickandsWrapper(pickands_expr)

    def cdf(self, u=None, v=None, **kwargs):
        """
        Evaluate the tEV CDF numerically via *cdf_vectorized*.
        """
        if u is None:
            u = kwargs.pop("u", None)
        if v is None:
            v = kwargs.pop("v", None)
        if u is None or v is None:
            raise TypeError("cdf() requires keyword arguments u and v")

        scalar_in = np.ndim(u) == 0 and np.ndim(v) == 0
        result = self.cdf_vectorized(
            np.atleast_1d(np.asarray(u, dtype=float)),
            np.atleast_1d(np.asarray(v, dtype=float)),
        )
        return float(result[0]) if scalar_in else result

    @property
    def pdf(self):
        """Numerical PDF via finite-difference on *cdf_vectorized*."""
        outer = self

        class _TEVPDF:
            def __call__(self, u=None, v=None, **kw):
                if u is None:
                    u = kw.get("u")
                if v is None:
                    v = kw.get("v")
                if u is None or v is None:
                    raise TypeError("pdf() requires u and v")
                return outer._pdf_numerical(float(u), float(v))

        return _TEVPDF()

    def _pdf_numerical(self, u: float, v: float, h: float = 1e-5) -> float:
        """Mixed partial derivative ∂²C/∂u∂v via central differences."""
        if u <= 0 or v <= 0 or u >= 1 or v >= 1:
            return 0.0

        h = min(h, u / 2, v / 2, (1 - u) / 2, (1 - v) / 2)
        ua = np.array([u + h, u + h, u - h, u - h], dtype=float)
        va = np.array([v + h, v - h, v + h, v - h], dtype=float)
        c = self.cdf_vectorized(ua, va)
        return float((c[0] - c[1] - c[2] + c[3]) / (4.0 * h * h))

    def cdf_vectorized(self, u, v):
        """
        Optimized vectorized implementation of the CDF.
        """
        u_array = np.asarray(u, dtype=float)
        v_array = np.asarray(v, dtype=float)

        shape = np.broadcast(u_array, v_array).shape
        u_array = np.broadcast_to(u_array, shape)
        v_array = np.broadcast_to(v_array, shape)

        if np.any((u_array < 0) | (u_array > 1)) or np.any(
            (v_array < 0) | (v_array > 1)
        ):
            raise ValueError("Marginals must be in [0, 1]")

        result = np.zeros(shape, dtype=float)
        result = np.where(u_array == 1.0, v_array, result)
        result = np.where(v_array == 1.0, u_array, result)

        interior = (u_array > 0) & (u_array < 1) & (v_array > 0) & (v_array < 1)
        if not np.any(interior):
            return result

        try:
            nu_val = float(self.nu)
            rho_val = float(self.rho)
        except (TypeError, ValueError):
            result[interior] = np.minimum(u_array[interior], v_array[interior])
            return result

        u_interior = u_array[interior]
        v_interior = v_array[interior]

        uv_product = u_interior * v_interior
        log_uv = np.log(uv_product)
        log_v = np.log(v_interior)
        t_vals = log_v / log_uv

        a_vals = np.ones_like(t_vals)

        if nu_val > 0 and -1 < rho_val < 1:
            try:

                def z_func(t_array):
                    valid_mask = (t_array > 0) & (t_array < 1)
                    result = np.ones_like(t_array)

                    if np.any(valid_mask):
                        valid_t = t_array[valid_mask]
                        ratio = valid_t / (1.0 - valid_t)
                        ratio = np.clip(ratio, 1e-12, 1e12)

                        result[valid_mask] = (
                            np.sqrt(1.0 + nu_val)
                            * (np.power(ratio, 1.0 / nu_val) - rho_val)
                            / np.sqrt(1.0 - rho_val**2)
                        )
                    return result

                valid_t_mask = (t_vals > 0) & (t_vals < 1) & np.isfinite(t_vals)
                if np.any(valid_t_mask):
                    valid_t = t_vals[valid_t_mask]

                    z_t = z_func(valid_t)
                    z_1_minus_t = z_func(1.0 - valid_t)

                    cdf_t = t_dist.cdf(z_t, df=nu_val + 1.0)
                    cdf_1_minus_t = t_dist.cdf(z_1_minus_t, df=nu_val + 1.0)

                    a_valid = (1.0 - valid_t) * cdf_1_minus_t + valid_t * cdf_t
                    a_vals[valid_t_mask] = a_valid
                    a_vals[~valid_t_mask] = 1.0

            except Exception:
                a_vals = np.ones_like(t_vals)

        lower_bound = np.maximum(t_vals, 1.0 - t_vals)
        a_vals = np.maximum(a_vals, lower_bound)
        a_vals = np.minimum(a_vals, 1.0)

        cdf_vals = np.power(uv_product, a_vals)
        result[interior] = cdf_vals
        return result
