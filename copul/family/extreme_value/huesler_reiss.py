import numpy as np
import sympy
from sympy import stats
from scipy.stats import norm
from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula


class HueslerReiss(BivExtremeValueCopula):
    r"""
    Hüsler–Reiss extreme value copula with parameter :math:`\delta \ge 0`.
    When :math:`\delta=0`, it reduces to the independence copula.
    """

    delta = sympy.Symbol("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, sympy.oo, left_open=False, right_open=True)}

    def __new__(cls, *args, **kwargs):
        if (len(args) == 1 and args[0] == 0) or kwargs.get("delta", None) == 0:
            return BivIndependenceCopula()
        return super().__new__(cls)

    def __call__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["delta"] = args[0]
        if kwargs.get("delta", None) == 0:
            kwargs.pop("delta")
            return BivIndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _pickands(self):
        std_norm = stats.cdf(stats.Normal("Z", 0, 1))
        return (1 - self.t) * std_norm(
            1 / self.delta + (self.delta / 2) * sympy.log((1 - self.t) / self.t)
        ) + self.t * std_norm(
            1 / self.delta + (self.delta / 2) * sympy.log(self.t / (1 - self.t))
        )

    def _A(self, t):
        std_norm = stats.cdf(stats.Normal("Z", 0, 1))
        return (1 - t) * std_norm(self._z(1 - t)) + t * std_norm(self._z(t))

    def _z(self, t):
        return 1 / self.delta + (self.delta / 2) * sympy.log(t / (1 - t))

    # ------------------------------------------------------------------
    # Fast CDF: route concrete evaluations to the vectorised implementation
    # instead of symbolic SymPy evaluation.
    # ------------------------------------------------------------------

    def cdf(self, u=None, v=None, **kwargs):
        """Evaluate the CDF numerically via *cdf_vectorized*."""
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

    # ------------------------------------------------------------------
    # Fast PDF: numerical mixed-partial via cdf_vectorized.
    # ------------------------------------------------------------------

    @property
    def pdf(self):
        """Numerical PDF via finite-difference on *cdf_vectorized*."""
        outer = self

        class _HueslerReissPDF:
            def __call__(self, u=None, v=None, **kw):
                if u is None:
                    u = kw.get("u")
                if v is None:
                    v = kw.get("v")
                if u is None or v is None:
                    raise TypeError("pdf() requires u and v")
                return outer._pdf_numerical(float(u), float(v))

        return _HueslerReissPDF()

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
        Vectorized implementation of the Hüsler–Reiss copula CDF:

            C(u,v) = (u * v)^A(t), where
            t = ln(v) / ln(u*v),
            A(t) = (1 - t)*Φ(z(1 - t)) + t*Φ(z(t)),
            z(x) = 1/delta + (delta/2)*ln(x/(1 - x)).
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        shape = np.broadcast(u, v).shape
        u = np.broadcast_to(u, shape)
        v = np.broadcast_to(v, shape)

        result = np.zeros(shape, dtype=float)

        if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
            raise ValueError("Both u and v must be in [0,1].")

        mask_u1 = u == 1.0
        if np.any(mask_u1):
            result[mask_u1] = v[mask_u1]

        mask_v1 = v == 1.0
        if np.any(mask_v1):
            result[mask_v1] = u[mask_v1]

        mask_u0 = u == 0.0
        mask_v0 = v == 0.0
        if np.any(mask_u0):
            result[mask_u0] = 0.0
        if np.any(mask_v0):
            result[mask_v0] = 0.0

        interior_mask = (u > 0) & (u < 1) & (v > 0) & (v < 1)
        if not np.any(interior_mask):
            return result

        delta_val = float(self.delta)
        if delta_val == 0.0:
            result[interior_mask] = u[interior_mask] * v[interior_mask]
            return result

        u_in = u[interior_mask]
        v_in = v[interior_mask]

        uv_in = u_in * v_in
        t_vals = np.log(v_in) / np.log(uv_in)

        def z_fn(x):
            return (1.0 / delta_val) + 0.5 * delta_val * np.log(x / (1.0 - x))

        z_1_minus_t = z_fn(1 - t_vals)
        z_t = z_fn(t_vals)

        Phi_1_minus_t = norm.cdf(z_1_minus_t)
        Phi_t = norm.cdf(z_t)

        A_vals = (1 - t_vals) * Phi_1_minus_t + t_vals * Phi_t
        cdf_vals = np.power(uv_in, A_vals)

        result[interior_mask] = cdf_vals
        return result
