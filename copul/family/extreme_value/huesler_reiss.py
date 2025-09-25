import numpy as np
import sympy
from sympy import stats
from scipy.stats import norm
from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper


class HueslerReiss(BivExtremeValueCopula):
    r"""
    Hüsler–Reiss extreme value copula with parameter :math:`\delta \ge 0`.
    When :math:`\delta=0`, it reduces to the independence copula.
    """

    # --------------------------------------------------------------
    # Symbolic/Sympy setup
    # --------------------------------------------------------------
    delta = sympy.Symbol("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, sympy.oo, left_open=False, right_open=True)}

    def __new__(cls, *args, **kwargs):
        # Special handling for delta=0
        if (len(args) == 1 and args[0] == 0) or kwargs.get("delta", None) == 0:
            return BivIndependenceCopula()
        return super().__new__(cls)

    def __call__(self, *args, **kwargs):
        """
        So that HueslerReiss(delta=0) -> IndependenceCopula at runtime.
        """
        if args and len(args) == 1:
            kwargs["delta"] = args[0]
        if kwargs.get("delta", None) == 0:
            # Return independence copula instead
            kwargs.pop("delta")
            return BivIndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    # --------------------------------------------------------------
    # Pickands function
    #
    # A(t) = (1 - t)*Φ(1/δ + (δ/2)*ln((1 - t)/t))
    #        + t*Φ(1/δ + (δ/2)*ln(t/(1 - t)))
    #
    # --------------------------------------------------------------
    @property
    def _pickands(self):
        """
        Symbolic expression for the Pickands dependence function A(t).
        """
        std_norm = stats.cdf(stats.Normal("Z", 0, 1))
        return (1 - self.t) * std_norm(
            1 / self.delta + (self.delta / 2) * sympy.log((1 - self.t) / self.t)
        ) + self.t * std_norm(
            1 / self.delta + (self.delta / 2) * sympy.log(self.t / (1 - self.t))
        )

    # --------------------------------------------------------------
    # CDF (symbolic)
    #
    #   ℓ(x, y) = x*Φ(1/δ + (δ/2)*ln(y/x)) + y*Φ(1/δ + (δ/2)*ln(x/y))
    #   C(u,v) = exp(-ℓ(x,y)), where x= -1/log(u), y= -1/log(v).
    #
    # --------------------------------------------------------------
    @property
    def cdf(self):
        u, v = self.u, self.v
        a = self._A(sympy.log(v) / (sympy.log(u) + sympy.log(v)))
        result = sympy.exp(sympy.log(u * v) * a)
        cdf = CDFWrapper(result)
        return cdf

    def _A(self, t):
        std_norm = stats.cdf(stats.Normal("Z", 0, 1))
        return (1 - t) * std_norm(self._z(1 - t)) + t * std_norm(self._z(t))

    def _z(self, t):
        return 1 / self.delta + (self.delta / 2) * sympy.log(t / (1 - t))

    # ToDo - why does the following lead to weird results?
    # it looks correct and seems to follow
    # https://search.r-project.org/CRAN/refmans/copBasic/html/HRcop.html
    # @property
    # def cdf(self):
    #     """
    #     Symbolic expression for the Hüsler–Reiss bivariate copula CDF.
    #     """
    #     # Standard normal CDF
    #     std_norm = stats.cdf(stats.Normal("Z", 0, 1))
    #     u, v = self.u, self.v

    #     # Convert to standard Fréchet scale
    #     x = -1 / sympy.log(u)
    #     y = -1 / sympy.log(v)

    #     part1 = x * std_norm(1/self.delta + (self.delta/2)*sympy.log(y/x))
    #     part2 = y * std_norm(1/self.delta + (self.delta/2)*sympy.log(x/y))
    #     result = sympy.exp(-(part1 + part2))

    #     return CDFWrapper(result)

    # --------------------------------------------------------------
    # CDF (vectorized, numeric)
    #
    #   C(u,v) = exp(-[ x*Φ(1/δ + (δ/2)*ln(y/x))
    #                   + y*Φ(1/δ + (δ/2)*ln(x/y)) ])
    #   where x=-1/log(u), y=-1/log(v).
    #
    # Boundary conditions:
    #   C(0,v)=0, C(u,0)=0, C(1,v)=v, C(u,1)=u.
    # --------------------------------------------------------------
    def cdf_vectorized(self, u, v):
        """
        Vectorized implementation of the Hüsler–Reiss copula CDF, mirroring the new
        cdf() approach:

            C(u,v) = (u * v)^A( t ),   where
            t = ln(v) / ln(u*v),
            A(t) = (1 - t)*Φ(z(1 - t)) + t*Φ(z(t)),
            z(x) = 1/delta + (delta/2)*ln(x/(1 - x)).

        Boundary handling:
        - C(0,v)=0, C(u,0)=0, C(1,v)=v, C(u,1)=u,
        - delta=0 => independence => C(u,v)=u*v.

        Parameters
        ----------
        u : array_like
            First uniform margin, 0 <= u <= 1.
        v : array_like
            Second uniform margin, 0 <= v <= 1.

        Returns
        -------
        numpy.ndarray
            The values of C(u, v).
        """

        # Convert to arrays
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # Determine the common broadcast shape
        shape = np.broadcast(u, v).shape

        # Broadcast both u and v to the same shape
        u = np.broadcast_to(u, shape)
        v = np.broadcast_to(v, shape)

        # Prepare output
        result = np.zeros(shape, dtype=float)

        # Check domain
        if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
            raise ValueError("Both u and v must be in [0,1].")

        # Handle boundary conditions
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

        # Identify interior points: (u, v) in (0,1)^2
        interior_mask = (u > 0) & (u < 1) & (v > 0) & (v < 1)
        if not np.any(interior_mask):
            return result

        delta_val = float(self.delta)
        if delta_val == 0.0:
            # independence copula
            result[interior_mask] = u[interior_mask] * v[interior_mask]
            return result

        # For interior points, replicate cdf = (u*v)^A(t)
        # with t = ln(v)/ln(u*v).
        u_in = u[interior_mask]
        v_in = v[interior_mask]

        uv_in = u_in * v_in
        t_vals = np.log(v_in) / np.log(uv_in)

        # z(x) = 1/delta + (delta/2)*ln(x/(1-x))
        def z_fn(x):
            return (1.0 / delta_val) + 0.5 * delta_val * np.log(x / (1.0 - x))

        z_1_minus_t = z_fn(1 - t_vals)
        z_t = z_fn(t_vals)

        # Evaluate standard normal CDF
        Phi_1_minus_t = norm.cdf(z_1_minus_t)
        Phi_t = norm.cdf(z_t)

        # A(t) = (1 - t)*Phi(z(1-t)) + t*Phi(z(t))
        A_vals = (1 - t_vals) * Phi_1_minus_t + t_vals * Phi_t

        # final copula values => (u*v)^{A(t)}
        cdf_vals = np.power(uv_in, A_vals)

        result[interior_mask] = cdf_vals
        return result
