# file: copul/families/diagonal_band_b_inverse_reflected.py
import sympy as sp
import numpy as np

from copul.family.core.biv_copula import BivCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.frechet.lower_frechet import LowerFrechet
from copul.family.frechet.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class XiRhoBoundaryCopula(BivCopula):
    r"""
    Optimal–:math:`\rho` diagonal–band copula with dependence parameter
    :math:`b\in\mathbb{R}\setminus\{0\}`.

    For :math:`b>0`, the copula is the upper/right boundary copula.  For
    :math:`b<0`, the lower/left boundary copula is obtained by the reflection

    .. math::

       C_b(u,v) = v - C_{|b|}(1-u,v).

    **Formulas**

    1. Maximal Spearman’s :math:`\rho` for :math:`b>0`:

       .. math::

          M(b) =
          \begin{cases}
            b - \dfrac{3b^2}{10}, & 0<b\le 1,\\[1ex]
            1 - \dfrac{1}{2b^2} + \dfrac{1}{5b^3}, & b\ge 1.
          \end{cases}

    2. Shift :math:`s_v(b)` for :math:`b>0`:

       .. math::

          s_v =
          \begin{cases}
            \sqrt{2v/b}, & v \le \frac{1}{2b}\wedge\frac{b}{2},\\[1ex]
            v + \frac{1}{2b}, & \frac{1}{2b} < v \le 1-\frac{1}{2b},\\[1ex]
            \frac{v}{b}+\frac12, & \frac{b}{2} < v \le 1-\frac{b}{2},\\[1ex]
            1 + \frac1b - \sqrt{2(1-v)/b},
              & v > 1 - \left(\frac{1}{2b}\wedge\frac{b}{2}\right).
          \end{cases}

    3. Copula CDF for :math:`b>0`:

       .. math::

          a_v = s_v - 1/b,
          \qquad
          C_b(u,v) =
          \begin{cases}
            u - \dfrac{b}{2}(u-a_v)^2 + \dfrac{b}{2}(a_v\wedge 0)^2,
              & a_v < u \le s_v,\\[1ex]
            \min\{u,v\}, & \text{else}.
          \end{cases}
    """

    # symbolic parameter & admissible interval
    b = sp.symbols("b", real=True)
    params = [b]
    intervals = {"b": sp.Interval(-sp.oo, 0).union(sp.Interval(0, sp.oo))}
    special_cases = {0: BivIndependenceCopula}

    # convenience symbols
    u, v = sp.symbols("u v", nonnegative=True)

    def __new__(cls, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["b"] = args[0]
        if "b" in kwargs and kwargs["b"] in cls.special_cases:
            special_case_cls = cls.special_cases[kwargs["b"]]
            del kwargs["b"]  # Remove b before creating special case
            return special_case_cls()
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["b"] = args[0]
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["b"] = args[0]
        if "b" in kwargs and kwargs["b"] in self.special_cases:
            special_case_cls = self.special_cases[kwargs["b"]]
            del kwargs["b"]  # Remove b before creating special case
            return special_case_cls()
        return super().__call__(**kwargs)

    # -------- Maximal Spearman’s rho M(b) -------- #
    @staticmethod
    def _M_expr(b):
        """Piecewise maximal Spearman’s ρ in terms of the original parameter b."""
        b_abs = sp.Abs(b)
        M_when_abs_b_le_1 = b_abs - sp.Rational(3, 10) * b_abs**2
        M_when_abs_b_ge_1 = 1 - 1 / (2 * b_abs**2) + 1 / (5 * b_abs**3)
        return sp.Piecewise(
            (M_when_abs_b_le_1, b_abs <= 1),
            (M_when_abs_b_ge_1, True),
        )

    # -------- Shift s_v(b) -------- #
    @staticmethod
    def _s_expr(v, b):
        """
        Compute s_v for the original parameter b, using b_inv = 1/|b|.
        """
        b_inv = 1 / sp.Abs(b)

        # Region |b| >= 1, so min(1/(2|b|), |b|/2) = 1/(2|b|).
        v1_s_s = b_inv / 2
        s1_s_s = sp.sqrt(2 * v * b_inv)
        s2_s_s = v + b_inv / 2
        s3_s_s = 1 + b_inv - sp.sqrt(2 * b_inv * (1 - v))
        s_small = sp.Piecewise(
            (s1_s_s, v <= v1_s_s),
            (s2_s_s, v <= 1 - v1_s_s),
            (s3_s_s, True),
        )

        # Region |b| < 1, so min(1/(2|b|), |b|/2) = |b|/2.
        v1_s_L = 1 / (2 * b_inv)
        s1_s_L = sp.sqrt(2 * v * b_inv)
        s2_s_L = v * b_inv + sp.Rational(1, 2)
        s3_s_L = 1 + b_inv - sp.sqrt(2 * b_inv * (1 - v))
        s_large = sp.Piecewise(
            (s1_s_L, v <= v1_s_L),
            (s2_s_L, v <= 1 - v1_s_L),
            (s3_s_L, True),
        )

        return sp.Piecewise(
            (s_small, sp.Abs(b) >= 1),
            (s_large, True),
        )

    # -------- Base‐CDF for b > 0 -------- #
    @staticmethod
    def _base_cdf_expr(u, v, b):
        r"""
        The theoretical CDF for the positive parameter case.
        """
        s = XiRhoBoundaryCopula._s_expr(v, b)
        a = s - 1 / b
        quadratic_branch = u - b / 2 * (u - a) ** 2 + b / 2 * sp.Min(a, 0) ** 2

        return sp.Piecewise(
            (quadratic_branch, (a < u) & (u <= s)),
            (sp.Min(u, v), True),
        )

    # -------- CDF / PDF definitions -------- #
    @property
    def _cdf_expr(self):
        b, u, v = self.b, self.u, self.v

        # The “upright” expression for b > 0:
        C_pos = self._base_cdf_expr(u, v, b)

        # For b < 0, we reflect:  C_neg(u,v) = v - C_pos(1-u, v) with b → |b|
        C_reflected = v - self._base_cdf_expr(1 - u, v, sp.Abs(b))

        # Piecewise: choose C_pos if b > 0, else reflection
        C_full = sp.Piecewise(
            (C_pos, b > 0),
            (C_reflected, True),
        )
        return C_full

    def _pdf_expr(self):
        r"""
        Explicit Joint density :math:`c(u,v)` derived from the property
        :math:`c(u,v) = \lvert b \rvert \cdot s_v'(v)` inside the diagonal band,
        and 0 outside.

        For :math:`\lvert b \rvert \ge 1`:
            Density is :math:`\sqrt{\lvert b \rvert / 2v}` near 0, :math:`\lvert b \rvert` in the middle,
            and :math:`\sqrt{\lvert b \rvert / 2(1-v)}` near 1.

        For :math:`\lvert b \rvert < 1`:
            Density is :math:`\sqrt{\lvert b \rvert / 2v}` near 0, :math:`1` in the middle,
            and :math:`\sqrt{\lvert b \rvert / 2(1-v)}` near 1.
        """
        b, u, v = self.b, self.u, self.v
        b_abs = sp.Abs(b)

        # 1. Determine the transition threshold for v
        # If |b| >= 1, the linear section starts at v = 1/(2|b|)
        # If |b| < 1, the linear section starts at v = |b|/2
        v_thresh = sp.Piecewise(
            (1 / (2 * b_abs), b_abs >= 1),
            (b_abs / 2, True),  # covers |b| < 1
        )

        # 2. Determine density value in the middle linear section
        # If |b| >= 1, density = |b|
        # If |b| < 1, density = 1
        mid_density_val = sp.Piecewise((b_abs, b_abs >= 1), (1, True))

        # 3. Construct the explicit density value function (ignoring band boundaries for a moment)
        # s'(v)*|b| logic:
        # Near 0: sqrt(|b| / 2v)
        # Near 1: sqrt(|b| / 2(1-v))
        density_on_band = sp.Piecewise(
            (sp.sqrt(b_abs / (2 * v)), v <= v_thresh),
            (sp.sqrt(b_abs / (2 * (1 - v))), v >= 1 - v_thresh),
            (mid_density_val, True),
        )

        # 4. Define the diagonal band boundaries
        # We fetch s_v for the parameter |b|
        s = XiRhoBoundaryCopula._s_expr(v, b)

        # 5. Combine density with spatial indicator
        # For b > 0: band is (s - 1/b < u < s)
        # For b < 0: symmetry implies c(u,v) = c(|b|)(1-u, v).
        # We can implement this by swapping u -> 1-u in the indicator check if b < 0.

        # Note: s is computed based on b. If b < 0, _s_expr uses |b| internally.
        # So 's' represents the shift for C_|b|.

        # Condition for being inside the band (assuming positive parameter logic)
        # We use u_eff (effective u) to handle reflection
        u_eff = sp.Piecewise((u, b > 0), (1 - u, True))

        # Check: 0 < |b|(s - u_eff) < 1  <=>  s - 1/|b| < u_eff < s
        in_band = (u_eff > s - 1 / b_abs) & (u_eff < s)

        pdf_full = sp.Piecewise((density_on_band, in_band), (0, True))

        return SymPyFuncWrapper(pdf_full)

    @classmethod
    def from_xi(cls, x):
        """Instantiate from xi. Optimized to use float arithmetic if x is float."""
        if x == 0:
            return cls(b=0.0)
        elif x == 1:
            return UpperFrechet()
        elif x == -1:
            return LowerFrechet()

        # Fast path for floats to avoid symbolic solver overhead
        if isinstance(x, (float, np.floating)):
            # Case 1: 0 < x <= 3/10
            if 0 < x <= 0.3:
                denom = 2.0 * np.cos(np.arccos(-3.0 * np.sqrt(6.0 * x) / 5.0) / 3.0)
                b_val = np.sqrt(6.0 * x) / denom
                return cls(b=float(b_val))
            # Case 2: 3/10 < x < 1
            elif 0.3 < x < 1:
                numer = 5.0 + np.sqrt(5.0 * (6.0 * x - 1.0))
                denom = 10.0 * (1.0 - x)
                b_val = numer / denom
                return cls(b=float(b_val))

        # Fallback to symbolic for general expressions
        x_sym = sp.sympify(x)
        b_ge_1 = sp.sqrt(6 * x_sym) / (
            2 * sp.cos(sp.acos(-3 * sp.sqrt(6 * x_sym) / 5) / 3)
        )
        b_lt_1 = (5 + sp.sqrt(5 * (6 * x_sym - 1))) / (10 * (1 - x_sym))

        b_expr = sp.Piecewise(
            (b_ge_1, (x_sym > 0) & (x_sym <= sp.Rational(3, 10))),
            (b_lt_1, (x_sym > sp.Rational(3, 10)) & (x_sym < 1)),
        )

        return cls(b=float(b_expr))

    @staticmethod
    def _s_expr_numpy(v, b):
        """
        Calculates shift s_v.
        Optimized: Uses boolean masks to compute only required branches, avoiding unnecessary ops.
        """
        v = np.asarray(v)
        b = float(b)
        b_abs = abs(b)
        b_inv = 1.0 / b_abs

        s = np.empty_like(v, dtype=float)

        if b_abs >= 1.0:
            thresh_lower = b_inv / 2.0
            thresh_upper = 1.0 - thresh_lower

            mask1 = v <= thresh_lower
            mask3 = v > thresh_upper
            mask2 = ~(mask1 | mask3)

            if np.any(mask1):
                s[mask1] = np.sqrt(2.0 * v[mask1] * b_inv)
            if np.any(mask2):
                s[mask2] = v[mask2] + b_inv / 2.0
            if np.any(mask3):
                s[mask3] = 1.0 + b_inv - np.sqrt(2.0 * b_inv * (1.0 - v[mask3]))

        else:
            # |b| < 1 implies |b_inv| > 1
            thresh_lower = 1.0 / (2.0 * b_inv)
            thresh_upper = 1.0 - thresh_lower

            mask1 = v <= thresh_lower
            mask3 = v > thresh_upper
            mask2 = ~(mask1 | mask3)

            if np.any(mask1):
                s[mask1] = np.sqrt(2.0 * v[mask1] * b_inv)
            if np.any(mask2):
                s[mask2] = v[mask2] * b_inv + 0.5
            if np.any(mask3):
                s[mask3] = 1.0 + b_inv - np.sqrt(2.0 * b_inv * (1.0 - v[mask3]))

        return s

    @staticmethod
    def _base_cdf_numpy(u, v, b):
        """
        Optimized base CDF calculation for b > 0 using the theoretical formula.
        Explicitly broadcasts inputs to handle scalar/array mixes.
        """
        # Ensure inputs are arrays
        u = np.asarray(u)
        v = np.asarray(v)
        b = float(b)

        # Explicit broadcasting to ensure u and v have compatible shapes
        # for boolean indexing operations later.
        if u.shape != v.shape:
            u, v = np.broadcast_arrays(u, v)

        s = XiRhoBoundaryCopula._s_expr_numpy(v, b)
        a = s - 1.0 / b
        result = np.minimum(u, v).astype(float, copy=False)
        mask = (a < u) & (u <= s)

        if np.any(mask):
            result = np.array(result, dtype=float, copy=True)
            result[mask] = (
                u[mask]
                - 0.5 * b * (u[mask] - a[mask]) ** 2
                + 0.5 * b * np.minimum(a[mask], 0.0) ** 2
            )

        return result

    def pdf_vectorized(self, u, v):
        """
        Fully vectorized PDF implementation.
        Computes density c(u,v) = |b| * s'_v(v) inside the band, 0 outside.
        Replaces symbolic logic for high performance.
        """
        u = np.asarray(u)
        v = np.asarray(v)
        b = self.b
        b_abs = abs(b)

        # Symmetry handling: if b < 0, c(u,v) = c(|b|)(1-u, v)
        u_eff = u if b > 0 else (1.0 - u)

        # 1. Calculate Shift s_v
        s = self._s_expr_numpy(v, b_abs)

        # 2. Calculate Derivative s'_v (ds/dv)
        # We need to replicate the thresholds from _s_expr_numpy to get derivatives
        ds_dv = np.empty_like(v, dtype=float)
        b_inv = 1.0 / b_abs

        if b_abs >= 1.0:
            thresh_lower = b_inv / 2.0
            thresh_upper = 1.0 - thresh_lower

            mask1 = v <= thresh_lower
            mask3 = v > thresh_upper
            mask2 = ~(mask1 | mask3)

            if np.any(mask1):
                # Avoid div by zero at v=0 (density infinite there anyway)
                with np.errstate(divide="ignore"):
                    ds_dv[mask1] = np.sqrt(b_inv / (2.0 * v[mask1]))

            if np.any(mask2):
                ds_dv[mask2] = 1.0

            if np.any(mask3):
                with np.errstate(divide="ignore"):
                    ds_dv[mask3] = np.sqrt(b_inv / (2.0 * (1.0 - v[mask3])))
        else:
            thresh_lower = 1.0 / (2.0 * b_inv)
            thresh_upper = 1.0 - thresh_lower

            mask1 = v <= thresh_lower
            mask3 = v > thresh_upper
            mask2 = ~(mask1 | mask3)

            if np.any(mask1):
                with np.errstate(divide="ignore"):
                    ds_dv[mask1] = np.sqrt(b_inv / (2.0 * v[mask1]))

            if np.any(mask2):
                ds_dv[mask2] = b_inv

            if np.any(mask3):
                with np.errstate(divide="ignore"):
                    ds_dv[mask3] = np.sqrt(b_inv / (2.0 * (1.0 - v[mask3])))

        # 3. Calculate Density
        # c(u,v) = |b| * s'(v)   if   a < u_eff < s   else 0
        # where a = s - 1/|b|
        density_val = b_abs * ds_dv

        lower_bound = s - (1.0 / b_abs)
        upper_bound = s

        in_band = (u_eff > lower_bound) & (u_eff < upper_bound)

        result = np.where(in_band, density_val, 0.0)

        return result

    def cdf_vectorized(self, u, v):
        """
        Vectorized implementation of the cumulative distribution function.
        This method allows for efficient computation of the CDF for arrays of points,
        which is detected by the `Checkerboarder` for fast approximation.
        """
        b = self.b
        if b > 0:
            return self._base_cdf_numpy(u, v, b)
        else:  # b < 0
            u, v = np.asarray(u), np.asarray(v)
            # Apply the reflection identity: C_neg(u,v) = v - C_pos(1-u, v) with b -> |b|
            return v - self._base_cdf_numpy(1 - u, v, np.abs(b))

    # ===================================================================
    # END: Vectorized CDF implementation
    # ===================================================================

    # -------- Metadata -------- #
    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    def chatterjees_xi(self):
        r"""
        Closed-form :math:`\xi(C_b)` for the original parameter :math:`b`.
        """
        b_abs = sp.Abs(self.b)
        xi_abs_b_le_1 = b_abs**2 * (5 - 2 * b_abs) / 10
        xi_abs_b_ge_1 = 1 - 1 / b_abs + sp.Rational(3, 10) / b_abs**2
        return sp.Piecewise(
            (xi_abs_b_le_1, b_abs <= 1),
            (xi_abs_b_ge_1, True),
        )

    def spearmans_rho(self):
        r"""
        Closed-form Spearman’s :math:`\rho(C_b)` for the original parameter :math:`b`.
        """
        b = self.b
        b_abs = sp.Abs(b)
        rho_abs_b_le_1 = b_abs - sp.Rational(3, 10) * b_abs**2
        rho_abs_b_ge_1 = 1 - 1 / (2 * b_abs**2) + 1 / (5 * b_abs**3)
        return sp.sign(b) * sp.Piecewise(
            (rho_abs_b_le_1, b_abs <= 1),
            (rho_abs_b_ge_1, True),
        )

    def kendalls_tau(self):
        r"""
        Closed-form Kendall’s :math:`\tau(C_b)` for the original parameter :math:`b`.
        """
        b = self.b
        b_abs = sp.Abs(b)
        tau_abs_b_le_1 = 2 * b_abs / 3 - b_abs**2 / 6
        tau_abs_b_ge_1 = 1 - 2 / (3 * b_abs) + 1 / (6 * b_abs**2)
        return sp.sign(b) * sp.Piecewise(
            (tau_abs_b_le_1, b_abs <= 1),
            (tau_abs_b_ge_1, True),
        )

    def blests_nu(self, *args, **kwargs):
        return self.spearmans_rho()

    #
    # def cond_distr_1(self, u=None, v=None):
    #     """
    #     First conditional distribution function: ∂C(u,v)/∂u
    #
    #     Parameters
    #     ----------
    #     u, v : float or None, optional
    #         Values to evaluate at. If None, returns the symbolic expression.
    #
    #     Returns
    #     -------
    #     CD1Wrapper
    #         The conditional distribution
    #     """
    #     b = self.b
    #     b_abs = sp.Abs(b)
    #
    #     # Symmetry: ∂₁C_b(u,v) = ∂₁C_|b|(1-u, v) for b < 0
    #     u_sym = self.u
    #     u_eff = sp.Piecewise((u_sym, b > 0), (1 - u_sym, True))
    #
    #     # Get shift s_v using |b|
    #     s = self._s_expr(self.v, b_abs)
    #
    #     # The generator formula is h(t) = clamp(|b|*(s - t), 0, 1)
    #     # where t is the effective u
    #     val = b_abs * (s - u_eff)
    #
    #     # Clamp between 0 and 1
    #     res = sp.Max(sp.Min(val, 1), 0)
    #
    #     return CD1Wrapper(res)(u, v)
    #
    # def cond_distr_2(self, u=None, v=None):
    #     """
    #     Second conditional distribution function: ∂C(u,v)/∂v
    #
    #     Parameters
    #     ----------
    #     u, v : float or None, optional
    #         Values to evaluate at. If None, returns the symbolic expression.
    #
    #     Returns
    #     -------
    #     CD2Wrapper
    #         The conditional distribution
    #     """
    #     b = self.b
    #     b_abs = sp.Abs(b)
    #
    #     # 1. Construct the CD2 for the positive parameter case |b|
    #     # The density inside the band is constant in u: c(u,v) = |b| * s'_v(v)
    #     s = self._s_expr(self.v, b_abs)
    #     ds_dv = s.diff(self.v)
    #     density = b_abs * ds_dv
    #
    #     # Support boundaries for |b| case
    #     # Lower: max(0, s - 1/|b|)
    #     # Upper: min(1, s)
    #     b_inv = 1 / b_abs
    #     L = sp.Max(0, s - b_inv)
    #     R = sp.Min(1, s)
    #
    #     # 2. Determine effective u based on symmetry
    #     # For b < 0: C_b(u,v) = v - C_|b|(1-u, v)
    #     # Therefore: ∂₂C_b(u,v) = 1 - ∂₂C_|b|(1-u, v)
    #     u_sym = self.u
    #     u_target = sp.Piecewise((u_sym, b > 0), (1 - u_sym, True))
    #
    #     # 3. Calculate CD2_|b|(u_target, v)
    #     # Since density is constant in u, CD2 is a linear ramp from 0 to 1
    #     val_pos = density * (u_target - L)
    #     cd2_pos_expr = sp.Piecewise(
    #         (0, u_target <= L),
    #         (1, u_target >= R),
    #         (val_pos, True),
    #     )
    #
    #     # 4. Apply reflection if b < 0
    #     result = sp.Piecewise((cd2_pos_expr, b > 0), (1 - cd2_pos_expr, True))
    #
    #     return CD2Wrapper(result)(u, v)


if __name__ == "__main__":
    # Example usage
    # XiRhoBoundaryCopula(b=-0.5).plot_cond_distr_1(plot_type="contour")
    # XiRhoBoundaryCopula(b=-0.5).plot_cond_distr_2(plot_type="contour")
    # XiRhoBoundaryCopula(b=-1).plot_cond_distr_1(plot_type="contour")
    # XiRhoBoundaryCopula(b=-1).plot_cond_distr_2(plot_type="contour")
    XiRhoBoundaryCopula(b=-0.5).plot_pdf(plot_type="contour", levels=100, zlim=(0, 5))
    XiRhoBoundaryCopula(b=-1).plot_pdf(plot_type="contour", levels=100, zlim=(0, 6.5))
    XiRhoBoundaryCopula(b=-2).plot_pdf(plot_type="contour", levels=100, zlim=(0, 8))
