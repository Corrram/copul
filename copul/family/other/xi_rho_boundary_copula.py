# file: copul/families/diagonal_band_b_inverse_reflected.py
import sympy as sp
import numpy as np

from copul.family.core.biv_copula import BivCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.frechet.lower_frechet import LowerFrechet
from copul.family.frechet.upper_frechet import UpperFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class XiRhoBoundaryCopula(BivCopula):
    r"""
    Optimal–:math:`\rho` diagonal–band copula, parameterised by :math:`b_{\mathrm{new}}`
    so that the original scale parameter :math:`b_{\mathrm{old}} = 1/\lvert b_{\mathrm{new}}\rvert`.
    For :math:`b_{\mathrm{new}} < 0`, we use the reflection identity

    .. math::

       C_{b_{\rm new}}^{\downarrow}(u,v)
       \;=\; v \;-\; C_{\lvert b_{\rm new}\rvert}^{\uparrow}\!\bigl(1 - u,\,v\bigr).

    **Parameter** — :math:`b_{\mathrm{new}}`
        :math:`b_{\mathrm{new}}\in\mathbb{R}\setminus\{0\}`.
        For :math:`b_{\mathrm{new}} > 0`, :math:`b_{\mathrm{old}} = 1/b_{\mathrm{new}} > 0`;
        for :math:`b_{\mathrm{new}} < 0`, use :math:`\lvert b_{\mathrm{new}}\rvert` as above and apply the
        “down–reflection.”

    **Formulas**

    1. Maximal Spearman’s :math:`\rho`:

       Let :math:`b := b_{\mathrm{new}}`. Then :math:`b_{\mathrm{old}} = 1/\lvert b\rvert`.
       We can write :math:`M(b)` piecewise in terms of :math:`\lvert b\rvert`:

       .. math::

          M(b) \;=\;
          \begin{cases}
            b - \dfrac{3\,b^{2}}{10}, & \lvert b\rvert \ge 1,\\[1ex]
            1 - \dfrac{1}{2\,b^{2}} + \dfrac{1}{5\,b^{3}}, & \lvert b\rvert < 1.
          \end{cases}

    2. Shift :math:`s_v(b)`:

       Define :math:`b_{\mathrm{old}} = 1/\lvert b\rvert`. For :math:`\lvert b_{\mathrm{old}}\rvert \le 1`
       (i.e. :math:`\lvert b\rvert \ge 1`),

       .. math::

          s_v \;=\;
          \begin{cases}
            \sqrt{2\,v\,b_{\text{old}}}, & v \le \tfrac{b_{\text{old}}}{2},\\
            v + \tfrac{b_{\text{old}}}{2}, & v \in \bigl(\tfrac{b_{\text{old}}}{2},\,1 - \tfrac{b_{\text{old}}}{2}\bigr],\\
            1 + b_{\text{old}} - \sqrt{2\,b_{\text{old}}(1-v)}, & v > 1 - \tfrac{b_{\text{old}}}{2}.
          \end{cases}

       For :math:`\lvert b_{\mathrm{old}}\rvert > 1` (i.e. :math:`\lvert b\rvert < 1`),

       .. math::

          s_v \;=\;
          \begin{cases}
            \sqrt{2\,v\,b_{\text{old}}}, & v \le \tfrac{1}{2\,b_{\text{old}}},\\
            v\,b_{\text{old}} + \tfrac12, & v \in \bigl(\tfrac{1}{2\,b_{\text{old}}},\,1 - \tfrac{1}{2\,b_{\text{old}}}\bigr],\\
            1 + b_{\text{old}} - \sqrt{2\,b_{\text{old}}(1-v)}, & v > 1 - \tfrac{1}{2\,b_{\text{old}}}.
          \end{cases}

    3. Copula CDF:

       For :math:`b_{\mathrm{new}} > 0`, use the triangle–band formula with :math:`b_{\mathrm{old}} = 1/b_{\mathrm{new}}`:

       .. math::

          a_v = s_v - b_{\text{old}}, \qquad
          C(u,v) =
          \begin{cases}
            u, & u \le a_v,\\[0.6ex]
            a_v + \dfrac{2\,s_v\,(u - a_v) - u^2 + a_v^2}{2\,b_{\text{old}}}, & a_v < u \le s_v,\\[1ex]
            v, & u > s_v.
          \end{cases}

       For :math:`b_{\mathrm{new}} < 0`, set

       .. math::

          C_{b_{\rm new}}(u,v) \;=\; v \;-\; C_{\lvert b_{\rm new}\rvert}\!\bigl(1 - u,\,v\bigr).
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
        """Piecewise maximal Spearman’s ρ in terms of b_new."""
        # When |b| ≥ 1, then b_old = 1/|b| ≤ 1 → formula b_old‐small → inverts to:
        M_when_abs_b_ge_1 = b - sp.Rational(3, 10) * b**2
        # When |b| < 1, then b_old = 1/|b| > 1 → formula b_old‐large → inverts to:
        M_when_abs_b_lt_1 = 1 - 1 / (2 * b**2) + 1 / (5 * b**3)
        return sp.Piecewise(
            (M_when_abs_b_ge_1, sp.Abs(b) >= 1),
            (M_when_abs_b_lt_1, True),
        )

    # -------- Shift s_v(b) -------- #
    @staticmethod
    def _s_expr(v, b):
        """
        Compute s_v for given v and new parameter b_new, where b_old = 1/|b|.
        """
        b_old = 1 / sp.Abs(b)

        # Region “small‐b_old”: |b_old| ≤ 1  ⇔  |b| ≥ 1
        v1_s_s = b_old / 2
        s1_s_s = sp.sqrt(2 * v * b_old)
        s2_s_s = v + b_old / 2
        s3_s_s = 1 + b_old - sp.sqrt(2 * b_old * (1 - v))
        s_small = sp.Piecewise(
            (s1_s_s, v <= v1_s_s),
            (s2_s_s, v <= 1 - v1_s_s),
            (s3_s_s, True),
        )

        # Region “large‐b_old”: |b_old| > 1  ⇔  |b| < 1
        v1_s_L = 1 / (2 * b_old)
        s1_s_L = sp.sqrt(2 * v * b_old)
        s2_s_L = v * b_old + sp.Rational(1, 2)
        s3_s_L = 1 + b_old - sp.sqrt(2 * b_old * (1 - v))
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
        Explicit integration of the clamped density:
        :math:`h_v(t) = \operatorname{clamp}(b(s_v - t), 0, 1)`.

        This robust implementation correctly handles cases where the linear
        section starts before :math:`t=0` (i.e., when :math:`a_v < 0`).
        """
        s = XiRhoBoundaryCopula._s_expr(v, b)
        a = s - 1 / b

        # We integrate density from 0 to u.
        # The density has two potential active regions on [0, 1]:
        # 1. Flat region (val=1): t <= a
        # 2. Linear region (val=b(s-t)): a < t <= s

        # 1. Flat part integration: Intersection of [0, u] and [0, a]
        # If a < 0, this intersection is empty (Max(0, a) handles this).
        upper_flat = sp.Min(u, sp.Max(0, a))
        term_flat = upper_flat  # Integral of 1 * dt is just the length

        # 2. Linear part integration: Intersection of [0, u] and [a, s]
        L = sp.Max(0, a)
        R = sp.Min(u, s)

        # Integral of b(s-t) dt from L to R:
        # Area = (R - L) * average_height
        # average_height = b*s - (b/2)*(R + L)
        term_linear = sp.Piecewise(
            ((R - L) * (b * s - (b / 2) * (R + L)), R > L), (0, True)
        )

        return term_flat + term_linear

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
            (1 / (2 * b_abs), b_abs >= 1), (b_abs / 2, True)  # covers |b| < 1
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
        b_old = 1.0 / b_abs

        s = np.empty_like(v, dtype=float)

        if b_abs >= 1.0:
            # |b| >= 1 implies |b_old| <= 1
            thresh_lower = b_old / 2.0
            thresh_upper = 1.0 - thresh_lower

            mask1 = v <= thresh_lower
            mask3 = v > thresh_upper
            mask2 = ~(mask1 | mask3)

            if np.any(mask1):
                s[mask1] = np.sqrt(2.0 * v[mask1] * b_old)
            if np.any(mask2):
                s[mask2] = v[mask2] + b_old / 2.0
            if np.any(mask3):
                s[mask3] = 1.0 + b_old - np.sqrt(2.0 * b_old * (1.0 - v[mask3]))

        else:
            # |b| < 1 implies |b_old| > 1
            thresh_lower = 1.0 / (2.0 * b_old)
            thresh_upper = 1.0 - thresh_lower

            mask1 = v <= thresh_lower
            mask3 = v > thresh_upper
            mask2 = ~(mask1 | mask3)

            if np.any(mask1):
                s[mask1] = np.sqrt(2.0 * v[mask1] * b_old)
            if np.any(mask2):
                s[mask2] = v[mask2] * b_old + 0.5
            if np.any(mask3):
                s[mask3] = 1.0 + b_old - np.sqrt(2.0 * b_old * (1.0 - v[mask3]))

        return s

    @staticmethod
    def _base_cdf_numpy(u, v, b):
        """
        Optimized base CDF calculation for b > 0.
        Uses masking to calculate linear terms only where active.
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

        # At this point, u and v (and thus s, a, L, R) share the same shape

        s = XiRhoBoundaryCopula._s_expr_numpy(v, b)
        a = s - 1.0 / b

        # 1. Flat top part (density = 1) -> valid for t <= a
        # Intersect [0, u] with [0, max(0, a)]
        term_flat = np.maximum(0.0, np.minimum(u, np.maximum(0.0, a)))

        # 2. Linear slope part (density = b(s-t)) -> valid for a < t <= s
        L = np.maximum(0.0, a)
        R = np.minimum(u, s)

        # Only compute linear term where R > L
        mask_linear = R > L

        # Initialize result with correct broadcasted shape
        term_linear = np.zeros_like(u, dtype=float)

        if np.any(mask_linear):
            # Safe to index because term_linear, R, L, s, and mask_linear
            # all have the same broadcasted shape
            R_sub = R[mask_linear]
            L_sub = L[mask_linear]
            s_sub = s[mask_linear]

            # Integral: (R-L) * (b*s - 0.5*b*(R+L))
            val = (R_sub - L_sub) * (b * s_sub - 0.5 * b * (R_sub + L_sub))
            term_linear[mask_linear] = val

        return term_flat + term_linear

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
        b_old = 1.0 / b_abs

        if b_abs >= 1.0:
            thresh_lower = b_old / 2.0
            thresh_upper = 1.0 - thresh_lower

            mask1 = v <= thresh_lower
            mask3 = v > thresh_upper
            mask2 = ~(mask1 | mask3)

            if np.any(mask1):
                # Avoid div by zero at v=0 (density infinite there anyway)
                with np.errstate(divide="ignore"):
                    ds_dv[mask1] = np.sqrt(b_old / (2.0 * v[mask1]))

            if np.any(mask2):
                ds_dv[mask2] = 1.0

            if np.any(mask3):
                with np.errstate(divide="ignore"):
                    ds_dv[mask3] = np.sqrt(b_old / (2.0 * (1.0 - v[mask3])))
        else:
            thresh_lower = 1.0 / (2.0 * b_old)
            thresh_upper = 1.0 - thresh_lower

            mask1 = v <= thresh_lower
            mask3 = v > thresh_upper
            mask2 = ~(mask1 | mask3)

            if np.any(mask1):
                with np.errstate(divide="ignore"):
                    ds_dv[mask1] = np.sqrt(b_old / (2.0 * v[mask1]))

            if np.any(mask2):
                ds_dv[mask2] = b_old

            if np.any(mask3):
                with np.errstate(divide="ignore"):
                    ds_dv[mask3] = np.sqrt(b_old / (2.0 * (1.0 - v[mask3])))

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
        Closed-form :math:`\xi(b_{\mathrm{new}})`. Recall :math:`b_{\mathrm{old}} = 1/\lvert b_{\mathrm{new}}\rvert`,
        so the “:math:`\le 1` / :math:`\ge 1`” conditions swap in the new scale.

        - If :math:`\lvert b_{\mathrm{new}}\rvert \ge 1` (i.e. :math:`\lvert b_{\mathrm{old}}\rvert \le 1`):
          :math:`\xi = \dfrac{1}{10\lvert b\rvert^{2}}\,(5 - 2/\lvert b\rvert)`.

        - If :math:`\lvert b_{\mathrm{new}}\rvert < 1` (i.e. :math:`\lvert b_{\mathrm{old}}\rvert \ge 1`):
          :math:`\xi = 1 - \lvert b\rvert + \dfrac{3}{10}\lvert b\rvert^{2}`.
        """
        b = 1 / self.b
        xi_large = (sp.Rational(1, 10) / sp.Abs(b) ** 2) * (5 - 2 / sp.Abs(b))
        xi_small = 1 - sp.Abs(b) + sp.Rational(3, 10) * sp.Abs(b) ** 2
        return sp.Piecewise(
            (xi_large, sp.Abs(b) >= 1),  # |b_new| ≥ 1
            (xi_small, True),  # |b_new|  < 1
        )

    def spearmans_rho(self):
        r"""
        Closed-form Spearman’s :math:`\rho(b_{\mathrm{new}})` (from Prop. 3.4 with
        :math:`b_{\mathrm{old}} = 1/\lvert b_{\mathrm{new}}\rvert`).

        - If :math:`\lvert b_{\mathrm{new}}\rvert \ge 1` (i.e. :math:`\lvert b_{\mathrm{old}}\rvert \le 1`):

          .. math:: \rho = \operatorname{sgn}(b)\,\!\left(\frac{1}{\lvert b\rvert} - \frac{3}{10\,\lvert b\rvert^{2}}\right).

        - If :math:`\lvert b_{\mathrm{new}}\rvert < 1` (i.e. :math:`\lvert b_{\mathrm{old}}\rvert \ge 1`):

          .. math:: \rho = \operatorname{sgn}(b)\,\!\left(1 - \frac{\lvert b\rvert^{2}}{2}\right) + \frac{\lvert b\rvert^{3}}{5}.
        """
        b = 1 / self.b
        rho_large = sp.sign(b) * (1 / sp.Abs(b) - sp.Rational(3, 10) / sp.Abs(b) ** 2)
        rho_small = sp.sign(b) * (1 - sp.Abs(b) ** 2 / 2 + sp.Abs(b) ** 3 / 5)
        return sp.Piecewise(
            (rho_large, sp.Abs(b) >= 1),  # |b_new| ≥ 1
            (rho_small, True),  # |b_new|  < 1
        )

    def kendalls_tau(self):
        r"""
        Closed-form Kendall’s :math:`\tau(b_{\mathrm{new}})` (based on Prop. 3.5 with
        :math:`b_{\mathrm{old}} = 1/\lvert b_{\mathrm{new}}\rvert`).

        - If :math:`\lvert b_{\mathrm{new}}\rvert \ge 1` (i.e. :math:`\lvert b_{\mathrm{old}}\rvert \le 1`):

          .. math:: \tau = \operatorname{sgn}(b)\,\frac{6\lvert b\rvert^{2} - 4\lvert b\rvert + 1}{6\lvert b\rvert^{2}}.

        - If :math:`\lvert b_{\mathrm{new}}\rvert < 1` (i.e. :math:`\lvert b_{\mathrm{old}}\rvert \ge 1`):

          .. math:: \tau = \operatorname{sgn}(b)\,\frac{\lvert b\rvert(4-\lvert b\rvert)}{6}.
        """
        b = self.b
        b_abs = sp.Abs(b)

        # Case where |b_new| >= 1, which corresponds to b_old <= 1
        # Original formula: b_old * (4 - b_old) / 6
        tau_large_b = sp.sign(b) * (6 * b_abs**2 - 4 * b_abs + 1) / (6 * b_abs**2)

        # Case where |b_new| < 1, which corresponds to b_old > 1
        # Original formula: (6*b_old**2 - 4*b_old + 1) / (6*b_old**2)
        # = 1 - (4*b_old - 1) / (6*b_old**2)
        # = 1 - (4/|b| - 1) / (6/|b|**2) = 1 - (|b|*(4-|b|))/6
        tau_small_b = sp.sign(b) * (b_abs * (4 - b_abs)) / 6

        return sp.Piecewise(
            (tau_large_b, b_abs >= 1),
            (tau_small_b, True),
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
    #     b_old = 1 / b_abs
    #     L = sp.Max(0, s - b_old)
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
