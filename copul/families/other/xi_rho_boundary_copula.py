# file: copul/families/diagonal_band_b_inverse_reflected.py
import sympy as sp
import numpy as np

from copul.families.core.biv_copula import BivCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class XiRhoBoundaryCopula(BivCopula):
    r"""
    Optimal–ρ diagonal–band copula, parametrised by b_new so that the original
    scale‐parameter b_old = 1/|b_new|.  For b_new < 0, we use the reflection
    identity
    \[
      C_{b_{\rm new}}^{\downarrow}(u,v) \;=\; v \;-\;
      C_{|b_{\rm new}|}^{\uparrow}(1 - u,\,v)\,.
    \]

    -----------------
    Parameter b_new
    -----------------
    b_new ∈ ℝ \ {0}.  For b_new > 0, b_old = 1/b_new > 0; for b_new < 0,
    we treat |b_new| just as above and apply the “down‐reflection.”

    --------
    Formulas
    --------
    1. Maximal Spearman’s ρ:
      Let b := b_new.  Then b_old = 1/|b|.  Equivalently, one can write
      M(b) piecewise in terms of |b| just as in the “b_old‐param” version.
      We keep the same form, but with |b_old| ≤ 1 ↔ |b| ≥ 1, etc.  In symbolic
      form:
      \[
        M(b) \;=\;
        \begin{cases}
          b - \frac{3\,b^2}{10},
            & |b|\ge 1, \\[1ex]
          1 - \frac{1}{2\,b^2} + \frac{1}{5\,b^3},
            & |b| < 1.
        \end{cases}
      \]
      (Here b_old = 1/b_new simply swaps the roles of “small‐b_old” vs. “large‐b_old.”)

    2. Shift s_v(b):
      Define b_old = 1/|b|.  Then for |b_old| ≤ 1 (i.e. |b| ≥ 1):
      \[
        \begin{cases}
          s_v = \sqrt{2\,v\,b_{\text{old}}},
            & v \le \tfrac{b_{\text{old}}}{2},\\
          s_v = v + \tfrac{b_{\text{old}}}{2},
            & v \in (\tfrac{b_{\text{old}}}{2},\,1 - \tfrac{b_{\text{old}}}{2}],\\
          s_v = 1 + b_{\text{old}} - \sqrt{2\,b_{\text{old}}(1-v)},
            & v > 1 - \tfrac{b_{\text{old}}}{2}.
        \end{cases}
      \]
      For |b_old| > 1 (i.e. |b| < 1):
      \[
        \begin{cases}
          s_v = \sqrt{2\,v\,b_{\text{old}}},
            & v \le \tfrac{1}{2\,b_{\text{old}}},\\
          s_v = v\,b_{\text{old}} + \tfrac12,
            & v \in (\tfrac{1}{2\,b_{\text{old}}},\,1 - \tfrac{1}{2\,b_{\text{old}}}],\\
          s_v = 1 + b_{\text{old}} - \sqrt{2\,b_{\text{old}}(1-v)},
            & v > 1 - \tfrac{1}{2\,b_{\text{old}}}.
        \end{cases}
      \]

    3. Copula CDF:
      For b_new > 0, use the usual triangle‐band formula with b_old = 1/b_new:
      \[
        \;a_v \;=\; s_v - b_{\text{old}},
        \quad
        C(u,v) =
        \begin{cases}
          u,
            & u \le a_v,\\[0.6ex]
          a_v + \frac{2\,s_v\,(u - a_v) \;-\; u^2 + a_v^2}{2\,b_{\text{old}}},
            & a_v < u \le s_v,\\[1ex]
          v, & u > s_v.
        \end{cases}
      \]
      For b_new < 0, one sets
      \[
        C_{b_{\rm new}}(u,v) \;=\;
        v \;-\; C_{|b_{\rm new}|}\bigl(1 - u,\,v\bigr).
      \]
    """

    # symbolic parameter & admissible interval
    b = sp.symbols("b", real=True)
    params = [b]
    intervals = {"b": sp.Interval(-sp.oo, 0).union(sp.Interval(0, sp.oo))}
    special_cases = {0: BivIndependenceCopula}

    # convenience symbols
    u, v = sp.symbols("u v", positive=True)

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

    @classmethod
    def from_xi(cls, x):
        r"""
        Instantiates the copula from a target value for Chatterjee's xi.

        This method inverts the relationship between the parameter `b` and `xi`
        to find the `b` that produces the given `x`. This implementation
        assumes positive dependence (b > 0). The formula is:
        \[
            b_x =
            \begin{cases}
            \dfrac{\sqrt{6x}}{2\cos\left(\tfrac13\arccos\bigl(-\tfrac{3\sqrt{6x}}{5}\bigr)\right)},
              & 0<x\le\tfrac{3}{10},\\[4ex]
            \dfrac{5+\sqrt{5(6x-1)}}{10(1-x)},
              & \tfrac{3}{10}<x<1.
            \end{cases}
        \]

        Parameters
        ----------
        x : float or sympy expression
            The target value for Chatterjee's xi, in (0, 1).
        """
        if x == 0:
            return cls(
                b=0.0
            )  # Special case for xi = 0, which corresponds to independence
        elif x == 1:
            return UpperFrechet()
        elif x == -1:
            return LowerFrechet()
        x_sym = sp.sympify(x)

        # Case 1: 0 < x <= 3/10  (corresponds to |b| >= 1)
        b_ge_1 = sp.sqrt(6 * x_sym) / (
            2 * sp.cos(sp.acos(-3 * sp.sqrt(6 * x_sym) / 5) / 3)
        )

        # Case 2: 3/10 < x < 1  (corresponds to |b| < 1)
        b_lt_1 = (5 + sp.sqrt(5 * (6 * x_sym - 1))) / (10 * (1 - x_sym))

        # Create the piecewise expression for b
        b_expr = sp.Piecewise(
            (b_ge_1, (x_sym > 0) & (x_sym <= sp.Rational(3, 10))),
            (b_lt_1, (x_sym > sp.Rational(3, 10)) & (x_sym < 1)),
        )

        return cls(b=float(b_expr))

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
        """
        The “upright” CDF formula valid when b_new > 0.  Here b_old = 1/b_new.
        """
        b_old = 1 / b
        s = XiRhoBoundaryCopula._s_expr(v, b)
        a = sp.Max(s - b_old, 0)
        t = s
        middle = a + (2 * s * (u - a) - u**2 + a**2) / (2 * b_old)

        return sp.Piecewise(
            (u, u <= a),
            (middle, u <= t),
            (v, True),
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
        """Joint density c(u,v) = ∂²C/∂u∂v."""
        expr = self.cdf.func.diff(self.u).diff(self.v)
        return SymPyFuncWrapper(expr)

    # ===================================================================
    # START: Vectorized CDF implementation for performance improvement
    # ===================================================================

    @staticmethod
    def _s_expr_numpy(v, b):
        """
        Numpy-based vectorized implementation of the shift function s_v.
        This is a helper for `cdf_vectorized`.
        """
        v = np.asarray(v)
        b_old = 1 / np.abs(b)

        if np.abs(b) >= 1:  # Corresponds to |b_old| <= 1
            v1_s_s = b_old / 2
            s1_s_s = np.sqrt(2 * v * b_old)
            s2_s_s = v + b_old / 2
            s3_s_s = 1 + b_old - np.sqrt(2 * b_old * (1 - v))

            # np.select evaluates conditions in order, mimicking sympy.Piecewise
            return np.select(
                [v <= v1_s_s, v <= 1 - v1_s_s], [s1_s_s, s2_s_s], default=s3_s_s
            )
        else:  # Corresponds to |b_old| > 1
            v1_s_L = 1 / (2 * b_old)
            s1_s_L = np.sqrt(2 * v * b_old)
            s2_s_L = v * b_old + 0.5
            s3_s_L = 1 + b_old - np.sqrt(2 * b_old * (1 - v))

            return np.select(
                [v <= v1_s_L, v <= 1 - v1_s_L], [s1_s_L, s2_s_L], default=s3_s_L
            )

    @staticmethod
    def _base_cdf_numpy(u, v, b):
        """
        Numpy-based vectorized implementation of the base CDF for b > 0.
        This is a helper for `cdf_vectorized`.
        """
        u, v = np.asarray(u), np.asarray(v)
        b_old = 1 / b

        s = XiRhoBoundaryCopula._s_expr_numpy(v, b)
        a = np.maximum(s - b_old, 0)
        t = s

        middle = a + (2 * s * (u - a) - u**2 + a**2) / (2 * b_old)

        return np.select([u <= a, u <= t], [u, middle], default=v)

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

    def xi(self):
        """
        Closed-form ξ(b_new).  Recall b_old = 1/|b_new|, so the ‘≤ 1 / ≥ 1’
        conditions in Prop. 3.4 are swapped when we work in the new scale.

            • |b_new| ≥ 1  (⇔ |b_old| ≤ 1):
                  ξ = (1 / 10|b|²) · (5 − 2/|b|)

            • |b_new| < 1   (⇔ |b_old| ≥ 1):
                  ξ = 1 − |b| + (3/10) |b|²
        """
        b = 1 / self.b
        xi_large = (sp.Rational(1, 10) / sp.Abs(b) ** 2) * (5 - 2 / sp.Abs(b))
        xi_small = 1 - sp.Abs(b) + sp.Rational(3, 10) * sp.Abs(b) ** 2
        return sp.Piecewise(
            (xi_large, sp.Abs(b) >= 1),  # |b_new| ≥ 1
            (xi_small, True),  # |b_new|  < 1
        )

    # -------- ρ(b)  (Spearman’s rho) ----------------------------------- #
    def rho(self):
        """
        Closed-form ρ(b_new).  From Prop. 3.4 with the same change of
        parameter as above (b_old = 1/|b_new|):

            • |b_new| ≥ 1  (⇔ |b_old| ≤ 1):
                  ρ = sgn(b)·( 1/|b| − 3/(10|b|²) )

            • |b_new| < 1   (⇔ |b_old| ≥ 1):
                  ρ = sgn(b)·( 1 − |b|²/2 ) + |b|³/5
        """
        b = self.b
        rho_large = sp.sign(b) * (1 / sp.Abs(b) - sp.Rational(3, 10) / sp.Abs(b) ** 2)
        rho_small = sp.sign(b) * (1 - sp.Abs(b) ** 2 / 2) + sp.Abs(b) ** 3 / 5
        return sp.Piecewise(
            (rho_large, sp.Abs(b) >= 1),  # |b_new| ≥ 1
            (rho_small, True),  # |b_new|  < 1
        )

    def tau(self):
        """
        Closed-form τ(b_new). Based on Prop. 3.5 with b_old = 1/|b_new|.

            • |b_new| ≥ 1  (⇔ |b_old| ≤ 1):
                  τ = sgn(b) · (4|b| − 1) / (6|b|²)

            • |b_new| < 1   (⇔ |b_old| ≥ 1):
                  τ = sgn(b) · (1 - (4|b| - |b|²)/6)
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


if __name__ == "__main__":
    # Example usage
    copula = XiRhoBoundaryCopula(b=0.759)
    copula.plot_pdf(plot_type="contour")
    # copula.plot_cdf(plot_type="contour")
    print("CDF at (0.5, 0.5):", copula.cdf(0.5, 0.5))
    print("PDF at (0.5, 0.5):", copula.pdf(0.5, 0.5))
    print("xi:", copula.xi())
    print("rho:", copula.rho())
    print("tau:", copula.tau())
