# file: copul/families/diagonal_band_b_inverse_reflected.py
import sympy as sp

from copul.families.core.biv_copula import BivCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class RhoMinusXiMaximalCopula(BivCopula):
    r"""
    Optimal–ρ diagonal–band copula, parametrised by b_new so that the original
    scale‐parameter b_old = 1/|b_new|.  For b_new < 0, we use the reflection
    identity
    \[
      C_{b_{\rm new}}^{\downarrow}(u,v) \;=\; v \;-\; 
      C_{|b_{\rm new}|}^{\uparrow}(1 - u,\,v)\,.
    \]

    -----------------
    Parameter b_new
    -----------------
    b_new ∈ ℝ \ {0}.  For b_new > 0, b_old = 1/b_new > 0; for b_new < 0,
    we treat |b_new| just as above and apply the “down‐reflection.”

    --------
    Formulas
    --------
    1. Maximal Spearman’s ρ:
      Let b := b_new.  Then b_old = 1/|b|.  Equivalently, one can write
      M(b) piecewise in terms of |b| just as in the “b_old‐param” version.
      We keep the same form, but with |b_old| ≤ 1 ↔ |b| ≥ 1, etc.  In symbolic
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
      Define b_old = 1/|b|.  Then for |b_old| ≤ 1 (i.e. |b| ≥ 1):
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
        s = RhoMinusXiMaximalCopula._s_expr(v, b)
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

    # -------- Metadata -------- #
    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True


# ---------------- Demo ---------------- #
if __name__ == "__main__":
    # Example usage: pick any nonzero b_new
    for b_val in [0.5, 1, 5]:
        C = RhoMinusXiMaximalCopula(b_val)
        # C.plot_pdf(plot_type="contour", log_z=True, grid_size=2000)
        # C.plot_cond_distr_1(plot_type="functions", title=None, zlabel=None, xlabel="t")
        C.scatter_plot()
    # Optionally, contour‐plot the PDF for a given b_new:
    # C.plot_pdf(title=f"Contour of PDF for b={b_val}", plot_type="contour", log_z=True, grid_size=200)
