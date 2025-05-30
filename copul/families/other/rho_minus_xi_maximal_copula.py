# file: copul/families/diagonal_band.py
import sympy as sp

from copul.families.core.biv_copula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class RhoMinusXiMaximalCopula(BivCopula):
    r"""
    Optimal–$\rho$ diagonal-band copula family
    \[
      \bigl\{C_x : 0<x<1\bigr\},
    \]
    now with
    \[
      b_x =
      \begin{cases}
        \frac{2}{\sqrt{6x}}
          \cos\!\Bigl(\tfrac13\arccos\!\bigl(-\tfrac{3\sqrt{6x}}{5}\bigr)\Bigr),
          & 0<x\le 0.3,\\[1ex]
        \frac{5-\sqrt{5(6x-1)}}{3}, & 0.3< x <1,
      \end{cases}
    \]
    and
    \[
      M_x =
      \begin{cases}
        \displaystyle
        \frac{1}{b_x} - \frac{3}{10\,b_x^2}, & x\le 0.3,\\[1ex]
        1 - \frac{b_x^2}{2} + \frac{b_x^3}{5}, & x>0.3.
      \end{cases}
    \]
    The max‐ and min‐band parameters are
    \[
      s_v = 
      \begin{cases}
        \sqrt{2vb_x}, & \text{region 1},\\
        v+\tfrac{b_x}{2}, & \text{region 2},\\
        1 + b_x - \sqrt{2b_x(1-v)}, & \text{region 3},
      \end{cases}
      \quad
      a_v = s_v - b_x,
    \]
    and
    \[
      C_x(u,v) =
      \begin{cases}
        u, & u\le a_v,\\
        a_v + \frac{2s_v(u-a_v)-u^2 - a_v^2}{2\,b_x}, & a_v < u \le s_v,\\
        v, & u>s_v.
      \end{cases}
    \]
    """

    # symbolic parameter & admissible interval
    x = sp.symbols("x", positive=True)
    params = [x]
    intervals = {"x": sp.Interval.open(0, 1)}

    # convenience symbols
    u, v = sp.symbols("u v", positive=True)

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["x"] = args[0]
        self._validate_x(kwargs.get("x", self.x))
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["x"] = args[0]
        if "x" in kwargs:
            self._validate_x(kwargs["x"])
        return super().__call__(**kwargs)

    @staticmethod
    def _validate_x(x_val):
        if not (0 < x_val < 1):
            raise ValueError(f"Parameter x must be in (0,1), got {x_val}")

    @staticmethod
    def _b_expr(x):
        """Piece‐wise explicit scale parameter b_x as in (2.10)."""
        xt = sp.Rational(3, 10)
        # Regime B: 0<x<=0.3
        b_B = (2 / sp.sqrt(6 * x)) * sp.cos(
            sp.Rational(1, 3) * sp.acos(-3 * sp.sqrt(6 * x) / 5)
        )
        # Regime A: 0.3 < x < 1
        b_A = (5 - sp.sqrt(5 * (6 * x - 1))) / 3
        return sp.Piecewise((b_B, x <= xt), (b_A, True))

    @staticmethod
    def _M_expr(x):
        """Maximal Spearman's rho M_x for a given x as in (2.11)."""
        xt = sp.Rational(3, 10)
        b = RhoMinusXiMaximalCopula._b_expr(x)
        M_B = 1 / b - sp.Rational(3, 10) / b**2
        M_A = 1 - b**2 / 2 + b**3 / 5
        return sp.Piecewise((M_B, x <= xt), (M_A, True))

    @staticmethod
    def _s_expr(v, b):
        """Shift s_v (max‐band) as in (2.12)."""
        # small‐b region: b <= 1
        v1_s = b / 2
        s1_s = sp.sqrt(2 * v * b)
        s2_s = v + b / 2
        s3_s = 1 + b - sp.sqrt(2 * b * (1 - v))
        s_small = sp.Piecewise(
            (s1_s, v <= v1_s),
            (s2_s, v <= 1 - v1_s),
            (s3_s, True),
        )

        # large‐b region: b > 1
        v1_L = 1 / (2 * b)
        s1_L = sp.sqrt(2 * v * b)
        s2_L = v * b + sp.Rational(1, 2)
        s3_L = 1 + b - sp.sqrt(2 * b * (1 - v))
        s_large = sp.Piecewise(
            (s1_L, v <= v1_L),
            (s2_L, v <= 1 - v1_L),
            (s3_L, True),
        )

        return sp.Piecewise((s_small, b <= 1), (s_large, True))

    @property
    def cdf(self):
        u, v, x = self.u, self.v, self.x
        b = self._b_expr(x)
        s = self._s_expr(v, b)
        a = sp.Max(s - b, 0)      # a_v = s_v - b_x
        t = s                     # triangle tip
        middle = a + (2 * s * (u - a) - u**2 - a**2) / (2 * b)

        C = sp.Piecewise(
            (u, u <= a),
            (middle, u <= t),
            (v, True),
        )

        # cache for subsequent differentiation
        self._cdf_expr = C
        return SymPyFuncWrapper(C)

    def pdf(self):
        """Joint density c(u,v) = ∂²C/∂u∂v."""
        expr = self.cdf.func.diff(self.u).diff(self.v)
        return SymPyFuncWrapper(expr)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True


if __name__ == "__main__":
    x_val = 0.1
    C = RhoMinusXiMaximalCopula(x_val)

    # Example: evaluate CDF, PDF, or plot
    print("b(x) =", C._b_expr(x_val))
    print("M(x) =", C._M_expr(x_val))
    print("CDF at (0.5,0.5) =", C.cdf(0.5, 0.5))
    C.plot_pdf(title=f"Contour of PDF for x={x_val}", plot_type="contour", log_z=True, grid_size=200)
