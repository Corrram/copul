# file: copul/families/diagonal_band.py
import sympy as sp

from copul.families.core.biv_copula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper


class RhoMinusXiMaximalCopula(BivCopula):
    r"""
    Optimal–$\rho$ diagonal-band copula family
    \[
      \bigl\{C_x : 0<x<1\bigr\}.
    \]

    *Parameter* ``x``  
    adjusts the global quadratic mass in the optimisation problem
    (weak dependence for small `x`, perfect positive dependence as
    `x→1`).
    """

    # symbolic parameter & admissible interval
    x = sp.symbols("x", positive=True)
    params = [x]
    intervals = {"x": sp.Interval.open(0, 1)}  # 0 < x < 1

    # convenience symbols for the arguments
    u, v = sp.symbols("u v", positive=True)

    # ------------------------------------------------------------------
    # construction / parameter updates  (same pattern as FGM example)
    # ------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["x"] = args[0]
        self._validate_x(kwargs.get("x", self.x))
        BivCopula.__init__(self, **kwargs)

    def __call__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["x"] = args[0]
        if "x" in kwargs:
            self._validate_x(kwargs["x"])
        return BivCopula.__call__(self, **kwargs)

    @staticmethod
    def _validate_x(x_val):
        if not (0 < x_val < 1):
            raise ValueError(f"Parameter x must be in the open interval (0,1), got {x_val}")

    # ------------------------------------------------------------------
    # auxiliary explicit formulae  (taken from the proof)
    # ------------------------------------------------------------------
    @staticmethod
    def _b_expr(x):
        r"""Piece-wise explicit scale parameter \(b=b(x)\)."""
        xt = sp.Rational(3, 10)
        # Regime B (x ≤ 0.3): 0<b≤1
        b_B = (sp.Rational(5, 6)
               + sp.Rational(5, 3)
                 * sp.cos(sp.Rational(1, 3)
                          * sp.acos(1 - sp.Rational(108, 25) * x)
                          - 2 * sp.pi / 3))
        # Regime A (x > 0.3): b≥1
        b_A = (5 + sp.sqrt(5 * (6 * x - 1))) / (10 * (1 - x))
        return sp.Piecewise((b_B, x <= xt), (b_A, True))

    @staticmethod
    def _s_expr(v, b):
        r"""Shift \(s(v)\) (four cases, see proof)."""
        # decide via b<=1 instead of x
        big = sp.StrictGreaterThan(b, 1)

        def s_big(v):
            v1 = 1 / (2 * b)
            s1 = sp.sqrt(2 * v / b)
            s2 = v + 1 / (2 * b)
            s3 = 1 + 1 / b - sp.sqrt(2 * (1 - v) / b)
            return sp.Piecewise(
                (s1, v <= v1),
                (s2, v <= 1 - v1),
                (s3, True),
            )

        def s_small(v):
            s1 = sp.sqrt(2 * v / b)
            s2 = v / b + sp.Rational(1, 2)
            s3 = 1 + 1 / b - sp.sqrt(2 * (1 - v) / b)
            return sp.Piecewise(
                (s1, v <= b / 2),
                (s2, v <= 1 - b / 2),
                (s3, True),
            )

        return sp.Piecewise((s_big(v), big), (s_small(v), True))

    # ------------------------------------------------------------------
    # symbolic CDF  (cached for pdf / conditionals)
    # ------------------------------------------------------------------
    @property
    def cdf(self):
        u, v, x = self.u, self.v, self.x

        b = self._b_expr(x)
        s = self._s_expr(v, b)
        a = sp.Max(s - 1 / b, 0)        # length of plateau (possibly 0)
        t = sp.Min(s, 1)                # triangle tip (≤1)

        # ramp-integral over each t-region
        middle = a + b * (s * (u - a) - (u**2 - a**2) / 2)

        C = sp.Piecewise(
            (u, u <= a),          # plateau
            (middle, u <= t),     # ramp part
            (v, True),            # beyond the triangle tip
        )

        # cache the symbolic expression for later differentiation
        self._cdf_expr = C
        return SymPyFuncWrapper(C)

    # ------------------------------------------------------------------
    # conditional distributions & density
    # ------------------------------------------------------------------
    # def cond_distr_1(self, u=None, v=None):
    #     r"""\(F_{U|V}(u\mid v)=\partial_v C(u,v)\)."""
    #     expr = self.cdf.func.diff(self.v)
    #     return CD1Wrapper(expr)(u, v)

    # def cond_distr_2(self, u=None, v=None):
    #     r"""\(F_{V|U}(v\mid u)=\partial_u C(u,v)\)."""
    #     expr = self.cdf.func.diff(self.u)
    #     return CD2Wrapper(expr)(u, v)

    def pdf(self):
        r"""Joint density \(c(u,v)=\partial_{uv}^2 C(u,v)\)."""
        expr = self.cdf.func.diff(self.u).diff(self.v)
        return SymPyFuncWrapper(expr)

    # ------------------------------------------------------------------
    # absolutely continuous & symmetric flags
    # ------------------------------------------------------------------
    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True


if __name__ == "__main__":
    # Example usage
    x = 0.7
    C = RhoMinusXiMaximalCopula(x)
    # C.scatter_plot()
    # C.plot_cdf()
    C.plot_cond_distr_1()
    # C.plot_cond_distr_2()
    # C.plot_pdf() 
    C.plot_pdf(title=f"rho_minus_xi_maximal_copula (x={x})") 
    print("CDF at (0.5, 0.5):", C.cdf(0.5, 0.5))
    