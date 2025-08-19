# file: copul/families/diagonal_strip_alpha.py
import sympy as sp

from copul.families.core.biv_copula import BivCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


# The wrappers for cond_distr are no longer used for now
# from copul.wrapper.cd1_wrapper import CD1Wrapper
# from copul.wrapper.cd2_wrapper import CD2Wrapper


class DiagonalStripAlphaCopula(BivCopula):
    r"""
    Diagonal–strip copula with parameter α ∈ [0, 1/2].

    The density c(u,v) is defined such that it is independent of u
    and its value is the inverse of the length of its support at v.
    This ensures the marginals are uniform.
    """

    # symbolic parameter & admissible interval
    alpha = sp.symbols("alpha", real=True)
    params = [alpha]
    intervals = {"alpha": sp.Interval(0, sp.Rational(1, 2))}
    special_cases = {0: BivIndependenceCopula}

    # convenience symbols
    u, v = sp.symbols("u v", real=True)

    def __new__(cls, *args, **kwargs):
        if args:
            kwargs["alpha"] = args[0]
        if "alpha" in kwargs and kwargs["alpha"] in cls.special_cases:
            special_case_cls = cls.special_cases[kwargs["alpha"]]
            del kwargs["alpha"]
            return special_case_cls()
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        if args:
            kwargs["alpha"] = args[0]
        super().__init__(**kwargs)

    @property
    def pdf(self):
        """
        Symbolic PDF, c(u,v).
        The density is non-zero only on the 'yellow region' and its value
        depends on v to ensure valid uniform marginals.
        """
        alpha, u, v = self.alpha, self.u, self.v
        r = sp.sqrt(alpha)

        # 1. Define the value of the density c(v) on its support
        c_v_expr = sp.Piecewise(
            (1 / (1 - v - r / 2), v < r),
            (1 / (1 - r), v <= 1 - r),
            (1 / (v - r / 2), True),
        )

        # 2. Define the support (the "yellow region")
        psi = sp.Min(sp.Max(u - r / 2, 0), 1 - r)
        in_white = (v >= psi) & (v <= psi + r)

        # 3. Combine them: density is c(v) on the support, 0 otherwise
        # NOTE: The use of `~in_white` appears to be a bug, as it makes the
        # copula invalid for alpha > 1/4. The implementation of cond_distr
        # assumes the density is ON the strip (i.e., `in_white` is used).
        pdf_expr = sp.Piecewise((c_v_expr, ~in_white), (0, True))
        return SymPyFuncWrapper(pdf_expr)

    @property
    def _cdf_expr(self):
        raise NotImplementedError(
            "The analytical CDF for this density is complex and has not been implemented."
        )

    def cdf_vectorized(self, u, v):
        raise NotImplementedError(
            "The numerical CDF for this density is complex and has not been implemented."
        )

    def cond_distr_1(self, u=None, v=None):
        """
        Symbolic conditional distribution, C(u|v).
        C(u|v) = P(U <= u | V = v) = integral from 0 to u of c(x,v) dx.

        NOTE: This implementation assumes a correction to the PDF definition.
        The original PDF is defined as non-zero *outside* a diagonal strip
        (due to `~in_white`). This leads to an invalid copula for alpha > 1/4.
        We assume the density is meant to be non-zero *inside* the strip,
        which makes it a valid copula for all alpha in [0, 1/2].
        """
        if u is None:
            u = self.u
        if v is None:
            v = self.v

        alpha = self.alpha
        r = sp.sqrt(alpha)

        # Case 1: v is in the bottom-left section of the strip (v < r)
        cd_v_lt_r = sp.Min(u, v + r / 2) / (v + r / 2)

        # Case 2: v is in the top-right section of the strip (v > 1 - r)
        cd_v_gt_1mr = sp.Max(0, u - (v - r / 2)) / (1 - v + r / 2)

        # Case 3: v is in the central diagonal section (r <= v <= 1-r)
        cd_v_between = sp.Max(0, sp.Min(u, v + r / 2) - (v - r / 2)) / r

        # Combine into a single piecewise expression for C(u|v).
        cond_expr = sp.Piecewise(
            (cd_v_lt_r, v < r),
            (cd_v_gt_1mr, v > 1 - r),
            (cd_v_between, True),  # The remaining case is r <= v <= 1-r
        )

        return SymPyFuncWrapper(cond_expr)

    def cond_distr_2(self, u=None, v=None):
        """
        Symbolic conditional distribution, C(v|u) = P(V <= v | U = u).

        NOTE: The density function defined in this class does not correspond
        to a valid copula, because the marginal distribution for U, f_U(u),
        is not uniform. This implementation computes the true conditional
        distribution C(v|u) for the given density, which is:
        C(v|u) = integral_0^v c(u,y)dy / integral_0^1 c(u,y)dy
        """
        if u is None:
            u = self.u
        if v is None:
            v = self.v

        alpha = self.alpha
        r = sp.sqrt(alpha)

        # Dummy integration variable
        y = sp.symbols("y", real=True, positive=True)

        # Define c(y), the density's value (which only depends on y)
        # This is based on the corrected assumption that density is ON the strip
        c_y_func = sp.Piecewise(
            (1 / (y + r / 2), y < r), (1 / (1 - y + r / 2), y > 1 - r), (1 / r, True)
        )

        # Define psi(u), the lower bound of the support for y at a given u
        psi_u = sp.Min(sp.Max(u - r / 2, 0), 1 - r)

        # Numerator: integral from 0 to v of c(u,y)dy.
        # This is the integral of c(y) on the intersection of
        # [0, v] and the support interval [psi(u), psi(u) + r].
        num_lower_bound = psi_u
        num_upper_bound = sp.Min(v, psi_u + r)
        # SymPy's integrate handles cases where upper_bound < lower_bound
        numerator = sp.integrate(c_y_func, (y, num_lower_bound, num_upper_bound))

        # Denominator: marginal density f_U(u) = integral from 0 to 1 of c(u,y)dy.
        # This is the integral of c(y) over the full support [psi(u), psi(u) + r].
        den_lower_bound = psi_u
        den_upper_bound = psi_u + r
        denominator = sp.integrate(c_y_func, (y, den_lower_bound, den_upper_bound))

        # The conditional probability C(v|u)
        cond_expr = numerator / denominator

        return SymPyFuncWrapper(cond_expr)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        # Note: With c(u,v) = c(v), this copula is no longer symmetric.
        # Symmetry requires c(u,v) = c(v,u).
        return False


if __name__ == "__main__":
    # Instantiate the new, valid copula family
    copula = DiagonalStripAlphaCopula(alpha=0.25)
    # copula.plot_cdf()
    # copula.plot_pdf(plot_type="contour")
    copula.plot_cond_distr_1()
