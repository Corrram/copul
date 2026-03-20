# file: copul/families/xi_beta_boundary_copula.py
import sympy as sp
import numpy as np

from copul.family.core.biv_copula import BivCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class XiBetaBoundaryCopula(BivCopula):
    r"""
    Lower-boundary family for the exact region between Chatterjee's xi
    and Blomqvist's beta.

    This is the symmetric 2x2 checkerboard family with parameter
    :math:`b = \beta \in [-1,1]`, determined by the mass matrix

    .. math::

       \Delta_b
       =
       \begin{pmatrix}
         \frac{1+b}{4} & \frac{1-b}{4} \\
         \frac{1-b}{4} & \frac{1+b}{4}
       \end{pmatrix}.

    The associated density is piecewise constant on the four quadrants:

    .. math::

       c_b(u,v)=
       \begin{cases}
         1+b, & (u,v)\in[0,\tfrac12]^2 \cup (\tfrac12,1]^2,\\
         1-b, & (u,v)\in[0,\tfrac12]\times(\tfrac12,1]
                 \cup(\tfrac12,1]\times[0,\tfrac12],\\
       \end{cases}

    and the copula satisfies

    .. math::

       \beta(C_b)=b,
       \qquad
       \xi(C_b)=\frac{b^2}{2}.

    Thus this family traces the sharp lower boundary
    :math:`\xi = \beta^2/2` of the exact region.

    Special cases:
    - :math:`b=0`: independence copula.
    """

    b = sp.symbols("b", real=True)
    params = [b]
    intervals = {"b": sp.Interval(-1, 1)}
    special_cases = {0: BivIndependenceCopula}

    u, v = sp.symbols("u v", nonnegative=True)

    def __new__(cls, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["b"] = args[0]
        if "b" in kwargs and kwargs["b"] in cls.special_cases:
            special_case_cls = cls.special_cases[kwargs["b"]]
            del kwargs["b"]
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
            del kwargs["b"]
            return special_case_cls()
        return super().__call__(**kwargs)

    @property
    def _cdf_expr(self):
        b, u, v = self.b, self.u, self.v

        return sp.Piecewise(
            ((1 + b) * u * v, (u <= sp.Rational(1, 2)) & (v <= sp.Rational(1, 2))),
            (u * ((1 - b) * v + b), (u <= sp.Rational(1, 2)) & (v > sp.Rational(1, 2))),
            (v * ((1 - b) * u + b), (u > sp.Rational(1, 2)) & (v <= sp.Rational(1, 2))),
            (u * v + b * (1 - u) * (1 - v), True),
        )

    def _pdf_expr(self):
        b, u, v = self.b, self.u, self.v

        pdf = sp.Piecewise(
            (
                1 + b,
                ((u <= sp.Rational(1, 2)) & (v <= sp.Rational(1, 2)))
                | ((u > sp.Rational(1, 2)) & (v > sp.Rational(1, 2))),
            ),
            (1 - b, True),
        )
        return SymPyFuncWrapper(pdf)

    def cdf_vectorized(self, u, v):
        u = np.asarray(u)
        v = np.asarray(v)
        b = float(self.b)

        u, v = np.broadcast_arrays(u, v)

        q1 = (u <= 0.5) & (v <= 0.5)
        q2 = (u <= 0.5) & (v > 0.5)
        q3 = (u > 0.5) & (v <= 0.5)
        # q4 is the remaining region

        res = np.empty_like(u, dtype=float)

        res[q1] = (1.0 + b) * u[q1] * v[q1]
        res[q2] = u[q2] * ((1.0 - b) * v[q2] + b)
        res[q3] = v[q3] * ((1.0 - b) * u[q3] + b)

        q4 = ~(q1 | q2 | q3)
        res[q4] = u[q4] * v[q4] + b * (1.0 - u[q4]) * (1.0 - v[q4])

        return res

    def pdf_vectorized(self, u, v):
        u = np.asarray(u)
        v = np.asarray(v)
        b = float(self.b)

        u, v = np.broadcast_arrays(u, v)

        same_quadrant = ((u <= 0.5) & (v <= 0.5)) | ((u > 0.5) & (v > 0.5))
        return np.where(same_quadrant, 1.0 + b, 1.0 - b)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    def chatterjees_xi(self):
        r"""
        Closed-form Chatterjee's xi:

        .. math::

           \xi(C_b)=\frac{b^2}{2}.
        """
        return self.b**2 / 2

    def blomqvists_beta(self):
        r"""
        Closed-form Blomqvist's beta:

        .. math::

           \beta(C_b)=b.
        """
        return self.b

    def spearmans_rho(self):
        r"""
        Closed-form Spearman's rho:

        .. math::

           \rho(C_b)=\frac{3}{4}b.
        """
        return sp.Rational(3, 4) * self.b

    def kendalls_tau(self):
        r"""
        Closed-form Kendall's tau:

        .. math::

           \tau(C_b)=\frac{2}{3}b.
        """
        return sp.Rational(2, 3) * self.b

    def blests_nu(self, *args, **kwargs):
        return self.spearmans_rho()

    @classmethod
    def from_beta(cls, beta):
        beta = float(beta)
        if beta < -1 or beta > 1:
            raise ValueError("beta must lie in [-1,1].")
        return cls(b=beta)

    @classmethod
    def from_xi(cls, x, positive=True):
        r"""
        Instantiate from xi along one branch of the lower boundary
        :math:`\xi = b^2/2`.

        Parameters
        ----------
        x : float
            Value in [0, 1/2].
        positive : bool, default=True
            If True, choose b = +sqrt(2x); otherwise choose b = -sqrt(2x).
        """
        x = float(x)
        if x < 0 or x > 0.5:
            raise ValueError("For this boundary family, xi must lie in [0, 1/2].")

        b = np.sqrt(2.0 * x)
        if not positive:
            b = -b
        return cls(b=float(b))


if __name__ == "__main__":
    #XiBetaBoundaryCopula(b=-1).plot_cond_distr_1(plot_type="contour", levels=500)
    XiBetaBoundaryCopula(b=-0.7).plot_cond_distr_1(plot_type="contour", levels=500)
    # XiBetaBoundaryCopula(b=-0.5).plot_cond_distr_1(plot_type="contour", levels=500)
    XiBetaBoundaryCopula(b=-0.3).plot_cond_distr_1(plot_type="contour", levels=500)
    XiBetaBoundaryCopula(b=0.3).plot_cond_distr_1(plot_type="contour", levels=500)
    # XiBetaBoundaryCopula(b=0.5).plot_cond_distr_1(plot_type="contour", levels=500)
    XiBetaBoundaryCopula(b=0.7).plot_cond_distr_1(plot_type="contour", levels=500)
    # XiBetaBoundaryCopula(b=1).plot_cond_distr_1(plot_type="contour", levels=500)
