# file: copul/families/xi_counterexample_copula.py
import sympy as sp
import numpy as np

from copul.family.core.biv_copula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class XiCounterexampleCopula(BivCopula):
    r"""
    Absolutely continuous copula used as a counterexample to the unrestricted
    checkerboard lower-bound claim for Chatterjee's xi.

    The density is

    .. math::

        c(u,v)=
        \begin{cases}
            f_1(v), & 0 \le u < 1/2,\\
            f_2(v), & 1/2 \le u \le 1,
        \end{cases}

    where

    .. math::

        f_1(v)=
        \begin{cases}
            1, & 0 \le v < 1/4,\\
            2, & 1/4 \le v < 1/2,\\
            0, & 1/2 \le v < 3/4,\\
            1, & 3/4 \le v \le 1,
        \end{cases}
        \qquad
        f_2(v)=2-f_1(v).

    This copula has uniform marginals and is associated with the checkerboard
    matrix

    .. math::

        \Delta =
        \begin{pmatrix}
            3/8 & 1/8\\
            1/8 & 3/8
        \end{pmatrix}.

    Its Chatterjee xi is

    .. math::

        \xi(C)=1/16,

    while the associated 2x2 checkerboard copula has

    .. math::

        \xi(C^\Delta_\Pi)=1/8.
    """

    u, v = sp.symbols("u v", nonnegative=True)

    params = []
    intervals = {}
    special_cases = {}

    @staticmethod
    def _f1_expr(v):
        return sp.Piecewise(
            (1, v < sp.Rational(1, 4)),
            (2, v < sp.Rational(1, 2)),
            (0, v < sp.Rational(3, 4)),
            (1, True),
        )

    @staticmethod
    def _f2_expr(v):
        return 2 - XiCounterexampleCopula._f1_expr(v)

    @staticmethod
    def _F1_expr(v):
        r"""Primitive F_1(v)=\int_0^v f_1(t)dt."""
        return sp.Piecewise(
            (v, v < sp.Rational(1, 4)),
            (2 * v - sp.Rational(1, 4), v < sp.Rational(1, 2)),
            (sp.Rational(3, 4), v < sp.Rational(3, 4)),
            (v, True),
        )

    @staticmethod
    def _F2_expr(v):
        r"""Primitive F_2(v)=\int_0^v f_2(t)dt = 2v-F_1(v)."""
        return 2 * v - XiCounterexampleCopula._F1_expr(v)

    @property
    def _cdf_expr(self):
        u, v = self.u, self.v
        F1 = self._F1_expr(v)
        F2 = self._F2_expr(v)

        return sp.Piecewise(
            (u * F1, u < sp.Rational(1, 2)),
            (sp.Rational(1, 2) * F1 + (u - sp.Rational(1, 2)) * F2, True),
        )

    def _pdf_expr(self):
        u, v = self.u, self.v
        f1 = self._f1_expr(v)
        f2 = self._f2_expr(v)

        pdf = sp.Piecewise(
            (f1, u < sp.Rational(1, 2)),
            (f2, True),
        )
        return SymPyFuncWrapper(pdf)

    @staticmethod
    def _f1_numpy(v):
        v = np.asarray(v)
        return np.select(
            [
                v < 0.25,
                v < 0.5,
                v < 0.75,
            ],
            [
                1.0,
                2.0,
                0.0,
            ],
            default=1.0,
        )

    @staticmethod
    def _F1_numpy(v):
        v = np.asarray(v)
        return np.select(
            [
                v < 0.25,
                v < 0.5,
                v < 0.75,
            ],
            [
                v,
                2.0 * v - 0.25,
                0.75,
            ],
            default=v,
        )

    def pdf_vectorized(self, u, v):
        u = np.asarray(u)
        v = np.asarray(v)

        if u.shape != v.shape:
            u, v = np.broadcast_arrays(u, v)

        f1 = self._f1_numpy(v)
        f2 = 2.0 - f1

        return np.where(u < 0.5, f1, f2)

    def cdf_vectorized(self, u, v):
        u = np.asarray(u)
        v = np.asarray(v)

        if u.shape != v.shape:
            u, v = np.broadcast_arrays(u, v)

        F1 = self._F1_numpy(v)
        F2 = 2.0 * v - F1

        return np.where(
            u < 0.5,
            u * F1,
            0.5 * F1 + (u - 0.5) * F2,
        )

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    def chatterjees_xi(self):
        return sp.Rational(1, 16)

    def spearmans_rho(self):
        return sp.Rational(3, 16)

    def kendalls_tau(self):
        return sp.Rational(1, 8)

    def checkerboard_matrix(self):
        return sp.Matrix(
            [
                [sp.Rational(3, 8), sp.Rational(1, 8)],
                [sp.Rational(1, 8), sp.Rational(3, 8)],
            ]
        )

    def checkerboard_xi(self):
        r"""
        Chatterjee's xi of the associated 2x2 checkerboard copula.
        """
        return sp.Rational(1, 8)


if __name__ == "__main__":
    cop = XiCounterexampleCopula()
    print("is copula: ", cop.is_copula())

    print("xi(C) =", cop.chatterjees_xi())
    print("rho(C) =", cop.spearmans_rho())
    print("tau(C) =", cop.kendalls_tau())
    print("xi(C_Pi^Delta) =", cop.checkerboard_xi())
    check = cop.to_checkerboard(2)
    print("xi(C) =", check.chatterjees_xi())
    print("rho(C) =", check.spearmans_rho())
    print("tau(C) =", check.kendalls_tau())
    cop.plot_pdf()
    check.plot_pdf()
    # Optional visual check if your plotting methods are available:
    # cop.plot_pdf(plot_type="contour", levels=20)