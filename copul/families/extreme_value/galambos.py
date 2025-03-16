import numpy as np
import sympy

from copul.families.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Galambos(BivExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sympy.symbols("delta", positive=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, np.inf, left_open=True, right_open=True)}

    @property
    def _pickands(self):
        expr = 1 - (self.t ** (-self.delta) + (1 - self.t) ** (-self.delta)) ** (
            -1 / self.delta
        )
        return sympy.Piecewise(
            (1, sympy.Or(sympy.Eq(self.t, 0), sympy.Eq(self.t, 1))), (expr, True)
        )

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def cdf(self):
        u = self.u
        v = self.v
        delta = self.delta
        base_expr = (
            u
            * v
            * sympy.exp(
                (sympy.log(1 / u) ** (-delta) + sympy.log(1 / v) ** (-delta))
                ** (-1 / delta)
            )
        )
        # When u==0 or v==0, return 0; otherwise, use the computed expression.
        cdf_expr = sympy.Piecewise(
            (0, sympy.Or(sympy.Eq(u, 0), sympy.Eq(v, 0))), (base_expr, True)
        )
        return CDFWrapper(cdf_expr)

    @property
    def pdf(self):
        u = self.u
        v = self.v
        delta = self.delta
        sub_expr_3 = self._eval_sub_expr_3(delta, u, v)
        sub_expr = self._eval_sub_expr(delta, u, v)
        sub_expr_2 = self._eval_sub_expr_2(delta, u, v)
        result = (
            (u * v) ** ((sub_expr_3 ** (1 / delta) - 1) / sub_expr_3 ** (1 / delta))
            * (
                sub_expr_3 ** (1 / delta)
                * (delta + 1)
                * (
                    ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                    * (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        + (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                    * (sympy.log(v) - sympy.log(u * v)) ** 2
                    + (sympy.log(v) / sympy.log(u * v)) ** delta
                    * (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        + (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                    * sympy.log(v) ** 2
                    - sub_expr**2
                )
                + (sub_expr + sub_expr_2 * (sympy.log(v) - sympy.log(u * v)))
                * (sub_expr + sub_expr_2 * sympy.log(v))
                * sympy.log(u * v)
            )
            / (
                u
                * v
                * sub_expr_3 ** (2 / delta)
                * (
                    ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                    + (sympy.log(v) / sympy.log(u * v)) ** delta
                )
                ** 2
                * (sympy.log(v) - sympy.log(u * v))
                * sympy.log(v)
                * sympy.log(u * v)
            )
        )
        return SymPyFuncWrapper(result)

    def _eval_sub_expr_2(self, delta, u, v):
        return ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta + (
            sympy.log(v) / sympy.log(u * v)
        ) ** delta * (self._eval_sub_expr_3(delta, u, v) ** (1 / delta) - 1)

    def _eval_sub_expr(self, delta, u, v):
        return ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta * (
            sympy.log(v) - sympy.log(u * v)
        ) + ((sympy.log(v) / sympy.log(u * v)) ** delta) * sympy.log(v)

    def _eval_sub_expr_3(self, delta, u, v):
        return (
            ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
            + (sympy.log(v) / sympy.log(u * v)) ** delta
        ) / (
            ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
            * (sympy.log(v) / sympy.log(u * v)) ** delta
        )


# B7 = Galambos
