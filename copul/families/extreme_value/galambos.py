import numpy as np
import sympy

from copul.cdf_wrapper import CDFWrapper
from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class Galambos(ExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sympy.symbols("delta", positive=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, np.inf, left_open=True, right_open=True)}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def pickands(self):
        t = self.t
        delta = self.delta
        func = 1 - (t ** (-delta) + (1 - t) ** (-delta)) ** (-1 / delta)
        return SymPyFunctionWrapper(func)

    @property
    def cdf(self):
        u = self.u
        delta = self.delta
        v = self.v
        cdf = (
            u
            * v
            * sympy.exp(
                (sympy.log(1 / u) ** (-delta) + sympy.log(1 / v) ** (-delta))
                ** (-1 / delta)
            )
        )
        return CDFWrapper(cdf)

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
        return SymPyFunctionWrapper(result)

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
