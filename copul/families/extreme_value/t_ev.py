import numpy as np
import sympy
from sympy import stats

from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


# noinspection PyPep8Naming
class tEV(ExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        pass

    rho = sympy.symbols("rho")
    nu = sympy.symbols("nu", positive=True)
    params = [nu, rho]
    intervals = {
        "nu": sympy.Interval(0, np.inf, left_open=True, right_open=True),
        "rho": sympy.Interval(-1, 1, left_open=True, right_open=True),
    }

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _pickands(self):
        def z(t):
            return (
                (1 + self.nu) ** (1 / 2)
                * ((t / (1 - t)) ** (1 / self.nu) - self.rho)
                * (1 - self.rho**2) ** (-1 / 2)
            )

        student_t = stats.StudentT("x", self.nu + 1)
        return (1 - self.t) * stats.cdf(student_t)(z(1 - self.t)) + self.t * stats.cdf(
            student_t
        )(z(self.t))

    # @property
    # def pdf(self):
    #     u = self.u
    #     v = self.v
    #     result = None
    #     return SymPyFunctionWrapper(result)
