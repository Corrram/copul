from abc import abstractmethod

import sympy

from copul.families.copula import Copula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet
from copul.sympy_wrapper import SymPyFunctionWrapper


class EllipticalCopula(Copula):
    t = sympy.symbols("t", positive=True)
    generator = None
    rho = sympy.symbols("rho", real=True)
    params = [rho]
    intervals = {"rho": sympy.Interval(-1, 1, left_open=False, right_open=False)}

    def __call__(self, **kwargs):
        if "rho" in kwargs:
            if kwargs["rho"] == -1:
                del kwargs["rho"]
                return LowerFrechet()(**kwargs)
            elif kwargs["rho"] == 1:
                del kwargs["rho"]
                return UpperFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def corr_matrix(self):
        return sympy.Matrix([[1, self.rho], [self.rho, 1]])

    def characteristic_function(self, t1, t2):
        arg = (
            t1**2 * self.corr_matrix[0, 0]
            + t2**2 * self.corr_matrix[1, 1]
            + 2 * t1 * t2 * self.corr_matrix[0, 1]
        )
        return self.generator(arg)

    @property
    @abstractmethod
    def cdf(self) -> SymPyFunctionWrapper:
        pass
