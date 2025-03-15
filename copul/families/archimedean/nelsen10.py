import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Nelsen10(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)
    special_cases = {0: IndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return sympy.log(2 * self.t ** (-self.theta) - 1)

    @property
    def _raw_inv_generator(self):
        return (2 / (sympy.exp(self.y) + 1)) ** (1 / self.theta)

    @property
    def cdf(self):  # ToDo check why this differs from Nelsen cdf
        gen = (
            2
            * self.u**self.theta
            * self.v**self.theta
            / (
                self.u**self.theta * self.v**self.theta
                + (self.u**self.theta - 2) * (self.v**self.theta - 2)
            )
        ) ** (1 / self.theta)
        return SymPyFuncWrapper(gen)
