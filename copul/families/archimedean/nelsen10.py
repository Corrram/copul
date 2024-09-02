import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class Nelsen10(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return sympy.log(2 * self.t ** (-self.theta) - 1)

    def __call__(self, **kwargs):
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def inv_generator(self):
        gen = (2 / (sympy.exp(self.y) + 1)) ** (1 / self.theta)
        return SymPyFunctionWrapper(gen)

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
        return SymPyFunctionWrapper(gen)
