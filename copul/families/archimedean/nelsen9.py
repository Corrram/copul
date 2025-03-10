import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class GumbelBarnett(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)
    special_cases = {0: IndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return sympy.log(1 - self.theta * sympy.log(self.t))

    @property
    def inv_generator(self):
        gen = sympy.exp((1 - sympy.exp(self.y)) / self.theta)
        return SymPyFuncWrapper(gen)

    @property
    def cdf(self):
        cdf = (
            self.u
            * self.v
            * sympy.exp(-self.theta * sympy.log(self.u) * sympy.log(self.v))
        )
        return SymPyFuncWrapper(cdf)

    def _xi_int_1(self, v):
        theta = self.theta
        return v**2 * (theta * sympy.log(v) - 1) ** 2 / (1 - 2 * theta * sympy.log(v))

    def _xi_int_2(self):
        theta = self.theta
        return (
            1
            / 72
            * (
                18
                + 4 * theta
                - 9 * sympy.exp(3 / (2 * theta)) * sympy.Ei(-3 / (2 * theta)) / theta
            )
        )

    def _rho_int_1(self):
        return -self.v / (self.theta * sympy.log(self.v) - 2)

    def _rho_int_2(self):
        theta = self.theta
        v = self.v
        integral = (
            -sympy.exp(4 / theta) * sympy.Ei(2 * sympy.log(v) - 4 / theta) / theta
        )
        return integral.subs(v, 1)  # todo check if this line is correct


Nelsen9 = GumbelBarnett
