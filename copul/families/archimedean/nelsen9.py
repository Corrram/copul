import sympy

from copul.cd1_wrapper import CD1Wrapper
from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class GumbelBarnett(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return sympy.log(1 - self.theta * sympy.log(self.t))

    def __call__(self, **kwargs):
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def inv_generator(self):
        gen = sympy.exp((1 - sympy.exp(self.y)) / self.theta)
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self):
        cdf = (
            self.u
            * self.v
            * sympy.exp(-self.theta * sympy.log(self.u) * sympy.log(self.v))
        )
        return SymPyFunctionWrapper(cdf)

    def _squared_cond_distr_1(self, u, v):
        theta = self.theta
        return v**2 * u ** (-2 * theta * sympy.log(v)) * (sympy.log(v**theta) - 1) ** 2

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
