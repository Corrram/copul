import sympy

from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class Nelsen11(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 0.5, left_open=False, right_open=False)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _generator(self):
        return sympy.log(2 - self.t**self.theta)

    def __call__(self, **kwargs):
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def inv_generator(self):
        ind = sympy.Piecewise((1, self.y <= sympy.log(2)), (0, True))
        gen = (2 - sympy.exp(self.y)) ** (1 / self.theta) * ind
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self):
        cdf = sympy.Max(
            self.u**self.theta * self.v**self.theta
            - 2 * (1 - self.u**self.theta) * (1 - self.v**self.theta),
            0,
        ) ** (1 / self.theta)
        return SymPyFunctionWrapper(cdf)

    def _rho_int_1(self):
        u = self.u
        v = self.v
        theta = self.theta
        integrand = u**theta * v**theta - (1 - v**theta) * (2 - 2 * u**theta)
        lower_limit = 2 * (1 - v**theta) / (v**theta - 2 * (v**theta - 1))
        return sympy.simplify(sympy.integrate(integrand, (u, lower_limit, 1)))

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        cond_distr = (
            self.v ** (theta - 1)
            * (2 - self.u**theta)
            * sympy.Heaviside(
                self.u**theta * self.v**theta
                - 2 * (self.u**theta - 1) * (self.v**theta - 1)
            )
            * sympy.Max(
                0,
                self.u**theta * self.v**theta
                - 2 * (self.u**theta - 1) * (self.v**theta - 1),
            )
            ** ((1 - theta) / theta)
        )
        return CD2Wrapper(cond_distr)(u, v)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
