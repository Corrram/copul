import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


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

    # def _rho_int_2(self):
    #     v = self.v
    #     theta = self.theta
    #     n1 = (v**theta - 2) * 2 * theta * (v**theta - 1)
    #     n2 = (
    #         (v**theta - 2) * v**theta * (2 * (v**theta - 1) / (v**theta - 2)) ** (theta + 1)
    #     )
    #     n3 = (v**theta - 2) * v**theta
    #     n4 = 4 * (v**theta - 1) * (theta + 1) * (v**theta - 2)
    #     n5 = 4 * (v**theta - 1) ** 2 * (2 * (v**theta - 1) / (v**theta - 2)) ** theta
    #     n6 = 4 * (v**theta - 1) * (theta + 1)
    #     d = (theta + 1) * (v**theta - 2)
    #     n5_divided_by_d = (
    #         2
    #         * ((2 * v**theta - 2) / (v**theta - 2)) ** (theta + 1)
    #         * (v**theta - 1)
    #         / (theta + 1)
    #     )
    #     simplified_n2_n5 = (2 * (v**theta - 1)) ** (theta + 1) / (
    #         (v**theta - 2) ** theta * (theta + 1)
    #     )
    #     integrand = (
    #         sympy.simplify(n1 / d)
    #         # + sympy.simplify(simplified_n2_n5)
    #         + sympy.simplify(n3 / d)
    #         - sympy.simplify(n4 / d)
    #         + sympy.simplify(n6 / d)
    #     )
    #     return sympy.simplify(sympy.integrate(integrand, (v, 0, 1))) + sympy.Integral(
    #         simplified_n2_n5, (v, 0, 1)
    #     )
    #
    # def _rho(self):
    #     v = self.v
    #     theta = self.theta
    #     return (
    #         12
    #         * (
    #             sympy.Integral(
    #                 (2 * v**theta - 2) ** (theta + 1) / ((theta + 1) * (v**theta - 2) ** theta),
    #                 (v, 0, 1),
    #             )
    #             + (
    #                 2 * theta**3
    #                 + 2 * theta**2 * sympy.lerchphi(1 / 2, 1, 1 / theta)
    #                 - 2 * theta**2 * sympy.lerchphi(1 / 2, 1, (theta + 1) / theta)
    #                 + 4 * theta**2
    #                 + 4 * theta * sympy.lerchphi(1 / 2, 1, 1 / theta)
    #                 - 4 * theta * sympy.lerchphi(1 / 2, 1, (theta + 1) / theta)
    #                 + theta
    #                 + 2 * sympy.lerchphi(1 / 2, 1, 1 / theta)
    #                 - 2 * sympy.lerchphi(1 / 2, 1, (theta + 1) / theta)
    #             )
    #             / (theta * (theta + 1) ** 2)
    #         )
    #         - 3
    #     )

    def cond_distr_2(self):
        u = self.u
        v = self.v
        theta = self.theta
        cond_distr = (
            v ** (theta - 1)
            * (2 - u**theta)
            * sympy.Heaviside(u**theta * v**theta - 2 * (u**theta - 1) * (v**theta - 1))
            * sympy.Max(0, u**theta * v**theta - 2 * (u**theta - 1) * (v**theta - 1))
            ** ((1 - theta) / theta)
        )
        return SymPyFunctionWrapper(cond_distr)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
