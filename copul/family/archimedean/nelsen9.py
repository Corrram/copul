import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula


class GumbelBarnett(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)
    special_cases = {0: BivIndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return sympy.log(1 - self.theta * sympy.log(self.t))

    @property
    def _raw_inv_generator(self):
        return sympy.exp((1 - sympy.exp(self.y)) / self.theta)

    @property
    def _cdf_expr(self):
        return (
            self.u
            * self.v
            * sympy.exp(-self.theta * sympy.log(self.u) * sympy.log(self.v))
        )

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
