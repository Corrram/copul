import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.cd2_wrapper import CD2Wrapper


class Nelsen11(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 0.5, left_open=False, right_open=False)
    special_cases = {0: BivIndependenceCopula}
    _generator_at_0 = sympy.log(2)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _raw_generator(self):
        return sympy.log(2 - self.t**self.theta)

    @property
    def _raw_inv_generator(self):
        ind = sympy.Piecewise((1, self.y <= sympy.log(2)), (0, True))
        return (2 - sympy.exp(self.y)) ** (1 / self.theta) * ind

    @property
    def _cdf_expr(self):
        return sympy.Max(
            self.u**self.theta * self.v**self.theta
            - 2 * (1 - self.u**self.theta) * (1 - self.v**self.theta),
            0,
        ) ** (1 / self.theta)

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
