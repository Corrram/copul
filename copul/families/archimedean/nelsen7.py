import sympy

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.other import BivIndependenceCopula, LowerFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper


class Nelsen7(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)
    special_cases = {0: LowerFrechet, 1: BivIndependenceCopula}

    @property
    def _generator_at_0(self):
        return sympy.log(1 / (1 - self.theta))

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _raw_generator(self):
        return -sympy.log(self.theta * self.t + 1 - self.theta)

    @property
    def _raw_inv_generator(self):  # ToDo multiply indicator function
        y = self.y
        ind = sympy.Heaviside(-y - sympy.log(1 - self.theta))
        return ind * (
            (self.theta * sympy.exp(y) - sympy.exp(y) + 1) * sympy.exp(-y) / self.theta
        )

    @property
    def _cdf_expr(self):
        u = self.u
        v = self.v
        theta = self.theta
        return sympy.Max(theta * u * v + (1 - theta) * (u + v - 1), 0)

    def xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return 1 - self.theta

    def cond_distr_1(self, u=None, v=None):
        diff = (self.theta * self.v - self.theta + 1) * sympy.Heaviside(
            self.theta * self.u * self.v + (1 - self.theta) * (self.u + self.v - 1)
        )
        return CD1Wrapper(diff)(u, v)

    def cond_distr_2(self, u=None, v=None):
        diff = (self.theta * self.u - self.theta + 1) * sympy.Heaviside(
            self.theta * self.u * self.v + (1 - self.theta) * (self.u + self.v - 1)
        )
        return CD2Wrapper(diff)(u, v)

    def _rho_int_1(self):
        theta = self.theta
        v = self.v
        u = self.u
        lower_border = (theta - 1) * (v - 1) / (1 + theta * (v - 1))
        integrand = theta * u * v - (theta - 1) * (u + v - 1)
        integral = sympy.integrate(integrand, (u, lower_border, 1))
        return sympy.simplify(integral)

    def _tau_int_1(self):
        theta = self.theta
        v = self.v
        u = self.u
        lower_border = (theta - 1) * (v - 1) / (1 + theta * (v - 1))
        positive_cdf = theta * u * v - (theta - 1) * (u + v - 1)
        integrand = positive_cdf * sympy.diff(sympy.diff(positive_cdf, v), u)
        integral = sympy.integrate(integrand, (u, lower_border, 1))
        return sympy.simplify(integral)

    def _rho(self):
        theta = self.theta
        if theta == 0:
            return -1
        elif theta == 1:
            return 0
        rho = (
            -3
            + 9 / theta
            - 6 / theta**2
            - 6 * (theta - 1) ** 2 * sympy.log(1 - theta) / theta**3
        )
        return sympy.Piecewise((rho, theta < 1), (0, True))

    def tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        theta = self.theta
        if theta == 0:
            return -1
        elif theta == 1:
            return 0
        tau = 2 - 2 / theta - 2 * (theta - 1) ** 2 * sympy.log(1 - theta) / theta**2
        return sympy.Piecewise((tau, theta < 1), (0, True))

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
