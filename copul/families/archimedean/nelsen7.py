import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other import IndependenceCopula, LowerFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Nelsen7(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)

    def __init__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return LowerFrechet()(**kwargs)
        if "theta" in kwargs and kwargs["theta"] == 1:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _generator(self):
        return -sympy.log(self.theta * self.t + 1 - self.theta)

    @property
    def inv_generator(self):  # ToDo multiply indicator function
        y = self.y
        ind = sympy.Heaviside(-y - sympy.log(1 - self.theta))
        gen = ind * (
            (self.theta * sympy.exp(y) - sympy.exp(y) + 1) * sympy.exp(-y) / self.theta
        )
        return SymPyFuncWrapper(gen)

    @property
    def cdf(self):
        u = self.u
        v = self.v
        theta = self.theta
        cdf = sympy.Max(theta * u * v + (1 - theta) * (u + v - 1), 0)
        return SymPyFuncWrapper(cdf)

    def chatterjees_xi(self, *args, **kwargs):
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

    def kendalls_tau(self, *args, **kwargs):
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
