import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.archimedean.nelsen1 import PiOverSigmaMinusPi
from copul.sympy_wrapper import SymPyFunctionWrapper


class AliMikhailHaq(ArchimedeanCopula):
    """
    Ali-Mikhail-Haq copula (Nelsen 3)
    """

    ac = ArchimedeanCopula
    theta_interval = sympy.Interval(-1, 1, left_open=False, right_open=True)

    def __str__(self):
        return super().__str__()

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return sympy.log((1 - self.theta * (1 - self.t)) / self.t)

    def __call__(self, **kwargs):
        if "theta" in kwargs and kwargs["theta"] == 1:
            new_copula = PiOverSigmaMinusPi()
            return new_copula(**kwargs)
        return super().__call__(**kwargs)

    @property
    def inv_generator(self):
        theta = self.theta
        gen = (theta - 1) / (theta - sympy.exp(self.y))
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self):
        u = self.u
        v = self.v
        theta = self.theta
        cdf = (u * v) / (1 - theta * (1 - u) * (1 - v))
        return SymPyFunctionWrapper(cdf)

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        cond_distr_1 = (
            self.v
            * (theta * self.u * (self.v - 1) - theta * (self.u - 1) * (self.v - 1) + 1)
            / (theta * (self.u - 1) * (self.v - 1) - 1) ** 2
        )
        return SymPyFunctionWrapper(cond_distr_1)(u, v)

    def spearmans_rho(self):
        th = self.theta
        t = sympy.symbols("t")
        integral = sympy.integrate(sympy.log(t) / (1 - t), (t, 1, 1 - th))
        return (
            12 * (1 + th) * integral - 24 * (1 - th) * sympy.log(1 - th)
        ) / th**2 - 3 * (th + 12) / th

    def kendalls_tau(self):
        theta = self.theta
        return (
            1
            - 2 / (3 * theta)
            - 2 * (1 - theta) ** 2 / (3 * theta**2) * sympy.log(1 - theta)
        )

    def chatterjees_xi(self):
        theta = self.theta
        return (
            3 / theta
            - theta / 6
            - 2 / 3
            - 2 / theta**2
            - 2 * (1 - theta) ** 2 * sympy.log(1 - theta) / theta**3
        )


Nelsen3 = AliMikhailHaq
