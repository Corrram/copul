import sympy

from typing import TypeAlias

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class AliMikhailHaq(BivArchimedeanCopula):
    """
    Ali-Mikhail-Haq copula (Nelsen 3)
    """

    ac = BivArchimedeanCopula
    theta_interval = sympy.Interval(-1, 1, left_open=False, right_open=False)
    special_cases = {
        0: IndependenceCopula,
        1: PiOverSigmaMinusPi,
    }

    def __str__(self):
        return super().__str__()

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return sympy.log((1 - self.theta * (1 - self.t)) / self.t)

    @property
    def _raw_inv_generator(self):
        theta = self.theta
        return (theta - 1) / (theta - sympy.exp(self.y))

    @property
    def cdf(self):
        u = self.u
        v = self.v
        theta = self.theta
        cdf = (u * v) / (1 - theta * (1 - u) * (1 - v))
        return SymPyFuncWrapper(cdf)

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        cond_distr_1 = (
            self.v
            * (theta * self.u * (self.v - 1) - theta * (self.u - 1) * (self.v - 1) + 1)
            / (theta * (self.u - 1) * (self.v - 1) - 1) ** 2
        )
        return SymPyFuncWrapper(cond_distr_1)(u, v)

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        th = self.theta
        t = sympy.symbols("t")
        integral = sympy.integrate(sympy.log(t) / (1 - t), (t, 1, 1 - th))
        return (
            12 * (1 + th) * integral - 24 * (1 - th) * sympy.log(1 - th)
        ) / th**2 - 3 * (th + 12) / th

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        theta = self.theta
        return (
            1
            - 2 / (3 * theta)
            - 2 * (1 - theta) ** 2 / (3 * theta**2) * sympy.log(1 - theta)
        )

    def chatterjees_xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        theta = self.theta
        return (
            3 / theta
            - theta / 6
            - 2 / 3
            - 2 / theta**2
            - 2 * (1 - theta) ** 2 * sympy.log(1 - theta) / theta**3
        )


Nelsen3: TypeAlias = AliMikhailHaq
