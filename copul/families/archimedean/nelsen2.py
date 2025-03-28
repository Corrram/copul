import numpy as np
import sympy

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper


class Nelsen2(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)
    _generator_at_0 = 1

    def __str__(self):
        return super().__str__()

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _raw_generator(self):
        return (1 - self.t) ** self.theta

    @property
    def _raw_inv_generator(self):
        return sympy.Max(1 - self.y ** (1 / self.theta), 0)

    @property
    def _cdf_expr(self):
        expr = 1 - ((1 - self.u) ** self.theta + (1 - self.v) ** self.theta) ** (
            1 / self.theta
        )
        return sympy.Max(0, expr)

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        cond_distr_1 = (
            (1 - self.u) ** (theta - 1)
            * ((1 - self.u) ** theta + (1 - self.v) ** theta) ** ((1 - theta) / theta)
            * sympy.Heaviside(
                1 - ((1 - self.u) ** theta + (1 - self.v) ** theta) ** (1 / theta)
            )
        )
        return CD1Wrapper(cond_distr_1)(u, v)

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        cond_distr = (
            (1 - self.v) ** (theta - 1)
            * ((1 - self.u) ** theta + (1 - self.v) ** theta) ** ((1 - theta) / theta)
            * sympy.Heaviside(
                1 - ((1 - self.u) ** theta + (1 - self.v) ** theta) ** (1 / theta)
            )
        )
        return CD2Wrapper(cond_distr)(u, v)

    def _squared_cond_distr_1(self, u, v):
        theta = self.theta
        return (
            (1 - u) ** (2 * theta - 2)
            * ((1 - u) ** theta + (1 - v) ** theta) ** (-2 + 2 / theta)
            * sympy.Heaviside(1 - ((1 - u) ** theta + (1 - v) ** theta) ** (1 / theta))
        )

    def _xi_int_1(self, v):
        u = self.u
        theta = self.theta
        undefined_integral = sympy.Piecewise(
            (
                -(
                    (1 - u) ** (2 * theta - 1)
                    * (1 - v) ** (-2 * theta)
                    * ((1 - u) ** theta * (1 - v) ** (-theta) + 1) ** (-2 / theta)
                    * ((1 - u) ** theta + (1 - v) ** theta) ** (2 / theta)
                    * sympy.hyper(
                        [2 - 2 / theta, 2 - 1 / theta],
                        [3 - 1 / theta],
                        -((1 - u) ** theta) * (1 - v) ** (-theta),
                    )
                )
                / (2 * theta - 1),
                ((1 - u) ** theta + (1 - v) ** theta) ** (1 / theta) <= 1,
            ),
            (0, True),
        )

        return undefined_integral.subs(u, 1) - undefined_integral.subs(u, 0)

    #
    # def _int_2(self):
    #     v = self.v
    #     theta = self.theta
    #     return sympy.Integral(
    #         ((1 - v) ** theta + 1) ** (-1 + 2 / theta)
    #         * sympy.hyper((1, 1 + 1 / theta), (3 - 1 / theta,), -((1 - v) ** (-theta)))
    #         / (1 - v) ** theta,
    #         (v, 0, 1),
    #     ) / (2 * theta - 1)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)
