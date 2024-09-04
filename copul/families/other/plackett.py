import copy

import sympy

from copul.families import get_simplified_solution
from copul.families.bivcopula import BivCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class Plackett(BivCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    theta = sympy.symbols("theta", positive=True)
    params = [theta]
    intervals = {"theta": sympy.Interval(0, sympy.oo, left_open=False, right_open=True)}

    def __call__(self, **kwargs):
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return LowerFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def cdf(self):
        theta = self.theta
        u = self.u
        v = self.v
        cdf = (
            1
            + (theta - 1) * (u + v)
            - sympy.sqrt(
                (1 + (theta - 1) * (u + v)) ** 2 - 4 * u * v * theta * (theta - 1)
            )
        ) / (2 * (theta - 1))
        simplified_cdf = get_simplified_solution(cdf)
        return SymPyFunctionWrapper(simplified_cdf)

    @property
    def pdf(self):
        pdf = sympy.diff(self.cdf.func, self.u, self.v)
        return SymPyFunctionWrapper(get_simplified_solution(pdf))

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return (self.theta + 1) / (self.theta - 1) - 4 * self.theta * sympy.log(
            self.theta
        ) / (self.theta - 1) ** 2

    def get_density_of_density(self):
        # D_vu(pdf)
        u = self.u
        theta = self.theta
        v = self.v
        return (
            -(
                (2 * u * theta - 2 * u - theta + 1)
                * (
                    u**2 * theta**2
                    - 2 * u**2 * theta
                    + u**2
                    - 2 * u * v * theta**2
                    + 2 * u * v
                    + 2 * u * theta
                    - 2 * u
                    + v**2 * theta**2
                    - 2 * v**2 * theta
                    + v**2
                    + 2 * v * theta
                    - 2 * v
                    + 1
                )
                + 3
                * (-u * theta**2 + u + v * theta**2 - 2 * v * theta + v + theta - 1)
                * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
            )
            * (2 * v * theta - 2 * v - theta + 1)
            * (
                u**2 * theta**2
                - 2 * u**2 * theta
                + u**2
                - 2 * u * v * theta**2
                + 2 * u * v
                + 2 * u * theta
                - 2 * u
                + v**2 * theta**2
                - 2 * v**2 * theta
                + v**2
                + 2 * v * theta
                - 2 * v
                + 1
            )
            + 2
            * (
                (2 * u * theta - 2 * u - theta + 1)
                * (
                    u**2 * theta**2
                    - 2 * u**2 * theta
                    + u**2
                    - 2 * u * v * theta**2
                    + 2 * u * v
                    + 2 * u * theta
                    - 2 * u
                    + v**2 * theta**2
                    - 2 * v**2 * theta
                    + v**2
                    + 2 * v * theta
                    - 2 * v
                    + 1
                )
                + 3
                * (-u * theta**2 + u + v * theta**2 - 2 * v * theta + v + theta - 1)
                * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
            )
            * (u * theta**2 - 2 * u * theta + u - v * theta**2 + v + theta - 1)
            * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
            + (
                -2
                * (theta - 1)
                * (
                    u**2 * theta**2
                    - 2 * u**2 * theta
                    + u**2
                    - 2 * u * v * theta**2
                    + 2 * u * v
                    + 2 * u * theta
                    - 2 * u
                    + v**2 * theta**2
                    - 2 * v**2 * theta
                    + v**2
                    + 2 * v * theta
                    - 2 * v
                    + 1
                )
                + 3
                * (theta**2 - 1)
                * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
                - 2
                * (2 * u * theta - 2 * u - theta + 1)
                * (u * theta**2 - 2 * u * theta + u - v * theta**2 + v + theta - 1)
                + 3
                * (2 * v * theta - 2 * v - theta + 1)
                * (-u * theta**2 + u + v * theta**2 - 2 * v * theta + v + theta - 1)
            )
            * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
            * (
                u**2 * theta**2
                - 2 * u**2 * theta
                + u**2
                - 2 * u * v * theta**2
                + 2 * u * v
                + 2 * u * theta
                - 2 * u
                + v**2 * theta**2
                - 2 * v**2 * theta
                + v**2
                + 2 * v * theta
                - 2 * v
                + 1
            )
        ) / (
            (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1) ** 2
            * (
                u**2 * theta**2
                - 2 * u**2 * theta
                + u**2
                - 2 * u * v * theta**2
                + 2 * u * v
                + 2 * u * theta
                - 2 * u
                + v**2 * theta**2
                - 2 * v**2 * theta
                + v**2
                + 2 * v * theta
                - 2 * v
                + 1
            )
            ** 2
        )

    def get_numerator_double_density(self):
        v = self.v
        u = self.u
        theta = self.theta
        return (
            -(
                (2 * u * theta - 2 * u - theta + 1)
                * (
                    u**2 * theta**2
                    - 2 * u**2 * theta
                    + u**2
                    - 2 * u * v * theta**2
                    + 2 * u * v
                    + 2 * u * theta
                    - 2 * u
                    + v**2 * theta**2
                    - 2 * v**2 * theta
                    + v**2
                    + 2 * v * theta
                    - 2 * v
                    + 1
                )
                + 3
                * (-u * theta**2 + u + v * theta**2 - 2 * v * theta + v + theta - 1)
                * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
            )
            * (2 * v * theta - 2 * v - theta + 1)
            * (
                u**2 * theta**2
                - 2 * u**2 * theta
                + u**2
                - 2 * u * v * theta**2
                + 2 * u * v
                + 2 * u * theta
                - 2 * u
                + v**2 * theta**2
                - 2 * v**2 * theta
                + v**2
                + 2 * v * theta
                - 2 * v
                + 1
            )
            + 2
            * (
                (2 * u * theta - 2 * u - theta + 1)
                * (
                    u**2 * theta**2
                    - 2 * u**2 * theta
                    + u**2
                    - 2 * u * v * theta**2
                    + 2 * u * v
                    + 2 * u * theta
                    - 2 * u
                    + v**2 * theta**2
                    - 2 * v**2 * theta
                    + v**2
                    + 2 * v * theta
                    - 2 * v
                    + 1
                )
                + 3
                * (-u * theta**2 + u + v * theta**2 - 2 * v * theta + v + theta - 1)
                * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
            )
            * (u * theta**2 - 2 * u * theta + u - v * theta**2 + v + theta - 1)
            * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
            + (
                -2
                * (theta - 1)
                * (
                    u**2 * theta**2
                    - 2 * u**2 * theta
                    + u**2
                    - 2 * u * v * theta**2
                    + 2 * u * v
                    + 2 * u * theta
                    - 2 * u
                    + v**2 * theta**2
                    - 2 * v**2 * theta
                    + v**2
                    + 2 * v * theta
                    - 2 * v
                    + 1
                )
                + 3
                * (theta**2 - 1)
                * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
                - 2
                * (2 * u * theta - 2 * u - theta + 1)
                * (u * theta**2 - 2 * u * theta + u - v * theta**2 + v + theta - 1)
                + 3
                * (2 * v * theta - 2 * v - theta + 1)
                * (-u * theta**2 + u + v * theta**2 - 2 * v * theta + v + theta - 1)
            )
            * (-2 * u * v * theta + 2 * u * v + u * theta - u + v * theta - v + 1)
            * (
                u**2 * theta**2
                - 2 * u**2 * theta
                + u**2
                - 2 * u * v * theta**2
                + 2 * u * v
                + 2 * u * theta
                - 2 * u
                + v**2 * theta**2
                - 2 * v**2 * theta
                + v**2
                + 2 * v * theta
                - 2 * v
                + 1
            )
        )

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        cond_distr_1 = (
            theta
            - (
                -2 * theta * self.v * (theta - 1)
                + (2 * theta - 2) * ((theta - 1) * (self.u + self.v) + 1) / 2
            )
            / sympy.sqrt(
                -4 * theta * self.u * self.v * (theta - 1)
                + ((theta - 1) * (self.u + self.v) + 1) ** 2
            )
            - 1
        ) / (2 * (theta - 1))
        return SymPyFunctionWrapper(cond_distr_1)(u, v)


# B2 = Plackett
