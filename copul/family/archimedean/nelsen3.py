from typing import TypeAlias

import numpy as np
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi
from copul.wrapper.cd1_wrapper import CD1Wrapper


class AliMikhailHaq(BivArchimedeanCopula):
    """
    Ali-Mikhail-Haq copula (Nelsen 3)
    """

    ac = BivArchimedeanCopula
    theta_interval = sympy.Interval(-1, 1, left_open=False, right_open=False)
    special_cases = {
        0: BivIndependenceCopula,
        1: PiOverSigmaMinusPi,
    }

    def __str__(self):
        return super().__str__()

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def _numeric_callables(self):
        r"""Closed forms for the numerical API.

        With :math:`D=1-\theta(1-u)(1-v)`: :math:`C=uv/D`,
        :math:`\partial_1C=v(1-\theta(1-v))/D^2` and
        :math:`c=\bigl((1-\theta+2\theta v)D-2\theta(1-u)v(1-\theta(1-v))\bigr)/D^3`;
        for :math:`0<\theta<1` exact sampling with a geometric frailty
        (Marshall--Olkin), otherwise conditional inversion.
        """
        from copul.family.archimedean import _frailty

        th = float(self.theta)

        def cdf(u, v):
            return u * v / (1.0 - th * (1.0 - u) * (1.0 - v))

        def _h(a, b):
            d = 1.0 - th * (1.0 - a) * (1.0 - b)
            return b * (1.0 - th * (1.0 - b)) / d**2

        def pdf(u, v):
            d = 1.0 - th * (1.0 - u) * (1.0 - v)
            num = (1.0 - th + 2.0 * th * v) * d - 2.0 * th * (1.0 - u) * v * (1.0 - th * (1.0 - v))
            return num / d**3

        out = {"cdf": cdf, "h1": _h, "h2": lambda u, v: _h(v, u), "pdf": pdf}
        if 0.0 < th < 1.0:

            def rvs(n, rng):
                frailty = _frailty.geometric_frailty(n, rng, 1.0 - th)
                return _frailty.marshall_olkin(
                    lambda t: (1.0 - th) / (np.exp(t) - th), frailty, rng
                )

            out["rvs"] = rvs
        return out

    @property
    def _raw_generator(self):
        return sympy.log((1 - self.theta * (1 - self.t)) / self.t)

    @property
    def _raw_inv_generator(self):
        theta = self.theta
        return (theta - 1) / (theta - sympy.exp(self.y))

    @property
    def _cdf_expr(self):
        u = self.u
        v = self.v
        cdf = (u * v) / (1 - self.theta * (1 - u) * (1 - v))
        return cdf

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        cond_distr_1 = (
            self.v
            * (theta * self.u * (self.v - 1) - theta * (self.u - 1) * (self.v - 1) + 1)
            / (theta * (self.u - 1) * (self.v - 1) - 1) ** 2
        )
        return CD1Wrapper(cond_distr_1)(u, v)

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        cond_distr_2 = (
            self.u
            * (theta * self.v * (self.u - 1) - theta * (self.v - 1) * (self.u - 1) + 1)
            / (theta * (self.u - 1) * (self.v - 1) - 1) ** 2
        )
        return CD1Wrapper(cond_distr_2)(u, v)

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        th = self.theta
        # int_1^{1-th} log(t)/(1-t) dt = Li_2(th)
        integral = sympy.polylog(2, th)
        return (12 * (1 + th) * integral - 24 * (1 - th) * sympy.log(1 - th)) / th**2 - 3 * (
            th + 12
        ) / th

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        theta = self.theta
        return 1 - 2 / (3 * theta) - 2 * (1 - theta) ** 2 / (3 * theta**2) * sympy.log(1 - theta)

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
