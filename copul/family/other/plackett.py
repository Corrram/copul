import numpy as np
import sympy

from copul.family.core.biv_copula import BivCopula
from copul.family.frechet.lower_frechet import LowerFrechet
from copul.family.helpers import get_simplified_solution
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


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

    def _numeric_callables(self):
        r"""Closed forms incl. the conditional quantile.

        With :math:`S=1+(\theta-1)(u+v)`, :math:`R=\sqrt{S^2-4\theta(\theta-1)uv}`:
        :math:`C=(S-R)/(2(\theta-1))`, :math:`\partial_1C=(1-(S-2\theta v)/R)/2`,
        :math:`c=\theta(1+(\theta-1)(u+v-2uv))/R^3`.  The quantile of
        :math:`V\mid U=u` at level :math:`t` is (Johnson 1987; Nelsen 2006,
        Ex. 3.38) :math:`v=(c-(1-2t)d)/(2b)` with :math:`a=t(1-t)`,
        :math:`b=\theta+a(\theta-1)^2`, :math:`c=2a(u\theta^2+1-u)+\theta(1-2a)`,
        :math:`d=\sqrt\theta\sqrt{\theta+4au(1-u)(1-\theta)^2}`.
        """
        th = float(self.theta)
        if th == 0.0:
            from copul.family.frechet.frechet import fr_mixture_callables

            return fr_mixture_callables(0.0, 1.0)  # lower Frechet bound
        if th == 1.0:
            return {
                "cdf": lambda u, v: u * v,
                "h1": lambda u, v: v * np.ones_like(u),
                "h2": lambda u, v: u * np.ones_like(v),
                "pdf": lambda u, v: np.ones(np.broadcast(u, v).shape),
                "h1_inv": lambda u, w: w * np.ones_like(u),
                "h2_inv": lambda v, w: w * np.ones_like(v),
            }

        def parts(u, v):
            s = 1.0 + (th - 1.0) * (u + v)
            r = np.sqrt(np.maximum(s * s - 4.0 * th * (th - 1.0) * u * v, 0.0))
            return s, r

        def cdf(u, v):
            s, r = parts(u, v)
            # (S - R) / (2(theta - 1)) = 2 theta u v / (S + R); use the form
            # without cancellation (S + R for S >= 0, S - R otherwise)
            with np.errstate(all="ignore"):
                return np.where(s >= 0, 2.0 * th * u * v / (s + r), (s - r) / (2.0 * (th - 1.0)))

        def _h(a, b):
            s, r = parts(a, b)
            return 0.5 * (1.0 - (s - 2.0 * th * b) / r)

        def pdf(u, v):
            _, r = parts(u, v)
            return th * (1.0 + (th - 1.0) * (u + v - 2.0 * u * v)) / r**3

        def _h_inv(x, t):
            a = t * (1.0 - t)
            b = th + a * (th - 1.0) ** 2
            c = 2.0 * a * (x * th**2 + 1.0 - x) + th * (1.0 - 2.0 * a)
            d = np.sqrt(th) * np.sqrt(th + 4.0 * a * x * (1.0 - x) * (1.0 - th) ** 2)
            return (c - (1.0 - 2.0 * t) * d) / (2.0 * b)

        return {
            "cdf": cdf,
            "h1": _h,
            "h2": lambda u, v: _h(v, u),
            "pdf": pdf,
            "logpdf": lambda u, v: np.log(pdf(u, v)),
            "h1_inv": _h_inv,
            "h2_inv": _h_inv,
        }

    @property
    def cdf(self):
        theta = self.theta
        u = self.u
        v = self.v
        cdf = (
            1
            + (theta - 1) * (u + v)
            - sympy.sqrt((1 + (theta - 1) * (u + v)) ** 2 - 4 * u * v * theta * (theta - 1))
        ) / (2 * (theta - 1))
        simplified_cdf = get_simplified_solution(cdf)
        return SymPyFuncWrapper(simplified_cdf)

    @property
    def pdf(self):
        pdf = sympy.diff(self.cdf.func, self.u, self.v)
        return SymPyFuncWrapper(get_simplified_solution(pdf))

    def spearmans_rho(self, *args, **kwargs):
        """
        Calculate Spearman's rho for the Plackett copula.

        For the Plackett copula, the formula is:
        rho = (theta + 1) / (theta - 1) - 4 * theta * log(theta) / (theta - 1)^2

        Special case: when theta = 1, rho = 0 (independence)
        """
        self._set_params(args, kwargs)
        theta = self.theta

        # Special case: independence (theta = 1)
        if theta == 1:
            return 0

        # Regular formula for theta != 1
        return (theta + 1) / (theta - 1) - 2 * theta * sympy.log(theta) / (theta - 1) ** 2

    def blests_nu(self):
        return self.spearmans_rho()

    def blomqvist(self, *args, **kwargs):
        """Nelsen Exercise 5.18"""
        return (sympy.sqrt(self.theta) - 1) / (sympy.sqrt(self.theta) + 1)

    def blomqvists_beta(self, *args, **kwargs):
        r"""Blomqvist's :math:`\beta = (\sqrt\theta-1)/(\sqrt\theta+1)`,
        see Nelsen (2006), Exercise 5.18."""
        self._set_params(args, kwargs)
        return (sympy.sqrt(self.theta) - 1) / (sympy.sqrt(self.theta) + 1)

    def schweizer_wolff_sigma(self, *args, **kwargs):
        r"""Schweizer-Wolff :math:`\sigma = |\rho_S|`; the Plackett family
        is PQD for :math:`\theta\ge1` and NQD for :math:`\theta\le1`."""
        self._set_params(args, kwargs)
        return abs(self.spearmans_rho())

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
        return SymPyFuncWrapper(cond_distr_1)(u, v)


# B2 = Plackett
