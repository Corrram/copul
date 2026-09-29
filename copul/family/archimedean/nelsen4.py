import numpy as np
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula


class GumbelHougaard(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)
    special_cases = {1: BivIndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return (-sympy.log(self.t)) ** self.theta

    @property
    def _raw_inv_generator(self):
        return sympy.exp(-(self.y ** (1 / self.theta)))

    @property
    def _cdf_expr(self):
        return sympy.exp(
            -(
                ((-sympy.log(self.u)) ** self.theta + (-sympy.log(self.v)) ** self.theta)
                ** (1 / self.theta)
            )
        )

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)

    def kendalls_tau(self, *args, **kwargs):
        r"""Kendall's :math:`\tau = 1 - 1/\theta` of the Gumbel-Hougaard copula."""
        self._set_params(args, kwargs)
        return 1 - 1 / self.theta

    def spearmans_footrule(self, *args, **kwargs):
        """
        Compute Spearman's footrule (ψ) for the Gumbel–Hougaard copula.

        Closed-form expression:
            ψ(C_θ) = 6 / (2^(1/θ) + 1) - 2

        For θ = 1 (independence), this yields ψ = 0.
        As θ → ∞ (comonotonicity), this yields ψ = 1.

        Returns
        -------
        float
            Spearman's footrule value (ψ).
        """
        self._set_params(args, kwargs)
        theta = float(self.theta)
        return 6.0 / (2.0 ** (1.0 / theta) + 1.0) - 2.0

    def blomqvists_beta(self, *args, **kwargs):
        r"""
        Blomqvist's :math:`\beta` for the Gumbel-Hougaard copula.

        .. math::

           \beta = 4\,(2^{-1/\theta} + 2^{-1/\theta} - 1)^{1/\theta}\cdot ... - 1

        The diagonal section :math:`C(t,t) = t^{2-2^{1-1/\theta}}` is NOT
        exact for GH (it uses the Pickands form).  Evaluate directly:

        .. math::

           C(\tfrac12,\tfrac12) = \exp\!\bigl(-[(-\ln\tfrac12)^\theta
                                  + (-\ln\tfrac12)^\theta]^{1/\theta}\bigr)
                                = \exp\bigl(-2^{1/\theta}\ln 2\bigr)
                                = 2^{-2^{1/\theta}}
        """
        self._set_params(args, kwargs)
        theta = float(self.theta)
        return 4.0 * 2.0 ** (-(2.0 ** (1.0 / theta))) - 1.0

    def schweizer_wolff_sigma(self, *args, **kwargs):
        r"""
        Schweizer–Wolff :math:`\sigma` for the Gumbel-Hougaard copula.

        GH is PQD for :math:`\theta \ge 1`, so :math:`\sigma = \rho_S`.
        """
        self._set_params(args, kwargs)
        return abs(self.spearmans_rho())

    def _numeric_callables(self):
        r"""Closed forms for the numerical API.

        With :math:`x=-\log u`, :math:`y=-\log v`,
        :math:`A=(x^\theta+y^\theta)^{1/\theta}` (log-sum-exp):
        :math:`C=e^{-A}`,
        :math:`\partial_1C=C\,A^{1-\theta}x^{\theta-1}/u`,
        :math:`\log c=-A+x+y+(\theta-1)\log(xy)+(1-2\theta)\log A
        +\log(A+\theta-1)`; exact sampling with a positive
        :math:`1/\theta`-stable frailty (Marshall--Olkin, Kanter's method).
        """
        from copul.family.archimedean import _frailty

        th = float(self.theta)
        alpha = 1.0 / th

        def parts(u, v):
            x, y = -np.log(u), -np.log(v)
            lx, ly = np.log(x), np.log(y)
            log_a = np.logaddexp(th * lx, th * ly) / th
            return x, y, lx, ly, log_a

        def cdf(u, v):
            *_, log_a = parts(u, v)
            return np.exp(-np.exp(log_a))

        def _h(a, b):
            x, _, lx, _, log_a = parts(a, b)
            return np.exp(-np.exp(log_a) + (1.0 - th) * log_a + (th - 1.0) * lx + x)

        def logpdf(u, v):
            x, y, lx, ly, log_a = parts(u, v)
            big_a = np.exp(log_a)
            return (
                -big_a
                + x
                + y
                + (th - 1.0) * (lx + ly)
                + (1.0 - 2.0 * th) * log_a
                + np.log(big_a + th - 1.0)
            )

        def pdf(u, v):
            return np.exp(logpdf(u, v))

        def rvs(n, rng):
            frailty = _frailty.positive_stable(n, rng, alpha)
            return _frailty.marshall_olkin(lambda t: np.exp(-(t**alpha)), frailty, rng)

        return {
            "cdf": cdf,
            "h1": _h,
            "h2": lambda u, v: _h(v, u),
            "pdf": pdf,
            "logpdf": logpdf,
            "rvs": rvs,
        }


Nelsen4 = GumbelHougaard

# B6 = GumbelHougaard

if __name__ == "__main__":
    # Example usage
    copula = GumbelHougaard(theta=2)
    footrule = copula.spearmans_footrule()
    ccop = copula.to_checkerboard()
    ccop_footrule = ccop.spearmans_footrule()
    ccop_xi = ccop.chatterjees_xi()
    ccop_rho = ccop.spearmans_rho()
    print(
        f"Footrule distance: {footrule}, Checkerboard footrule: {ccop_footrule}",
        f"Checkerboard xi: {ccop_xi}",
        f"Checkerboard rho: {ccop_rho}",
    )
