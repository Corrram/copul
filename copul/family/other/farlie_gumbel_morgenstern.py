import numpy as np
import sympy

from copul.family.core.biv_copula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class FarlieGumbelMorgenstern(BivCopula):
    """
    Farlie-Gumbel-Morgenstern (FGM) Copula.

    The FGM copula is defined as:
    C(u,v) = u*v + theta*u*v*(1-u)*(1-v)

    It has limited dependence range with Spearman's rho in [-1/3, 1/3] and
    Kendall's tau in [-2/9, 2/9].

    Parameters:
    -----------
    theta : float, -1 ≤ theta ≤ 1
        Dependence parameter that determines the strength and direction of dependence.
        theta = 0 gives the independence copula.
        theta > 0 indicates positive dependence.
        theta < 0 indicates negative dependence.
    """

    theta = sympy.symbols("theta")
    params = [theta]
    intervals = {"theta": sympy.Interval(-1, 1, left_open=False, right_open=False)}

    def __init__(self, *args, **kwargs):
        """Initialize the FGM copula with parameter validation."""
        if args and len(args) == 1:
            kwargs["theta"] = args[0]

        if "theta" in kwargs:
            # Validate theta parameter
            theta_val = kwargs["theta"]
            if theta_val < -1 or theta_val > 1:
                raise ValueError(f"Parameter theta must be between -1 and 1, got {theta_val}")

        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Handle parameter updates when calling the instance."""
        if args and len(args) == 1:
            kwargs["theta"] = args[0]

        if "theta" in kwargs:
            # Validate theta parameter
            theta_val = kwargs["theta"]
            if theta_val < -1 or theta_val > 1:
                raise ValueError(f"Parameter theta must be between -1 and 1, got {theta_val}")

        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def _numeric_callables(self):
        r"""Closed forms incl. the conditional quantile.

        :math:`\partial_1 C(u,v)=v+a\,v(1-v)` with :math:`a=\theta(1-2u)`, whose
        inverse is :math:`v=2w/\bigl(1+a+\sqrt{(1+a)^2-4aw}\bigr)`.
        """
        th = float(self.theta)

        def cdf(u, v):
            return u * v * (1.0 + th * (1.0 - u) * (1.0 - v))

        def _h(a, b):
            return b * (1.0 + th * (1.0 - b) * (1.0 - 2.0 * a))

        def pdf(u, v):
            return 1.0 + th * (1.0 - 2.0 * u) * (1.0 - 2.0 * v)

        def _h_inv(a, w):
            k = th * (1.0 - 2.0 * a)
            return 2.0 * w / (1.0 + k + np.sqrt(np.maximum((1.0 + k) ** 2 - 4.0 * k * w, 0.0)))

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
    def is_symmetric(self) -> bool:
        return True

    @property
    def cdf(self):
        """
        Cumulative distribution function of the copula.

        C(u,v) = u*v + theta*u*v*(1-u)*(1-v)
        """
        u = self.u
        v = self.v
        cdf = u * v + self.theta * u * v * (1 - u) * (1 - v)
        return SymPyFuncWrapper(cdf)

    def cond_distr_2(self, u=None, v=None):
        """
        Conditional distribution function with respect to v.

        C_{2}(u,v) = u + theta*u*(1-u)*(1-2*v)
        """
        cd2 = self.u + self.theta * self.u * (1 - self.u) * (1 - 2 * self.v)
        return SymPyFuncWrapper(cd2)(u, v)

    @property
    def pdf(self):
        """
        Probability density function of the copula.

        c(u,v) = 1 + theta*(1-2*u)*(1-2*v)
        """
        result = 1 + self.theta * (1 - 2 * self.u) * (1 - 2 * self.v)
        return SymPyFuncWrapper(result)

    def spearmans_rho(self, *args, **kwargs):
        """
        Calculate Spearman's rho for the FGM copula.

        For FGM, rho = theta/3
        """
        self._set_params(args, kwargs)
        return self.theta / 3

    def chatterjees_xi(self, *args, condition_on_y=False, **kwargs):
        r"""Chatterjee's :math:`\xi = \theta^2/15` (both conditioning directions).

        With :math:`\partial_1 C = v + \theta v(1-v)(1-2u)`,
        :math:`\int\!\!\int(\partial_1C)^2 = \tfrac13 + \theta^2/90`.
        """
        self._set_params(args, kwargs)
        return self.theta**2 / 15

    def blomqvists_beta(self, *args, **kwargs):
        r"""Blomqvist's :math:`\beta = \theta/4`."""
        self._set_params(args, kwargs)
        return self.theta / 4

    def kendalls_tau(self, *args, **kwargs):
        """
        Calculate Kendall's tau for the FGM copula.

        For FGM, tau = 2*theta/9
        """
        self._set_params(args, kwargs)
        return 2 * self.theta / 9

    def spearmans_footrule(self, *args, **kwargs):
        r"""Spearman's footrule :math:`\psi = \theta/5`."""
        self._set_params(args, kwargs)
        return self.theta / 5

    def ginis_gamma(self, *args, **kwargs):
        r"""Gini's :math:`\gamma = 4\theta/15`."""
        self._set_params(args, kwargs)
        return 4 * self.theta / 15

    def blests_nu(self):
        return self.spearmans_rho()

    # ------------------------------------------------------------------
    # Dependence measures with known closed forms
    # ------------------------------------------------------------------

    def schweizer_wolff_sigma(self, *args, **kwargs):
        r"""
        Schweizer–Wolff :math:`\sigma` for the FGM copula.

        Since :math:`C - \Pi = \theta\,u\,v\,(1-u)(1-v)` has constant sign
        (PQD when :math:`\theta>0`, NQD when :math:`\theta<0`):

        .. math::

           \sigma = 12\,|\theta|\,
             \left[\int_0^1 t(1-t)\,dt\right]^2
             = \frac{|\theta|}{3}
        """
        self._set_params(args, kwargs)
        return abs(self.theta) / 3

    def hoeffdings_d(self, *args, **kwargs):
        r"""
        Hoeffding's :math:`D` for the FGM copula.

        .. math::

           D = 90\,\theta^2
             \left[\int_0^1 t^2(1-t)^2\,dt\right]^2
             = \frac{\theta^2}{10}
        """
        self._set_params(args, kwargs)
        return self.theta**2 / 10

    def lp_distance(self, p: float = 2, *args, **kwargs):
        r"""
        :math:`L_p` distance to independence for the FGM copula.

        .. math::

           \delta_p = k(p)\,|\theta|^p
             \left[\operatorname{B}(p+1,\,p+1)\right]^2,
           \qquad k(p) = \frac{p+1}{2\,\operatorname{B}(p+1,p+2)},

        where :math:`\operatorname{B}` is the beta function.
        """
        self._set_params(args, kwargs)
        from copul.measures.numeric import lp_constant

        if float(p).is_integer():
            from math import factorial

            q = int(p)
            beta_val = sympy.Rational(factorial(q) ** 2, factorial(2 * q + 1))
            k = sympy.Integer(int(lp_constant(q)))
        else:
            from scipy.special import beta as _beta

            beta_val = float(_beta(p + 1, p + 1))
            k = lp_constant(p)
        return k * sympy.Abs(self.theta) ** p * beta_val**2

    def blum_kiefer_rosenblatt(self, *args, **kwargs):
        r"""
        Blum-Kiefer-Rosenblatt coefficient for the FGM copula:

        .. math::

           B = 30\iint(C-\Pi)^2\,\mathrm{d}C = \frac{\theta^2}{30}\,,

        since the odd term of the density integrates to zero by symmetry.
        """
        self._set_params(args, kwargs)
        return self.theta**2 / 30

    def mutual_information(self, *args, **kwargs):
        r"""
        Mutual information for the FGM copula (numerical).

        The density is :math:`c(u,v) = 1 + \theta(1-2u)(1-2v)`. No known
        simple closed form for :math:`\int c \ln c`; delegates to the
        base-class numerical quadrature.
        """
        self._set_params(args, kwargs)
        return self._mutual_information_numerical()


if __name__ == "__main__":
    # Example usage
    fgm_copula = FarlieGumbelMorgenstern(theta=0.7)
    footrule = fgm_copula.spearmans_footrule()
    ccop = fgm_copula.to_checkerboard()
    footrule_ccop = ccop.spearmans_footrule()
    print(
        f"Footrule for FGM copula: {footrule:.3f}, Footrule for checkerboard: {footrule_ccop:.3f}"
    )
    gama = fgm_copula.ginis_gamma()
    ccop_gama = ccop.ginis_gamma()
    print(
        f"Gini's gamma for FGM copula: {gama:.3f}, Gini's gamma for checkerboard: {ccop_gama:.3f}"
    )
