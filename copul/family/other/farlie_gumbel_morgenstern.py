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
                raise ValueError(
                    f"Parameter theta must be between -1 and 1, got {theta_val}"
                )

        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """Handle parameter updates when calling the instance."""
        if args and len(args) == 1:
            kwargs["theta"] = args[0]

        if "theta" in kwargs:
            # Validate theta parameter
            theta_val = kwargs["theta"]
            if theta_val < -1 or theta_val > 1:
                raise ValueError(
                    f"Parameter theta must be between -1 and 1, got {theta_val}"
                )

        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

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

    def kendalls_tau(self, *args, **kwargs):
        """
        Calculate Kendall's tau for the FGM copula.

        For FGM, tau = 2*theta/9
        """
        self._set_params(args, kwargs)
        return 2 * self.theta / 9

    def spearmans_footrule(self):
        return self.theta / 5

    def blests_nu(self):
        return self.spearmans_rho()

    def ginis_gamma(self):
        """
        Calculate Gini's gamma for the FGM copula.

        For FGM, Gini's gamma = 4*theta/15
        """
        return 4 * self.theta / 15

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
        return self.theta ** 2 / 10

    def lp_concordance(self, p: int = 2, *args, **kwargs):
        r"""
        :math:`L_p` concordance distance for the FGM copula.

        .. math::

           \delta_p = k(p)\,|\theta|^p
             \left[\operatorname{B}(p+1,\,p+1)\right]^2

        where :math:`\operatorname{B}` is the beta function.
        """
        self._set_params(args, kwargs)
        from math import factorial

        k_table = {1: 12, 2: 90, 3: 560, 4: 3150, 5: 16632}
        k = k_table.get(p)
        if k is None:
            raise ValueError(f"k({p}) not tabulated; supported p: {sorted(k_table)}")
        # B(p+1, p+1) = (p!)^2 / (2p+1)!
        beta_val = factorial(p) ** 2 / factorial(2 * p + 1)
        return k * abs(self.theta) ** p * beta_val ** 2

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
