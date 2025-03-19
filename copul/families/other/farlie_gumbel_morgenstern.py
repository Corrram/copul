import sympy

from copul.families.core.biv_copula import BivCopula
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

    def rho(self, *args, **kwargs):
        """
        Calculate Spearman's rho for the FGM copula.

        For FGM, rho = theta/3
        """
        self._set_params(args, kwargs)
        return self.theta / 3

    def tau(self, *args, **kwargs):
        """
        Calculate Kendall's tau for the FGM copula.

        For FGM, tau = 2*theta/9
        """
        self._set_params(args, kwargs)
        return 2 * self.theta / 9
