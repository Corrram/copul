import sympy as sp

from copul.families.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.families.extreme_value.multivariate_gumbel_hougaard import (
    MultivariateGumbelHougaard,
)
from copul.families.other import BivIndependenceCopula


class GumbelHougaardEV(MultivariateGumbelHougaard, BivExtremeValueCopula):
    """
    Bivariate Gumbel-Hougaard Extreme Value Copula.

    A specialized version of the multivariate Gumbel-Hougaard copula for the bivariate case.
    This copula combines features of both the bivariate extreme value copula and the
    multivariate Gumbel-Hougaard copula.

    The CDF is given by:
    C(u,v) = exp(-(((-ln u)^θ + (-ln v)^θ)^(1/θ)))

    Special cases:
    - θ = 1: Independence copula
    - θ → ∞: Comonotonicity copula (perfect positive dependence)

    Parameters
    ----------
    theta : float, optional
        Dependence parameter (default is None).
        Must be greater than or equal to 1.
    """

    # Define parameters
    theta = sp.symbols("theta", positive=True)
    params = [theta]
    intervals = {
        str(theta): sp.Interval(1, float("inf"), left_open=False, right_open=True)
    }

    def __new__(cls, *args, **kwargs):
        """
        Custom instance creation to handle the special case of θ=1.

        When θ=1, the Gumbel-Hougaard copula reduces to the independence copula.

        Parameters
        ----------
        *args, **kwargs
            Additional arguments, potentially including 'theta'.

        Returns
        -------
        Copula
            Either a GumbelHougaard or IndependenceCopula instance.
        """
        # Check if theta=1 is passed either as a positional arg or keyword arg
        theta_is_one = False
        if len(args) > 0 and args[0] == 1:
            theta_is_one = True
        if "theta" in kwargs and kwargs["theta"] == 1:
            theta_is_one = True

        if theta_is_one:
            # Return an IndependenceCopula instance
            new_kwargs = kwargs.copy()
            if "theta" in new_kwargs:
                del new_kwargs["theta"]
            return BivIndependenceCopula(**new_kwargs)

        # If theta is not 1, proceed with normal initialization
        return super().__new__(cls)

    def __init__(self, theta=None, *args, **kwargs):
        """
        Initialize a Bivariate Gumbel-Hougaard copula.

        Parameters
        ----------
        theta : float, optional
            Dependence parameter. Must be greater than or equal to 1.
        *args, **kwargs
            Additional parameters.
        """
        # Fix for multiple inheritance: avoid passing dimension twice
        # Initialize BivExtremeValueCopula first (it will set dimension=2)
        MultivariateGumbelHougaard.__init__(self, 2, *args, **kwargs)
        BivExtremeValueCopula.__init__(self, *args, **kwargs)

        # Set theta parameter
        if theta is not None:
            self.theta = theta
            self._free_symbols = {"theta": self.theta}

    def __call__(self, *args, **kwargs):
        """
        Return a new instance with updated parameters.

        Handle the special case where θ=1, which should return
        an independence copula.

        Parameters
        ----------
        *args
            Positional arguments, potentially including theta.
        **kwargs
            Keyword arguments, potentially including 'theta'.

        Returns
        -------
        Copula
            Either a GumbelHougaard or IndependenceCopula instance.
        """
        if args and len(args) > 0:
            kwargs["theta"] = args[0]

        if "theta" in kwargs and kwargs["theta"] == 1:
            # If theta is 1, return an IndependenceCopula
            del kwargs["theta"]
            return BivIndependenceCopula()(**kwargs)

        # Otherwise, proceed with normal call
        return super().__call__(*args, **kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        The Gumbel-Hougaard copula is absolutely continuous.

        Returns
        -------
        bool
            True
        """
        return True

    @property
    def is_symmetric(self) -> bool:
        """
        Check if the copula is symmetric.

        The Gumbel-Hougaard copula is symmetric.

        Returns
        -------
        bool
            True
        """
        return True

    @property
    def _pickands(self):
        """
        Pickands dependence function for the Gumbel-Hougaard copula.

        A(t) = (t^θ + (1-t)^θ)^(1/θ)

        Returns
        -------
        sympy.Expr
            The Pickands dependence function.
        """
        return (self.t**self.theta + (1 - self.t) ** self.theta) ** (1 / self.theta)

    @property
    def _cdf_expr(self):
        return sp.exp(
            -(
                (sp.log(1 / self.v) ** self.theta + sp.log(1 / self.u) ** self.theta)
                ** (1 / self.theta)
            )
        )

    def tau(self, *args, **kwargs):
        """
        Compute Kendall's tau for the Gumbel-Hougaard copula.

        For the Gumbel-Hougaard copula, Kendall's tau has the closed form (θ-1)/θ.

        Parameters
        ----------
        *args, **kwargs
            Parameters for the copula.

        Returns
        -------
        float or sympy.Expr
            Kendall's tau value or expression.
        """
        self._set_params(args, kwargs)
        return (self.theta - 1) / self.theta

    def _rho(self):
        """
        Compute the expression for Spearman's rho.

        Returns
        -------
        sympy.Expr
            Spearman's rho expression.
        """
        t = self.t
        theta = self.theta
        integrand = ((t**theta + (1 - t) ** theta) ** (1 / theta) + 1) ** (-2)
        return 12 * sp.Integral(integrand, (t, 0, 1)) - 3
