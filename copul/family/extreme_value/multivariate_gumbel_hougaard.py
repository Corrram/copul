from typing import TypeAlias
import numpy as np
import sympy as sp

from copul.family.extreme_value.multivariate_extreme_value_copula import (
    MultivariateExtremeValueCopula,
)
from copul.family.other import BivIndependenceCopula
from copul.family.other.independence_copula import IndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper


class MultivariateGumbelHougaard(MultivariateExtremeValueCopula):
    """
    Multivariate Gumbel-Hougaard Extreme Value Copula.

    A specialized extreme value copula with the form:
    C(u₁, u₂, ..., uₙ) = exp(-((-ln u₁)^θ + (-ln u₂)^θ + ... + (-ln uₙ)^θ)^(1/θ))

    Special cases:
    - θ = 1: Independence copula
    - θ → ∞: Comonotonicity copula (perfect positive dependence)

    Parameters
    ----------
    dimension : int
        Dimension of the copula (number of variables).
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

    def __new__(cls, dimension=2, *args, **kwargs):
        """
        Custom instance creation to handle the special case of θ=1.

        When θ=1, the Gumbel-Hougaard copula reduces to the independence copula.

        Parameters
        ----------
        dimension : int, optional
            Dimension of the copula (default is 2).
        *args, **kwargs
            Additional arguments, potentially including 'theta'.

        Returns
        -------
        Copula
            Either a MultivariateGumbelHougaard or IndependenceCopula instance.
        """
        # Check if theta=1 is passed either as a positional arg or keyword arg
        theta_is_one = False
        if args and len(args) > 0 and args[0] == 1:
            theta_is_one = True
        if "theta" in kwargs and kwargs["theta"] == 1:
            theta_is_one = True

        if theta_is_one:
            # Return an IndependenceCopula instance
            new_kwargs = kwargs.copy()
            if "theta" in new_kwargs:
                del new_kwargs["theta"]
            if dimension == 2:
                return BivIndependenceCopula(dimension, **new_kwargs)
            return IndependenceCopula(dimension, **new_kwargs)

        # If theta is not 1, proceed with normal initialization
        return super().__new__(cls)

    def __init__(self, dimension=2, theta=None, *args, **kwargs):
        """
        Initialize a Multivariate Gumbel-Hougaard copula.

        Parameters
        ----------
        dimension : int, optional
            Dimension of the copula (default is 2).
        theta : float, optional
            Dependence parameter. Must be greater than or equal to 1.
        *args, **kwargs
            Additional parameters.
        """
        super().__init__(dimension=dimension, *args, **kwargs)

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
            Either a MultivariateGumbelHougaard or IndependenceCopula instance.
        """
        if args and len(args) > 0:
            kwargs["theta"] = args[0]

        if "theta" in kwargs and kwargs["theta"] == 1:
            # If theta is 1, return an IndependenceCopula
            del kwargs["theta"]
            if self.dim == 2:
                return BivIndependenceCopula(self.dim, **kwargs)
            return IndependenceCopula(self.dim, **kwargs)

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

        The Gumbel-Hougaard copula is symmetric in all its arguments.

        Returns
        -------
        bool
            True
        """
        return True

    def _compute_extreme_value_function(self, u_values):
        """
        Compute the Gumbel-Hougaard extreme value function.

        Parameters
        ----------
        u_values : list
            List of u values (marginals) for evaluation.

        Returns
        -------
        float
            The computed extreme value function value.
        """
        # Check for boundary conditions
        if any(u <= 0 for u in u_values):
            return 0
        if all(u == 1 for u in u_values):
            return 1

        # Extract theta value (handle both numeric and symbolic cases)
        theta_val = self.theta
        if not isinstance(theta_val, (int, float)) and hasattr(theta_val, "evalf"):
            theta_val = float(theta_val.evalf())

        # Compute the sum of negative log u values raised to theta
        neg_log_sum = sum((-np.log(u)) ** theta_val for u in u_values)

        # Return the extremal function value
        return np.exp(-((neg_log_sum) ** (1 / theta_val)))

    @property
    def cdf(self):
        """
        C(u1,...,ud) = exp(-((sum_j (-log uj)^theta)^(1/theta)))
        With boundary handling:
          - if any u_j == 0 -> 0
          - if all u_j == 1 -> 1
          - else -> interior expression
        """
        u_symbols = self.u_symbols

        # interior expression
        neg_log_sum = sum((-sp.log(u)) ** self.theta for u in u_symbols)
        expr = sp.exp(-(neg_log_sum ** (sp.Integer(1) / self.theta)))

        # boundaries
        any_zero = sp.Or(*[sp.Eq(u, 0) for u in u_symbols])
        all_one = sp.And(*[sp.Eq(u, 1) for u in u_symbols])

        # IMPORTANT: order matters (first match wins)
        cdf_piecewise = sp.Piecewise(
            (0, any_zero),
            (1, all_one),
            (expr, True),
        )

        return CDFWrapper(cdf_piecewise)

    def kendalls_tau(self, *args, **kwargs):
        """
        Compute Kendall's tau for the multivariate Gumbel-Hougaard copula.

        For the bivariate case, Kendall's tau is (θ-1)/θ.
        For higher dimensions, the pairwise Kendall's tau is the same for any pair.

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

    def cdf_vectorized(self, *args):
        """
        Vectorized CDF with correct boundary handling.
        """
        arrays = [np.asarray(arg) for arg in args]
        if len(arrays) != self.dim:
            raise ValueError(f"Expected {self.dim} arrays, got {len(arrays)}")

        # Broadcast shape
        shape = np.broadcast(*arrays).shape

        # Masks for exact boundaries (before any epsilon tricks)
        any_zero = np.zeros(shape, dtype=bool)
        for arr in arrays:
            any_zero |= (arr == 0)

        all_one = np.ones(shape, dtype=bool)
        for arr in arrays:
            all_one &= (arr == 1)

        # Stabilize logs only for the interior computation
        adjusted = [np.maximum(arr, 1e-300) for arr in arrays]  # tiny epsilon

        # Resolve theta to float if symbolic
        theta_val = self.theta
        if not isinstance(theta_val, (int, float)) and hasattr(theta_val, "evalf"):
            theta_val = float(theta_val.evalf())

        # Interior expression
        neg_log_sum = np.zeros(shape, dtype=float)
        for arr in adjusted:
            neg_log_sum += (-np.log(arr)) ** theta_val

        result = np.exp(-(neg_log_sum ** (1.0 / theta_val)))

        # Enforce boundaries exactly
        result[any_zero] = 0.0
        result[all_one] = 1.0

        return result


MultivariateGumbelHougaardEV: TypeAlias = MultivariateGumbelHougaard
