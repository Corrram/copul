"""
Copula sampling module for generating random variates from copula distributions.

This module provides functionality to sample from arbitrary copula distributions
using conditional distribution methods.
"""

import inspect
import logging
import random
import warnings
from typing import Any, Callable, Tuple, Optional

import numpy as np
import scipy.optimize as opt
import sympy
from copul.checkerboard.checkerboarder import Checkerboarder
from copul.checkerboard.check_pi import CheckPi

# Set up module logger
log = logging.getLogger(__name__)


class CopulaSampler:
    """
    Sampler for generating random variates from copula distributions.

    This class provides methods to sample from arbitrary copula distributions using
    conditional distribution methods and numerical root-finding.

    Attributes:
        err_counter (int): Class variable tracking the number of sampling errors across all instances.
    """

    # Class-level error counter
    err_counter = 0

    def __init__(
        self, copul: Any, precision: int = 3, random_state: Optional[int] = None
    ):
        """
        Initialize a CopulaSampler instance.

        Args:
            copul: The copula object to sample from.
            precision: Precision level for numerical methods (default: 3).
            random_state: Random seed for reproducibility (default: None).
        """
        self._copul = copul
        self._precision = precision
        self._random_state = random_state

    def rvs(self, n: int = 1, approximate=False) -> np.ndarray:
        """
        Sample n random variates from the copula.

        Args:
            n: Number of samples to generate (default: 1).
            approximate: Use approximate sampling method (default: True).

        Returns:
            np.ndarray: Array of shape (n, 2) containing the sampled (u, v) pairs.
        """
        if approximate:
            grid_partitions = np.ceil(np.sqrt(n)).astype(int)
            checkerboarder = Checkerboarder(grid_partitions, dim=2)
            ccop = checkerboarder.compute_check_pi(self._copul)
            return ccop.rvs(n)
        # Set random seed if specified
        if self._random_state is not None:
            random.seed(self._random_state)

        # Get the conditional distribution function
        cond_distr = self._copul.cond_distr_2

        # Determine if conditional distribution has parameters in common with copula intervals
        sig = inspect.signature(cond_distr)
        params = set(sig.parameters.keys()) & set(self._copul.intervals)

        # Use direct function or create a lambda function based on symbolic expression
        if params or isinstance(self._copul, CheckPi):
            # Use the conditional distribution directly
            sampling_func = cond_distr
        else:
            # Get symbolic expression and convert to a callable function
            try:
                func_expr = cond_distr().func
                sampling_func = sympy.lambdify(
                    self._copul.u_symbols, func_expr, ["numpy"]
                )
            except Exception as e:
                log.error(f"Error creating lambda function: {e}")
                raise ValueError(f"Could not create sampling function: {e}")

        # Generate samples
        results = self._sample_val(sampling_func, n)
        return results

    def _sample_val(self, function: Callable, n: int = 1) -> np.ndarray:
        """
        Generate multiple samples using the provided function.

        Args:
            function: The sampling function.
            n: Number of samples to generate.

        Returns:
            np.ndarray: Array of shape (n, 2) containing the sampled (u, v) pairs.
        """
        # Generate n independent samples
        samples = [self.sample_val(function) for _ in range(n)]

        # Convert to numpy array
        result = np.array(samples)

        # Log error counter if debugging is enabled
        log.debug(f"Error counter: {self.err_counter}")

        return result

    def sample_val(self, function: Callable) -> Tuple[float, float]:
        """
        Generate a single sample from the copula using inverse CDF method.

        Args:
            function: The conditional distribution function.

        Returns:
            Tuple[float, float]: A (u, v) pair from the copula distribution.
        """
        # Generate uniform random variables
        v = random.uniform(0, 1)
        t = random.uniform(0, 1)

        # Create function to find root: F(u|v) = t
        def func2(u: float) -> float:
            try:
                return function(u, v) - t
            except Exception as e:
                # Handle potential errors in the function evaluation
                log.debug(f"Function evaluation error at u={u}, v={v}: {e}")
                # Return a large value to guide search away from problematic areas
                return 1000.0 * (0.5 - u)

        # Suppress numerical warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Try to find the root using numerical methods
            try:
                # Use robust root-finding method
                result = opt.root_scalar(
                    func2,
                    x0=0.5,
                    bracket=[0.000000001, 0.999999999],
                    method="brentq",  # Use Brent's method for reliable root finding
                )
            except (ZeroDivisionError, ValueError, TypeError) as e:
                # Log the error and fall back to visual solution
                log.debug(f"{self._copul.__class__.__name__}; {type(e).__name__}: {e}")
                self.err_counter += 1
                return self._get_visual_solution(func2), v

            # Check if root finding converged
            if not result.converged:
                if not result.iterations:
                    log.warning(f"{self._copul.__class__.__name__}; {result}")
                self.err_counter += 1
                return self._get_visual_solution(func2), v

        # Return the found root and the randomly generated v
        return result.root, v

    def _get_visual_solution(self, func: Callable) -> float:
        """
        Find an approximate root by grid search when numerical methods fail.

        Args:
            func: The function to find the root for.

        Returns:
            float: Approximate root where the function value is minimized.
        """
        # Define search grid based on precision
        start = 10 ** (-self._precision)
        end = 1 - 10 ** (-self._precision)
        x = np.linspace(start, end, 10**self._precision)

        # Evaluate function on grid points
        y = np.zeros_like(x)
        for i, x_i in enumerate(x):
            try:
                y[i] = func(x_i)
            except Exception:
                # Assign large value on error
                y[i] = 1e10

        # Find point where function is closest to zero
        min_idx = np.abs(y).argmin()
        return x[min_idx]
