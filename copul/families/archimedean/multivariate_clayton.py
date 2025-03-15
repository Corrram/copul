import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class MultivariateClayton(ArchimedeanCopula):
    """
    Multivariate Clayton Copula implementation.

    The Clayton copula is defined by its generator:
    φ(t) = (t^(-θ) - 1) / θ for θ > 0
    φ(t) = -log(t) for θ = 0

    Parameters
    ----------
    theta : float
        The parameter controlling the strength of dependence.
        θ > 0 indicates positive dependence, θ = 0 gives independence.
    dimension : int, optional
        The dimension of the copula (default is 2).
    """

    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)
    special_cases = {0: IndependenceCopula}

    def __init__(self, dimension=2, **kwargs):
        """
        Initialize a multivariate Clayton copula.

        Parameters
        ----------
        theta : float, optional
            The parameter controlling the strength of dependence. Default is 1.0.
        dimension : int, optional
            The dimension of the copula. Default is 2.
        **kwargs
            Additional keyword arguments for the parent class.
        """
        self._dimension = dimension
        super().__init__(dimension=dimension, **kwargs)

    @property
    def dimension(self) -> int:
        """
        Return the dimension of the copula.

        Returns
        -------
        int
            The dimension
        """
        return self._dimension

    @property
    def _generator_at_0(self):
        """
        Value of the generator at t=0.

        Returns
        -------
        sympy expression
            Infinity for θ ≥ 0
        """
        return sympy.oo

    @property
    def _raw_generator(self):
        """
        Raw generator function for Clayton copula.

        Returns
        -------
        sympy expression
            The Clayton generator: (t^(-θ) - 1) / θ for θ > 0, -log(t) for θ = 0
        """
        # Regular case expression for theta != 0
        regular_expr = ((1 / self.t) ** self.theta - 1) / self.theta
        # Logarithmic generator for theta = 0
        log_expr = -sympy.log(self.t)

        # Return appropriate expression based on theta
        return sympy.Piecewise(
            (log_expr, self.theta == 0),  # Independence case (θ = 0)
            (regular_expr, True),  # Regular case (θ > 0)
        )

    @property
    def _raw_inv_generator(self):
        """
        Raw inverse generator function for Clayton copula.

        Returns
        -------
        sympy expression
            The inverse generator: (1 + θ*y)^(-1/θ) for θ > 0, exp(-y) for θ = 0
        """
        # Handle independence case (θ = 0)
        if self.theta == 0:
            return sympy.exp(-self.y)
        # Regular case (θ > 0)
        return (self.theta * self.y + 1) ** (-1 / self.theta)

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        Returns
        -------
        bool
            True for Clayton with θ ≥ 0
        """
        return self.theta >= 0

    def cdf(self, *args):
        """
        Cumulative distribution function of the multivariate Clayton copula.

        Parameters
        ----------
        *args : float
            The uniform marginals u₁, u₂, ..., uₙ

        Returns
        -------
        SymPyFuncWrapper
            The CDF value
        """
        # Handle specific case when evaluated at specific points
        if args and all(isinstance(arg, (int, float, np.number)) for arg in args):
            # Check for boundary conditions
            if any(arg == 0 for arg in args):
                return 0.0
            if all(arg == 1 for arg in args):
                return 1.0

            # Independence case (θ = 0)
            if self.theta == 0:
                return np.prod(args)

            # Regular case (θ > 0)
            # C(u₁,...,uₙ) = (∑ᵢ uᵢ^(-θ) - n + 1)^(-1/θ)
            term_sum = sum(u ** (-self.theta) for u in args) - len(args) + 1
            if term_sum <= 0:
                return 0.0
            return term_sum ** (-1 / self.theta)

        # Symbolic case - build an expression with the right number of variables
        # First, create the symbolic variables
        u_vars = []
        for i in range(self.dimension):
            u_vars.append(sympy.symbols(f"u{i+1}", positive=True))

        # Handle independence case (θ = 0)
        if self.theta == 0:
            return SymPyFuncWrapper(sympy.prod(u_vars))

        # Regular case
        term_sum = sum(u ** (-self.theta) for u in u_vars) - self.dimension + 1
        cdf_expr = sympy.Max(0, term_sum) ** (-1 / self.theta)

        return SymPyFuncWrapper(cdf_expr)

    def cdf_vectorized(self, *args):
        """
        Vectorized implementation of the CDF for multivariate Clayton copula.

        Parameters
        ----------
        *args : array_like
            The uniform marginals as numpy arrays

        Returns
        -------
        numpy.ndarray
            The CDF values
        """
        # Convert all inputs to numpy arrays and check shapes
        arrays = [np.asarray(arg) for arg in args]

        # Check if the number of arrays matches the dimension
        if len(arrays) != self.dimension:
            raise ValueError(f"Expected {self.dimension} inputs, got {len(arrays)}")

        # Ensure all arrays have compatible shapes
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            # If shapes differ, try broadcasting
            try:
                # Get broadcast shape
                broadcast_shape = np.broadcast(*arrays).shape
                arrays = [np.broadcast_to(arr, broadcast_shape) for arr in arrays]
            except ValueError:
                raise ValueError("Input arrays have incompatible shapes")

        # Ensure inputs are within [0, 1]
        for i, arr in enumerate(arrays):
            if np.any((arr < 0) | (arr > 1)):
                raise ValueError(f"Input {i} contains values outside [0, 1]")

        # For independence case (θ = 0)
        if self.theta == 0:
            return np.prod(arrays, axis=0)

        # Initialize the result array
        result = np.zeros_like(arrays[0], dtype=float)

        # Handle boundary conditions: if any input is 0, result is 0
        non_zero_mask = np.ones_like(arrays[0], dtype=bool)
        for arr in arrays:
            non_zero_mask = non_zero_mask & (arr > 0)

        # Only compute for non-zero inputs
        if np.any(non_zero_mask):
            # Compute the sum term: ∑ᵢ uᵢ^(-θ) - n + 1
            sum_term = -self.dimension + 1
            for arr in arrays:
                sum_term = sum_term + np.power(arr[non_zero_mask], -self.theta)

            # Apply max with 0 to ensure we don't have negative values
            sum_term = np.maximum(0, sum_term)

            # Compute final result: (sum_term)^(-1/θ)
            result[non_zero_mask] = np.power(sum_term, -1 / self.theta)

        return result

    def lambda_L(self):
        """
        Lower tail dependence coefficient.

        For Clayton copula with θ > 0, this is 2^(-1/θ).

        Returns
        -------
        float or sympy expression
            The lower tail dependence coefficient
        """
        # Clayton with θ = 0 has no tail dependence
        if self.theta == 0:
            return 0
        # Formula for Clayton with θ > 0
        return 2 ** (-1 / self.theta)

    def lambda_U(self):
        """
        Upper tail dependence coefficient.

        For Clayton copula, this is always 0.

        Returns
        -------
        float
            Always 0 for Clayton
        """
        return 0
