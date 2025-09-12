import sympy as sp
from abc import abstractmethod

from copul.family.core.copula import Copula


class MultivariateEllipticalCopula(Copula):
    """
    Abstract base class for multivariate elliptical copulas.

    Elliptical copulas are derived from elliptical distributions and are characterized
    by a correlation structure. In the multivariate case, they are defined by a
    correlation matrix R.
    """

    t = sp.symbols("t", positive=True)
    # symbols rho_ij for off-diagonal elements of the correlation matrix
    rho = sp.symbols("rho")
    params = [rho]

    generator = None

    def __init__(self, dimension, *args, corr_matrix=None, **kwargs):
        """
        Initialize a multivariate elliptical copula.

        Parameters
        ----------
        dimension : int, optional
            Dimension of the copula. If None, it will use the dimension from kwargs.
        corr_matrix : array_like or sympy.Matrix, optional
            Correlation matrix. If None, an identity matrix is used.
        *args, **kwargs
            Additional arguments passed to the superclass constructor.
        """
        # Initialize correlation matrix attribute
        self._corr_matrix = None

        if "dimension" in kwargs:
            dimension = kwargs["dimension"]
            del kwargs["dimension"]
        Copula.__init__(self, dimension, *args, **kwargs)

        # Set up correlation matrix (identity if not provided)
        if corr_matrix is not None:
            # Convert to SymPy matrix if needed
            if not isinstance(corr_matrix, sp.Matrix):
                corr_matrix = sp.Matrix(corr_matrix)

            # Validate correlation matrix
            if corr_matrix.shape != (self.dim, self.dim):
                raise ValueError(f"Correlation matrix must be {self.dim}x{self.dim}")

            # Check if matrix is symmetric and has ones on diagonal
            if not all(corr_matrix[i, i] == 1 for i in range(self.dim)):
                raise ValueError("Correlation matrix must have ones on the diagonal")

            if not all(
                corr_matrix[i, j] == corr_matrix[j, i]
                for i in range(self.dim)
                for j in range(self.dim)
            ):
                raise ValueError("Correlation matrix must be symmetric")

            self.corr_matrix = corr_matrix

    @property
    def params_from_matrix(self):
        """
        Get the parameters of the copula from the correlation matrix.

        For a multivariate elliptical copula, these are the off-diagonal
        elements of the correlation matrix.

        Returns
        -------
        list
            List of parameters (correlation values).
        """
        params_list = []
        n = self.corr_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                params_list.append(self.corr_matrix[i, j])
        return params_list

    def characteristic_function(self, t_vector):
        """
        Compute the characteristic function of the elliptical copula.

        Parameters
        ----------
        t_vector : array_like
            Vector of arguments for the characteristic function.

        Returns
        -------
        sympy.Expr
            Value of the characteristic function.

        Raises
        ------
        NotImplementedError
            If generator is not defined in the subclass.
        """
        if self.generator is None:
            raise NotImplementedError("Generator function must be defined in subclass")

        # Convert to SymPy matrix if needed
        if not isinstance(t_vector, sp.Matrix):
            t_vector = sp.Matrix(t_vector)

        # Compute quadratic form t' * corr_matrix * t
        arg = (t_vector.T * self.corr_matrix * t_vector)[0]

        # Make a proper substitution with a dictionary
        t = self.t  # Get the symbol
        return self.generator.subs({t: arg})

    @property
    @abstractmethod
    def is_absolutely_continuous(self):
        """
        Check if the copula is absolutely continuous.

        Returns
        -------
        bool
            True if the copula is absolutely continuous, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def is_symmetric(self):
        """
        Check if the copula is symmetric.

        Returns
        -------
        bool
            True if the copula is symmetric, False otherwise.
        """
        pass

    @property
    def corr_matrix(self):
        """
        Get the correlation matrix.

        Returns
        -------
        sympy.Matrix
            Correlation matrix.
        """
        return self._corr_matrix

    @corr_matrix.setter
    def corr_matrix(self, matrix):
        """
        Set the correlation matrix.

        Parameters
        ----------
        matrix : sympy.Matrix
            Correlation matrix.
        """
        self._corr_matrix = matrix

    @property
    @abstractmethod
    def cdf(self):
        """
        Abstract method to compute the cumulative distribution function.

        Must be implemented by subclasses.

        Returns
        -------
        SymPyFuncWrapper
            Wrapped CDF function.
        """
        pass
