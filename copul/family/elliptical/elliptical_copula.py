import sympy as sp

from copul.family.elliptical.multivar_elliptical_copula import (
    MultivariateEllipticalCopula,
)
from copul.family.core.biv_core_copula import BivCoreCopula
from copul.family.other.lower_frechet import LowerFrechet
from copul.family.other.upper_frechet import UpperFrechet


class EllipticalCopula(MultivariateEllipticalCopula, BivCoreCopula):
    t = sp.symbols("t", positive=True)
    generator = None
    rho = sp.symbols("rho", real=True)
    params = [rho]
    intervals = {"rho": sp.Interval(-1, 1, left_open=False, right_open=False)}

    def __init__(self, *args, **kwargs):
        # Set dimension to 2 since this is a bivariate copula
        if "dimension" in kwargs and kwargs["dimension"] != 2:
            raise ValueError("EllipticalCopula is a bivariate copula with dimension=2")

        kwargs["dimension"] = 2

        # Preserve symbolic parameters if numeric values aren't provided
        if "rho" not in kwargs and len(args) == 0:
            self.rho = self.__class__.rho  # Use the class's symbolic rho
            self.params = list(self.__class__.params)  # Copy the class's params list
            self.intervals = dict(
                self.__class__.intervals
            )  # Copy the class's intervals dict

        # If rho is provided, construct a 2x2 correlation matrix
        if "rho" in kwargs:
            rho_val = kwargs["rho"]
            corr_matrix = sp.Matrix([[1, rho_val], [rho_val, 1]])
            kwargs["corr_matrix"] = corr_matrix
            self.rho = rho_val  # Set the instance's rho

            # Update params and intervals for numeric rho
            if hasattr(self, "params"):
                self.params = [p for p in self.params if str(p) != "rho"]
            if hasattr(self, "intervals") and "rho" in self.intervals:
                self.intervals = {k: v for k, v in self.intervals.items() if k != "rho"}

        # Initialize from MultivariateEllipticalCopula
        # Since we've already set dimension in kwargs, don't pass it again
        if "dimension" in kwargs:
            dimension = kwargs["dimension"]
            del kwargs["dimension"]
        MultivariateEllipticalCopula.__init__(self, dimension, *args, **kwargs)
        BivCoreCopula.__init__(self)

    def __call__(self, **kwargs):

        if "rho" in kwargs:
            if kwargs["rho"] == -1:
                del kwargs["rho"]
                return LowerFrechet()(**kwargs)
            elif kwargs["rho"] == 1:
                del kwargs["rho"]
                return UpperFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def corr_matrix(self):
        r"""Return the 2×2 correlation matrix defined by :math:`\rho`.

        Returns
        -------
        sympy.Matrix
            Matrix

            .. math::
               \begin{bmatrix}
               1 & \rho \\
               \rho & 1
               \end{bmatrix}
        """

        return sp.Matrix([[1, self.rho], [self.rho, 1]])

    @corr_matrix.setter
    def corr_matrix(self, matrix):
        r"""Set the correlation matrix and update :math:`\rho`.

        Parameters
        ----------
        matrix : sympy.Matrix
            A 2×2 correlation matrix. Only the off-diagonal element is
            used to update :math:`\rho`.
        """

        # Only extract rho from the matrix if it's a 2x2 matrix
        if isinstance(matrix, sp.Matrix) and matrix.shape == (2, 2):
            self.rho = matrix[0, 1]

    def characteristic_function(self, t1, t2):
        r"""Characteristic function of the elliptical copula.

        Parameters
        ----------
        t1 : float or sympy.Symbol
            First argument.
        t2 : float or sympy.Symbol
            Second argument.

        Returns
        -------
        sympy.Expr
            Value of the characteristic function.

        Raises
        ------
        NotImplementedError
            If the subclass does not define a generator function.
        """

        if self.generator is None:
            raise NotImplementedError("Generator function must be defined in subclass")

        arg = (
            t1**2 * self.corr_matrix[0, 0]
            + t2**2 * self.corr_matrix[1, 1]
            + 2 * t1 * t2 * self.corr_matrix[0, 1]
        )
        # Make a proper substitution with a dictionary
        t = self.t  # Get the symbol
        return self.generator.subs({t: arg})
