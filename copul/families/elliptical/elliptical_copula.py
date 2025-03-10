from abc import abstractmethod
import sympy
from copul.families.bivcopula import BivCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class EllipticalCopula(BivCopula):
    """
    Abstract base class for elliptical copulas.

    Elliptical copulas are derived from elliptical distributions and are
    characterized by a correlation parameter rho in [-1, 1].

    Special cases:
    - rho = -1: Lower Fréchet bound (countermonotonicity)
    - rho = 1: Upper Fréchet bound (comonotonicity)
    """

    t = sympy.symbols("t", positive=True)
    generator = None
    rho = sympy.symbols("rho", real=True)
    params = [rho]
    intervals = {"rho": sympy.Interval(-1, 1, left_open=False, right_open=False)}

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
        """
        Returns the 2x2 correlation matrix based on the rho parameter.

        Returns:
            sympy.Matrix: 2x2 correlation matrix with ones on the diagonal
                          and rho on the off-diagonal.
        """
        return sympy.Matrix([[1, self.rho], [self.rho, 1]])

    def characteristic_function(self, t1, t2):
        """
        Computes the characteristic function of the elliptical copula.

        Args:
            t1 (float or sympy.Symbol): First argument
            t2 (float or sympy.Symbol): Second argument

        Returns:
            sympy.Expr: Value of the characteristic function

        Raises:
            NotImplementedError: If generator is not defined in the subclass
        """
        if self.generator is None:
            raise NotImplementedError("Generator function must be defined in subclass")

        arg = (
            t1**2 * self.corr_matrix[0, 0]
            + t2**2 * self.corr_matrix[1, 1]
            + 2 * t1 * t2 * self.corr_matrix[0, 1]
        )
        return self.generator(arg)

    @property
    @abstractmethod
    def cdf(self) -> SymPyFuncWrapper:
        """
        Abstract method to compute the cumulative distribution function.

        Must be implemented by subclasses.

        Returns:
            SymPyFuncWrapper: Wrapped CDF function
        """
        pass
