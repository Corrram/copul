import sympy as sp

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula


class MultivariateArchimedeanIndependence(ArchimedeanCopula, IndependenceCopula):
    """
    Multivariate Independence Copula as an Archimedean Copula.

    This class represents the independence copula as a special case of an Archimedean copula
    with generator function φ(t) = -log(t).

    The independence copula represents statistical independence between random variables:
    C(u₁, u₂, ..., uₙ) = u₁ × u₂ × ... × uₙ

    Parameters
    ----------
    dimension : int, optional
        Dimension of the copula (number of variables). Default is 2.
    """

    # Define class-level generator expressions
    _t_min = 0
    _t_max = 1
    t = sp.symbols("t", nonnegative=True)
    y = sp.symbols("y", nonnegative=True)
    _generator_at_0 = sp.oo

    def __init__(self, dimension=2, **kwargs):
        """
        Initialize a multivariate independence copula as an Archimedean copula.

        Parameters
        ----------
        dimension : int, optional
            Dimension of the copula (default is 2).
        **kwargs
            Additional keyword arguments (ignored).
        """
        # Initialize both parent classes
        IndependenceCopula.__init__(self, dimension=dimension, **kwargs)

    @property
    def _raw_generator(self):
        """
        Raw generator function for the independence copula.

        The independence copula has generator φ(t) = -log(t).

        Returns
        -------
        sympy.Expr
            The generator function expression.
        """
        return -sp.log(self.t)

    @property
    def _raw_inv_generator(self):
        """
        Raw inverse generator function for the independence copula.

        The independence copula has inverse generator ψ(s) = exp(-s).

        Returns
        -------
        sympy.Expr
            The inverse generator function expression.
        """
        return sp.exp(-self.y)

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        The independence copula is absolutely continuous.

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

        The independence copula is symmetric in all its arguments.

        Returns
        -------
        bool
            True
        """
        return True


# Register the independence copula as a special case of Archimedean copulas
# This allows Archimedean copula factories to return the independence copula
# when the parameter value corresponds to independence
def register_independence_special_case(archimedean_class, independence_param_value):
    """
    Register the MultivariateArchimedeanIndependence class as a special case
    of an Archimedean copula family.

    Parameters
    ----------
    archimedean_class : class
        The Archimedean copula class to register with.
    independence_param_value : float or int
        The parameter value that corresponds to independence.
    """
    if not hasattr(archimedean_class, "special_cases"):
        archimedean_class.special_cases = {}

    archimedean_class.special_cases[independence_param_value] = (
        MultivariateArchimedeanIndependence
    )
