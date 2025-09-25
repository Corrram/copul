import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.frechet.frechet import Frechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class BivIndependenceCopula(Frechet, BivArchimedeanCopula):
    """Bivariate Independence Copula implementation.

    The independence copula represents statistical independence between random variables:
    C(u,v) = u*v

    This is a special case of both the Frechet family (with alpha=beta=0) and
    the Archimedean family (with generator -log(t)).
    """

    _alpha = 0
    _beta = 0

    @property
    def alpha(self):
        return 0

    @property
    def beta(self):
        return 0

    @property
    def pickands(self):
        return sympy.Max(1)

    @property
    def _raw_generator(self):
        # independence copula is a special case of an Archimedean copula
        return -sympy.log(self.t)

    @property
    def _raw_inv_generator(self):
        return sympy.exp(-self.y)

    @property
    def cdf(self):
        """CDF of the independence copula: C(u,v) = u*v"""
        # Return a CDFWrapper around the expression u*v
        return CDFWrapper(self.u * self.v)

    def cond_distr_1(self, u=None, v=None):
        """Conditional distribution: C_2(v|u) = v

        For an independence copula, the conditional distribution of v given u is just v.
        """
        if v is None:
            # Return a wrapped symbolic expression
            return CD1Wrapper(self.v)
        return v

    def cond_distr_2(self, u=None, v=None):
        """Conditional distribution: C_1(u|v) = u

        For an independence copula, the conditional distribution of u given v is just u.
        """
        if u is None:
            # Return a wrapped symbolic expression
            return CD2Wrapper(self.u)
        return u

    @property
    def pdf(self):
        """PDF of the independence copula is constant 1 on the unit square."""
        # Override the Frechet implementation which doesn't provide a PDF
        return SymPyFuncWrapper(sympy.Integer(1))

    def lambda_L(self):
        """Lower tail dependence coefficient (= 0 for independence)."""
        return 0

    def lambda_U(self):
        """Upper tail dependence coefficient (= 0 for independence)."""
        return 0
