from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class PiOverSigmaMinusPi(ArchimedeanCopula):
    """
    Implementation of the Pi/(Sigma-Pi) copula.

    This copula corresponds to the fixed parameter value theta=1 for the Clayton copula,
    but is implemented independently from the Clayton class.

    Properties:
    - Generator: (1/t - 1)
    - Inverse Generator: 1/(1+y)
    - CDF: C(u,v) = uv/(u+v-uv)
    - Upper tail dependence coefficient: 0
    - Lower tail dependence coefficient: 0.5
    """

    # The class is initialized with a fixed theta=1
    theta = 1

    @property
    def _generator(self):
        """Generate function: phi(t) = (1/t - 1)"""
        return (1 / self.t) - 1

    @property
    def inv_generator(self):
        """Inverse generator function: psi(y) = 1/(1+y)"""
        return SymPyFuncWrapper(1 / (1 + self.y))

    @property
    def cdf(self):
        """
        Cumulative distribution function of the copula
        C(u,v) = uv/(u+v-uv)
        """
        u = self.u
        v = self.v

        cdf = (u * v) / (u + v - u * v)
        return CDFWrapper(cdf)

    def cond_distr_1(self, u=None, v=None):
        """
        First conditional distribution: ∂C(u,v)/∂u
        """
        # Formula: v / (u + v - u*v)²
        cond_distr = self.v / (self.u + self.v - self.u * self.v) ** 2
        wrapped_cd1 = CD1Wrapper(cond_distr)
        return wrapped_cd1(u, v)

    def cond_distr_2(self, u=None, v=None):
        """
        Second conditional distribution: ∂C(u,v)/∂v
        """
        # Formula: u / (u + v - u*v)²
        cond_distr = self.u / (self.u + self.v - self.u * self.v) ** 2
        return CD2Wrapper(cond_distr)(u, v)

    @property
    def pdf(self):
        """
        Probability density function of the copula
        c(u,v) = 2(u+v-uv) / (u+v-uv)³
        """
        u = self.u
        v = self.v
        denominator = (u + v - u * v) ** 3
        numerator = 2 * (u + v - u * v)
        return SymPyFuncWrapper(numerator / denominator)

    @property
    def is_absolutely_continuous(self) -> bool:
        """This copula is absolutely continuous."""
        return True

    def lambda_L(self):
        """Lower tail dependence coefficient: λ_L = 0.5"""
        return 0.5

    def lambda_U(self):
        """Upper tail dependence coefficient: λ_U = 0"""
        return 0

    def kendalls_tau(self, *args, **kwargs):
        """Kendall's tau for this copula is 1/3"""
        return 1 / 3
