import numpy as np
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.other.biv_independence_copula import BivIndependenceCopula


class GumbelHougaard(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)
    special_cases = {1: BivIndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return (-sympy.log(self.t)) ** self.theta

    @property
    def _raw_inv_generator(self):
        return sympy.exp(-(self.y ** (1 / self.theta)))

    @property
    def _cdf_expr(self):
        return sympy.exp(
            -(
                (
                    (-sympy.log(self.u)) ** self.theta
                    + (-sympy.log(self.v)) ** self.theta
                )
                ** (1 / self.theta)
            )
        )

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)

    def spearmans_footrule(self, *args, **kwargs):
        """
        Compute Spearman's footrule (ψ) for the Gumbel–Hougaard copula.

        Closed-form expression:
            ψ(C_θ) = 6 / (2^(1/θ) + 1) - 2

        For θ = 1 (independence), this yields ψ = 0.
        As θ → ∞ (comonotonicity), this yields ψ = 1.

        Returns
        -------
        float
            Spearman's footrule value (ψ).
        """
        self._set_params(args, kwargs)
        theta = float(self.theta)
        return 6.0 / (2.0 ** (1.0 / theta) + 1.0) - 2.0


Nelsen4 = GumbelHougaard

# B6 = GumbelHougaard

if __name__ == "__main__":
    # Example usage
    copula = GumbelHougaard(theta=2)
    footrule = copula.spearmans_footrule()
    ccop = copula.to_checkerboard()
    ccop_footrule = ccop.spearmans_footrule()
    ccop_xi = ccop.chatterjees_xi()
    ccop_rho = ccop.spearmans_rho()
    print(
        f"Footrule distance: {footrule}, Checkerboard footrule: {ccop_footrule}",
        f"Checkerboard xi: {ccop_xi}",
        f"Checkerboard rho: {ccop_rho}",
    )
