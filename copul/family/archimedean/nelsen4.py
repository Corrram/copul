from typing import Optional

import numpy as np
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula


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

    def rvs(
        self, n: int = 1, random_state: Optional[int] = None, approximate: bool = False
    ) -> np.ndarray:
        """
        Fast vectorized Marshall–Olkin sampler for Gumbel–Hougaard.

        Steps:
          1) α = 1/θ
          2) Sample V ~ positive α-stable via Kanter's method
          3) Sample E1,E2 ~ Exp(1) i.i.d.
          4) Return (U, V) with U = exp(-(E1/V)^α), V = exp(-(E2/V)^α)

        Independence (θ≈1) is handled by returning U(0,1)^2 directly.
        """
        rng = np.random.default_rng(random_state)
        theta = float(self.theta)

        # Independence shortcut (θ = 1)
        if np.isclose(theta, 1.0):
            return rng.random((n, 2))

        # α-stable index (0 < α <= 1)
        alpha = 1.0 / theta

        # --- Kanter's sampler for positive α-stable -----------------------
        # U ~ Uniform(0, π), W ~ Exp(1)
        U = rng.uniform(0.0, np.pi, size=n)
        W = rng.exponential(scale=1.0, size=n)

        # Kanter (1975) representation:
        # S = [ sin(αU) / (sin U)^(1/α) ] * [ sin((1-α)U) / W ]^((1-α)/α)
        # S > 0 has Laplace transform E[e^{-s S}] = exp(-s^α)
        sinU = np.sin(U)
        # guard against rare 0s
        sinU[sinU == 0.0] = np.finfo(float).tiny

        part1 = np.sin(alpha * U) / (sinU ** (1.0 / alpha))
        part2 = (np.sin((1.0 - alpha) * U) / W) ** ((1.0 - alpha) / alpha)
        S = part1 * part2  # V in the frailty construction

        # --- Marshall–Olkin transform ------------------------------------
        E = rng.exponential(scale=1.0, size=(n, 2))
        # U_i = ψ(E_i / S) with ψ(s)=exp(-s^α)
        out = np.exp(-((E / S[:, None]) ** alpha))
        return out  # shape (n, 2)


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
