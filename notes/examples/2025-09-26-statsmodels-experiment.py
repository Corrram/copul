import numpy as np
from statsmodels.distributions.copula.copulas import Copula


class DiagonalCopula(Copula):
    """C(u, v) = min(u, v): Fréchet–Hoeffding upper bound."""

    def __init__(self, k_dim=2):
        super().__init__(k_dim=k_dim)
        if k_dim != 2:
            raise ValueError("DiagonalCopula is 2D.")

    def cdf(self, u, args=()):
        u = np.asarray(u)
        return np.min(u, axis=1)

    def pdf(self, u, args=()):
        # No density in the usual sense (singular measure)
        raise NotImplementedError("No standard density on the diagonal.")

    def rvs(self, nobs=1, args=(), random_state=None):
        rng = np.random.default_rng(random_state)
        z = rng.uniform(size=nobs)
        return np.column_stack([z, z])


C = DiagonalCopula()
samples = C.rvs(nobs=5, random_state=123)
print(samples)
print(C.cdf(samples))
