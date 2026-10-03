import numpy as np
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi


class Nelsen19(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)
    special_cases = {0: PiOverSigmaMinusPi}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    # -- overflow-free numerics (exp(theta/t) overflows for small t) -------
    def _stable_numerics(self):
        from copul.family.archimedean._exp_generator import nelsen19

        theta = float(self.theta)
        if theta <= 0:
            return None
        cache = getattr(self, "_stable_cache", None)
        if cache is None or cache[0] != theta:
            cache = (theta, nelsen19(theta))
            self._stable_cache = cache
        return cache[1]

    def _numeric_callables(self):
        """Hook for :func:`copul.measures.backend.numeric_backend`."""
        stable = self._stable_numerics()
        if stable is None:
            raise NotImplementedError("independence special case")
        return stable.callables()

    def cdf_vectorized(self, u, v):
        try:
            stable = self._stable_numerics()
        except (TypeError, ValueError):
            stable = None
        if stable is None:
            return super().cdf_vectorized(u, v)
        return stable.cdf(u, v)

    @property
    def _raw_generator(self):
        return sympy.exp(self.theta / self.t) - sympy.exp(self.theta)

    @property
    def _raw_inv_generator(self):
        return self.theta / sympy.log(self.y + sympy.exp(self.theta))

    @property
    def _cdf_expr(self):
        return self.theta / sympy.log(
            -sympy.exp(self.theta) + sympy.exp(self.theta / self.u) + sympy.exp(self.theta / self.v)
        )
