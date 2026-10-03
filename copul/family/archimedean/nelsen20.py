import logging

import numpy as np
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.archimedean.heavy_compute_arch import HeavyComputeArch
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.cd2_wrapper import CD2Wrapper

log = logging.getLogger(__name__)


class Nelsen20(HeavyComputeArch):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)
    special_cases = {0: BivIndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    # -- overflow-free numerics (exp(theta/t) overflows for small t) -------
    def _stable_numerics(self):
        from copul.family.archimedean._exp_generator import nelsen20

        theta = float(self.theta)
        if theta <= 0:
            return None
        cache = getattr(self, "_stable_cache", None)
        if cache is None or cache[0] != theta:
            cache = (theta, nelsen20(theta))
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
        return sympy.exp(self.t ** (-self.theta)) - sympy.exp(1)

    @property
    def _raw_inv_generator(self):
        return sympy.log(self.y + sympy.E) ** (-1 / self.theta)

    @property
    def _cdf_expr(self):
        return sympy.log(
            sympy.exp(self.u ** (-self.theta)) + sympy.exp(self.v ** (-self.theta)) - np.e
        ) ** (-1 / self.theta)

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        cond_distr = 1 / (
            self.v ** (theta + 1)
            * (
                sympy.exp(self.u ** (-theta) - self.v ** (-theta))
                + 1
                - np.e * sympy.exp(-(self.v ** (-theta)))
            )
            * sympy.log(sympy.exp(self.u ** (-theta)) + sympy.exp(self.v ** (-theta)) - np.e)
            ** ((theta + 1) / theta)
        )
        return CD2Wrapper(cond_distr)(u, v)
