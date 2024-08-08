import logging
import random
import warnings
from abc import ABC

import numpy as np
import sympy
from scipy import optimize

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula

log = logging.getLogger(__name__)


class HeavyComputeArch(ArchimedeanCopula, ABC):
    err_counter = 0

    def rvs(self, n=1):
        """Sample a value from the copula"""
        results = []
        for _ in range(n):
            v = random.uniform(0, 1)
            sympy_func = self.cond_distr_2().subs(self.v, v).func
            function = sympy.lambdify(self.u, sympy_func, ["numpy"])
            sympy_func = self.cond_distr_2().subs(self.v, v).func
            result = np.array(
                [self._sample_values(function, v, sympy_func) for _ in range(1)]
            )
            results.append(result)
            # array of lists to array
        log.info(self.err_counter)
        return np.concatenate(results)

    def _sample_values(self, function, v, sympy_func):
        t = random.uniform(0, 1)

        def func2(u):
            return function(u) - t

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                result = optimize.root_scalar(
                    func2, x0=0.5, bracket=[0.00000001, 0.99999999]
                )
            except (ZeroDivisionError, ValueError, TypeError) as e:
                log.debug(f"{self.__class__.__name__}; {type(e).__name__}: {e}")
                self.err_counter += 1
                return self._get_visual_solution(sympy_func - t), v
            if not result.converged:
                if not result.iterations:
                    log.warning(
                        f"{self.__class__.__name__}: {result.flag} - {result.root}"
                    )
                self.err_counter += 1
                return self._get_visual_solution(sympy_func - t), v
            return result.root, v

    def _get_visual_solution(self, func):
        x_values = np.linspace(0.001, 0.999, 100)
        y_values = np.array([func.subs(self.u, val) for val in x_values])
        try:
            first_positive_index = np.where(y_values > 0)[0][0]
        except IndexError:
            # log.warning(f"{self.__class__.__name__}: {func} has no positive values")
            first_positive_index = np.argmax(y_values)
        return x_values[first_positive_index]
