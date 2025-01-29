import inspect
import logging
import random
import warnings

import numpy as np
import scipy.optimize as opt
import sympy

from copul.checkerboard.check_pi import CheckPi

log = logging.getLogger(__name__)


class CopulaSampler:
    err_counter = 0

    def __init__(self, copul, precision=3, random_state=None):
        self._copul = copul
        self._precision = precision
        self._random_state = random_state

    def rvs(self, n=1):
        """Sample a value from the copula"""
        cond_distr = self._copul.cond_distr_2
        sig = inspect.signature(cond_distr)
        params = set(sig.parameters.keys()) & set(self._copul.intervals)
        if self._random_state is not None:
            random.seed(self._random_state)
        if params or isinstance(self._copul, CheckPi):
            func2_ = cond_distr
        else:
            func_ = cond_distr().func
            func2_ = sympy.lambdify(self._copul.u_symbols, func_, ["numpy"])
        results = self._sample_val(func2_, n)
        return results

    def _sample_val(self, function, n=1):
        result = np.array([self.sample_val(function) for _ in range(n)])
        log.debug(self.err_counter)
        return result

    def sample_val(self, function):
        v = random.uniform(0, 1)
        t = random.uniform(0, 1)

        def func2(u: object) -> object:
            return function(u, v) - t

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                result = opt.root_scalar(
                    func2, x0=0.5, bracket=[0.000000001, 0.999999999]
                )
            except (ZeroDivisionError, ValueError, TypeError) as e:
                log.debug(f"{self._copul.__class__.__name__}; {type(e).__name__}: {e}")
                self.err_counter += 1
                return self._get_visual_solution(func2), v
            if not result.converged:
                if not result.iterations:
                    log.warning(f"{self._copul.__class__.__name__}; {result}")
                self.err_counter += 1
                return self._get_visual_solution(func2), v
        return result.root, v

    def _get_visual_solution(self, func):
        start = 10 ** (-self._precision)
        end = 1 - 10 ** (-self._precision)
        x = np.linspace(start, end, 10**self._precision)
        y = np.array([func(x_i) for x_i in x])
        return x[y.argmin()]
