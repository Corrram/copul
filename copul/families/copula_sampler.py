import logging
import warnings
import random

import numpy as np


import scipy.optimize as opt

log = logging.getLogger(__name__)


class CopulaSampler:
    err_counter = 0

    def __init__(self, copul):
        self._copul = copul

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

    @staticmethod
    def _get_visual_solution(func):
        x = np.linspace(0.001, 0.999, 1000)
        try:
            y = func(x)
        except ValueError:
            y = np.array([func(x_i) for x_i in x])
        return x[y.argmin()]
