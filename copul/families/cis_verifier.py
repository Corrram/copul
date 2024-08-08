import logging

import numpy as np

log = logging.getLogger(__name__)


class CISVerifier:
    def __init__(self, cond_distr=1):
        self.cond_distr = cond_distr

    def is_cis(self, copul, range_min=None, range_max=None):
        log.info(f"Checking if {type(self).__name__} copula is CI")
        range_min = -10 if range_min is None else range_min
        n_interpolate = 20
        linspace = np.linspace(0.001, 0.999, 20)
        try:
            param = str(copul.params[0])
        except IndexError:
            is_ci, is_cd = self._is_copula_cis(copul, linspace)
            if is_ci:
                print("CI True for param: None")
            elif is_cd:
                print("CD True for param: None")
            else:
                print("False for param: None")
            return is_ci, is_cd
        interval = copul.intervals[param]
        range_min = float(max(interval.inf, range_min))
        if interval.left_open:
            range_min += 0.01
        param_range_max = 10 if range_max is None else range_max
        param_range_max = float(min(interval.end, param_range_max))
        if interval.right_open:
            param_range_max -= 0.01
        param_range = np.linspace(range_min, param_range_max, n_interpolate)
        points = linspace
        for param_value in param_range:
            param_dict = {param: param_value}
            my_copul = copul(**param_dict)
            is_cd, is_ci = self._is_copula_cis(my_copul, points)
            if is_ci:
                print(f"CI True for param: {param_value}")
            elif is_cd:
                print(f"CD True for param: {param_value}")
            else:
                print(f"False for param: {param_value}")
            if not is_ci or not is_cd:
                continue
        return is_ci, is_cd

    def _is_copula_cis(self, my_copul, points):
        is_ci = True
        is_cd = True
        if self.cond_distr == 1:
            cond_method = my_copul.cond_distr_1
        elif self.cond_distr == 2:
            cond_method = my_copul.cond_distr_2
        else:
            raise ValueError("cond_distr must be 1 or 2")
        try:
            cond_method = cond_method().func
        except TypeError:
            for v in points:
                for u, next_u in zip(points[:-1], points[1:]):
                    if self.cond_distr == 1:
                        val1 = cond_method(next_u, v) * 0.9999999
                        val2 = cond_method(u, v)
                    else:
                        val1 = cond_method(v, next_u) * 0.9999999
                        val2 = cond_method(v, u)
                    if val1 > val2:
                        is_ci = False
                    if val2 < val1:
                        is_cd = False
                    if not is_ci and not is_cd:
                        break
        else:
            for v in points:
                cond_distr_eval_u = self.cond_distr.subs(my_copul.v, v)
                for u, next_u in zip(points[:-1], points[1:]):
                    eval_u = cond_distr_eval_u.subs(my_copul.u, u)
                    eval_next_u = cond_distr_eval_u.subs(my_copul.u, next_u)
                    if eval_next_u * 0.9999999 > eval_u:
                        is_ci = False
                    if eval_next_u < eval_u * 0.9999999:
                        is_cd = False
                    if not is_ci and not is_cd:
                        break
                if not is_ci and not is_cd:
                    break
        return is_cd, is_ci
