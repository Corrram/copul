import itertools
import logging
import warnings

import numpy as np
import sympy
from sympy.logic.boolalg import BooleanFalse, BooleanTrue
from sympy.utilities.exceptions import SymPyDeprecationWarning

log = logging.getLogger(__name__)


class TP2Verifier:
    def __init__(self, range_min=None, range_max=None):
        self.range_min = range_min
        self.range_max = range_max

    def is_tp2(self, copul):
        log.info(f"Checking if {type(copul).__name__} copula is TP2")
        if (
            isinstance(copul.is_absolutely_continuous, bool)
            and not copul.is_absolutely_continuous
        ):
            return False
        range_min = -10 if self.range_min is None else self.range_min
        ranges = {}
        if len(copul.params) == 1:
            n_interpolate = 20
        elif len(copul.params) == 2:
            n_interpolate = 10
        else:
            n_interpolate = 6
        for param in copul.params:
            interval = copul.intervals[str(param)]
            range_min = float(max(interval.inf, range_min))
            if interval.left_open:
                range_min += 0.01
            param_range_max = 10 if self.range_max is None else self.range_max
            param_range_max = float(min(interval.end, param_range_max))
            if interval.right_open:
                param_range_max -= 0.01
            ranges[param] = np.linspace(range_min, param_range_max, n_interpolate)
        u = copul.u
        v = copul.v
        points = np.linspace(0.0001, 0.9999, 20)
        for param_values in itertools.product(*ranges.values()):
            param_dict = dict(zip(ranges.keys(), param_values))
            keys = [str(key) for key in ranges.keys()]
            param_dict_str = dict(zip(keys, param_values))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SymPyDeprecationWarning)
                my_copul = copul(**param_dict_str)
                if not my_copul.is_absolutely_continuous:
                    print("No density, False for params: ", param_dict)
                    continue
                my_log_pdf = sympy.log(my_copul.pdf)
            is_tp2 = True
            if not my_copul.is_absolutely_continuous:
                print("False for params: ", param_dict)
                continue
                # return False
            for i in range(len(points) - 1):
                for j in range(len(points) - 1):
                    if my_copul._check_extreme_mixed_term(
                        copul,
                        my_log_pdf,
                        str(u),
                        str(v),
                        points[i],
                        points[i + 1],
                        points[j],
                        points[j + 1],
                    ):
                        # return False
                        is_tp2 = False
                        break
                if not is_tp2:
                    break
            if is_tp2:
                print("True for params: ", param_dict)
            else:
                print("False for params: ", param_dict)
        return True

    def _check_extreme_mixed_term(self, copul, my_log_pdf, u, v, x1, x2, y1, y2):
        min_term = my_log_pdf.subs(u, x1).subs(v, y1)
        max_term = my_log_pdf.subs(u, x2).subs(v, y2)
        mix_term_1 = my_log_pdf.subs(u, x1).subs(v, y2)
        mix_term_2 = my_log_pdf.subs(u, x2).subs(v, y1)
        extreme_term = min_term + max_term
        mixed_term = mix_term_1 + mix_term_2
        try:
            comparison = extreme_term * 0.9999999999999 < mixed_term
        except TypeError:
            comparison = (
                extreme_term.as_real_imag()[0] * 0.9999999999999
                < mixed_term.as_real_imag()[0]
            )
        if not isinstance(comparison, (bool, BooleanFalse, BooleanTrue)):
            comparison = comparison.evalf()
        if not isinstance(comparison, (bool, BooleanFalse, BooleanTrue)):
            u = copul.u
            v = copul.v
            return self._check_extreme_mixed_term(
                copul, my_log_pdf, u, v, x1, x2, y1, y2
            )
        if comparison:
            # print("my_log_pdf: ", my_log_pdf)
            print("x1: ", x1, "x2: ", x2, "y1: ", y1, "y2: ", y2)
            # print("min_term: ", min_term)
            # print("max_term: ", max_term)
            # print("mix_term_1: ", mix_term_1)
            # print("mix_term_2: ", mix_term_2)
            # print("extreme_term: ", extreme_term)
            # print("mixed_term: ", mixed_term)
        return comparison
