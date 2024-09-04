import copy
from abc import ABC

import sympy

from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Copula(ABC):
    params = []
    intervals = {}
    log_cut_off = 4
    _cdf = None
    _free_symbols = {}

    def __init__(self, n):
        self.u_symbols = sympy.symbols(f"u1:{n+1}")

    def __call__(self, *args, **kwargs):
        new_copula = copy.copy(self)
        self._are_class_vars(kwargs)
        for i in range(len(args)):
            kwargs[str(self.params[i])] = args[i]
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(new_copula, k, v)
            # new_copula._cdf = new_copula.cdf.subs(getattr(self, k), v)
        new_copula.params = [param for param in self.params if str(param) not in kwargs]
        new_copula.intervals = {
            k: v for k, v in self.intervals.items() if str(k) not in kwargs
        }
        return new_copula

    def _are_class_vars(self, kwargs):
        class_vars = set(dir(self))
        assert set(kwargs).issubset(
            class_vars
        ), f"keys: {set(kwargs)}, free symbols: {class_vars}"

    def slice_interval(self, param, interval_start=None, interval_end=None):
        if not isinstance(param, str):
            param = str(param)
        left_open = self.intervals[param].left_open
        right_open = self.intervals[param].right_open
        if interval_start is None:
            interval_start = self.intervals[param].inf
        else:
            left_open = False
        if interval_end is None:
            interval_end = self.intervals[param].sup
        else:
            right_open = False
        self.intervals[param] = sympy.Interval(
            interval_start, interval_end, left_open, right_open
        )

    @property
    def cdf(self, *args, **kwargs):
        expr = self._cdf
        for key, value in self._free_symbols.items():
            expr = expr.subs(value, getattr(self, key))
        return SymPyFuncWrapper(expr)(*args, **kwargs)
