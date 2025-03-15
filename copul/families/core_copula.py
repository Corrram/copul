import copy
from abc import ABC

import sympy

from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CoreCopula(ABC):
    params = []
    intervals = {}
    log_cut_off = 4
    _cdf = None
    _free_symbols = {}

    def __str__(self):
        return self.__class__.__name__

    def __init__(self, dimension, *args, **kwargs):
        self.u_symbols = sympy.symbols(f"u1:{dimension + 1}")
        self.dim = dimension
        self._are_class_vars(kwargs)
        for i in range(len(args)):
            kwargs[str(self.params[i])] = args[i]
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(self, k, v)
        self.params = [param for param in self.params if str(param) not in kwargs]
        self.intervals = {
            k: v for k, v in self.intervals.items() if str(k) not in kwargs
        }

    def __call__(self, *args, **kwargs):
        new_copula = copy.copy(self)
        self._are_class_vars(kwargs)
        for i in range(len(args)):
            kwargs[str(self.params[i])] = args[i]
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(new_copula, k, v)
        new_copula.params = [param for param in self.params if str(param) not in kwargs]
        new_copula.intervals = {
            k: v for k, v in self.intervals.items() if str(k) not in kwargs
        }
        return new_copula

    def _set_params(self, args, kwargs):
        if args and len(args) <= len(self.params):
            for i in range(len(args)):
                kwargs[str(self.params[i])] = args[i]
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    @property
    def parameters(self):
        return self.intervals

    @property
    def is_absolutely_continuous(self) -> bool:
        raise NotImplementedError("This method must be implemented in a subclass")

    @property
    def is_symmetric(self) -> bool:
        raise NotImplementedError("This method must be implemented in a subclass")

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
        return CDFWrapper(expr)(*args, **kwargs)

    def cond_distr(self, i, u=None):
        assert i in range(1, self.dim + 1)
        result = SymPyFuncWrapper(sympy.diff(self.cdf, self.u_symbols[i - 1]))
        if u is None:
            return result
        return result(*u)

    def cond_distr_1(self, u=None):
        result = SymPyFuncWrapper(sympy.diff(self.cdf, self.u_symbols[0]))
        if u is None:
            return result
        return result(*u)

    def cond_distr_2(self, u=None):
        result = SymPyFuncWrapper(sympy.diff(self.cdf, self.u_symbols[1]))
        if u is None:
            return result
        return result(*u)

    def pdf(self, u=None):
        term = self.cdf
        for u_symbol in self.u_symbols:
            term = sympy.diff(term, u_symbol)
        pdf = SymPyFuncWrapper(term)
        return pdf(u)
