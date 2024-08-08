import sympy

from copul.families.abstract_copula import AbstractCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class CDFWrapper(SymPyFunctionWrapper):

    def __call__(self, **kwargs):
        free_symbols = {str(f) for f in self._func.free_symbols}
        assert set(kwargs).issubset(
            free_symbols
        ), f"keys: {set(kwargs)}, free symbols: {self._func.free_symbols}"
        vars_ = {f: kwargs[str(f)] for f in self._func.free_symbols if str(f) in kwargs}
        if {"u", "v"}.issubset(free_symbols):
            if ('u', 0) in kwargs.items() or ('v', 0) in kwargs.items():
                self._func = sympy.S.Zero
            if ('u', 1) in kwargs.items():
                self._func = AbstractCopula.v
            if ('v', 1) in kwargs.items():
                self._func = AbstractCopula.u
        self._func = self._func.subs(vars_)
        return self
