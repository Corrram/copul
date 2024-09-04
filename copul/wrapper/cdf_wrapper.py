import sympy

from copul.families.bivcopula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class CDFWrapper(SymPyFunctionWrapper):

    def __call__(self, *args, **kwargs):
        free_symbols = {str(f) for f in self._func.free_symbols}
        vars_, kwargs = self._prepare_call(args, kwargs)
        func = self._func
        if {"u", "v"}.issubset(free_symbols):
            if ("u", 0) in kwargs.items() or ("v", 0) in kwargs.items():
                return sympy.S.Zero
            if ("u", 1) in kwargs.items():
                func = BivCopula.v
            if ("v", 1) in kwargs.items():
                func = BivCopula.u
        func = func.subs(vars_)
        if isinstance(func, sympy.Number):
            return float(func)
        return CDFWrapper(func)
