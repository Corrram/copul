import sympy

from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class CD1Wrapper(SymPyFunctionWrapper):

    def __call__(self, *args, **kwargs):
        free_symbols = {str(f) for f in self._func.free_symbols}
        vars_, kwargs = self._prepare_call(args, kwargs)
        if {"u", "v"}.issubset(free_symbols):
            if ("v", 0) in kwargs.items():
                return sympy.S.Zero
            if ("v", 1) in kwargs.items():
                return sympy.S.One
        func = self._func.subs(vars_)
        if isinstance(func, sympy.Number):
            return float(func)
        return CD1Wrapper(func)
