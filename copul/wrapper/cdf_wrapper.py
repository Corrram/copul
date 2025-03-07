import sympy

from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CDFWrapper(SymPyFuncWrapper):
    def __call__(self, *args, **kwargs):
        free_symbols = {str(f): f for f in self._func.free_symbols}
        vars_, kwargs = self._prepare_call(args, kwargs)
        func = self._func
        if {"u", "v"}.issubset(set(free_symbols)):
            if ("u", 0) in kwargs.items() or ("v", 0) in kwargs.items():
                return SymPyFuncWrapper(sympy.S.Zero)
            if ("u", 1) in kwargs.items():
                func = free_symbols["v"]
            if ("v", 1) in kwargs.items():
                func = free_symbols["u"]
        elif {"u1", "u2"}.issubset(set(free_symbols)):
            if ("u1", 0) in kwargs.items() or ("u2", 0) in kwargs.items():
                return SymPyFuncWrapper(sympy.S.Zero)
        func = func.subs(vars_)
        # if isinstance(func, sympy.Number):
        #     return float(func)
        return CDFWrapper(func)
