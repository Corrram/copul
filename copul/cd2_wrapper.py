import sympy

from copul.sympy_wrapper import SymPyFunctionWrapper


class CD2Wrapper(SymPyFunctionWrapper):

    def __call__(self, *args, **kwargs):
        free_symbols = {str(f) for f in self._func.free_symbols}
        if args and len(free_symbols) == len(args):
            kwargs = {str(f): arg for f, arg in zip(free_symbols, args)}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        assert set(kwargs).issubset(
            free_symbols
        ), f"keys: {set(kwargs)}, free symbols: {self._func.free_symbols}"
        vars_ = {f: kwargs[str(f)] for f in self._func.free_symbols if str(f) in kwargs}
        if {"u", "v"}.issubset(free_symbols):
            if ("u", 0) in kwargs.items():
                self._func = sympy.S.Zero
            if ("u", 1) in kwargs.items():
                self._func = sympy.S.One
        self._func = self._func.subs(vars_)
        if isinstance(self._func, sympy.Number):
            return float(self._func)
        return self
