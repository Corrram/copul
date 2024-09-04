import sympy


class SymPyFuncWrapper:
    def __init__(self, sympy_func):
        if isinstance(sympy_func, SymPyFuncWrapper):
            sympy_func = sympy_func.func
        type_ = type(sympy_func)
        allowed = (sympy.Expr, float)
        assert isinstance(
            sympy_func, allowed
        ), f"Function must be from sympy, but is {type_}"
        self._func = sympy_func

    def __str__(self):
        return str(self._func)

    def __repr__(self):
        return repr(self._func)

    def __call__(self, *args, **kwargs):
        vars_, kwargs = self._prepare_call(args, kwargs)
        func = self._func.subs(vars_)
        if isinstance(func, sympy.Number):
            return float(func)
        return SymPyFuncWrapper(func)

    def _prepare_call(self, args, kwargs):
        free_symbols = sorted([str(f) for f in self._func.free_symbols])
        if args and len(free_symbols) == len(args):
            if kwargs:
                raise ValueError("Either args or kwargs, not both currently")
            kwargs = {str(f): arg for f, arg in zip(free_symbols, args)}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        assert set(kwargs).issubset(
            free_symbols
        ), f"keys: {set(kwargs)}, free symbols: {self._func.free_symbols}"
        vars_ = {f: kwargs[str(f)] for f in self._func.free_symbols if str(f) in kwargs}
        return vars_, kwargs

    @property
    def func(self):
        return self._func

    def subs(self, *args, **kwargs):
        self._func = self._func.subs(*args, **kwargs)
        return self

    def diff(self, *args, **kwargs):
        self._func = self._func.diff(*args, **kwargs)
        return self

    def to_latex(self):
        return sympy.latex(self._func)

    def evalf(self):
        return self._func.evalf()
