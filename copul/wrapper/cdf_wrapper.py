import sympy
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CDFWrapper(SymPyFuncWrapper):
    """
    Wrapper for copula cumulative distribution functions.

    This class handles the boundary conditions for copula CDFs:
    - C(0, v) = 0 (when u=0)
    - C(u, 0) = 0 (when v=0)
    - C(1, v) = v (when u=1)
    - C(u, 1) = u (when v=1)
    """

    def __call__(self, *args, **kwargs):
        free_symbols = {str(f): f for f in self._func.free_symbols}
        vars_, kwargs = self._prepare_call(args, kwargs)

        # Handle boundary conditions for standard u,v variable names
        if {"u", "v"}.issubset(set(free_symbols.keys())):
            if ("u", 0) in kwargs.items() or ("v", 0) in kwargs.items():
                return SymPyFuncWrapper(sympy.S.Zero)
            if ("u", 1) in kwargs.items():
                if "v" in kwargs:
                    return SymPyFuncWrapper(kwargs["v"])
                if "v" in free_symbols:
                    return SymPyFuncWrapper(free_symbols["v"])
            if ("v", 1) in kwargs.items():
                if "u" in kwargs:
                    return SymPyFuncWrapper(kwargs["u"])
                if "u" in free_symbols:
                    return SymPyFuncWrapper(free_symbols["u"])

        # Handle boundary conditions for u1,u2 variable names
        elif {"u1", "u2"}.issubset(set(free_symbols.keys())):
            if ("u1", 0) in kwargs.items() or ("u2", 0) in kwargs.items():
                return SymPyFuncWrapper(sympy.S.Zero)

        # Apply substitutions
        func = self._func.subs(vars_)

        # Return the result wrapped in CDFWrapper to maintain behavior
        return CDFWrapper(func)
