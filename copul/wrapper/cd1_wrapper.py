import sympy
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CD1Wrapper(SymPyFuncWrapper):
    """
    Wrapper for the partial derivative of a copula with respect to the first argument.

    This class handles the boundary conditions for conditional distributions:
    - CD1(u, 0) = 0 (when v=0)
    - CD1(u, 1) = 1 (when v=1)
    """

    def __call__(self, *args, **kwargs):
        free_symbols = {str(f): f for f in self._func.free_symbols}

        # First process the arguments to create variable substitutions
        vars_, kwargs = self._prepare_call(args, kwargs)

        # Check boundary conditions
        if {"u", "v"}.issubset(set(free_symbols.keys())):
            if ("v", 0) in kwargs.items():
                return SymPyFuncWrapper(sympy.S.Zero)
            if ("v", 1) in kwargs.items():
                return SymPyFuncWrapper(sympy.S.One)

        # Apply substitutions
        func = self._func.subs(vars_)

        # Wrap the result in CD1Wrapper to maintain behavior in chained calls
        result = CD1Wrapper(func)

        # If we've made a substitution for v, check if it's a boundary value
        if "v" in kwargs and isinstance(kwargs["v"], (int, float)):
            if kwargs["v"] == 0:
                return SymPyFuncWrapper(sympy.S.Zero)
            if kwargs["v"] == 1:
                return SymPyFuncWrapper(sympy.S.One)

        return result
