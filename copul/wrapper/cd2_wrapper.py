import sympy
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CD2Wrapper(SymPyFuncWrapper):
    """
    Wrapper for the partial derivative of a copula with respect to the second argument.

    This class handles the boundary conditions for conditional distributions:
    - CD2(0, v) = 0 (when u=0)
    - CD2(1, v) = 1 (when u=1)
    """

    def __call__(self, *args, **kwargs):
        free_symbols = {str(f): f for f in self._func.free_symbols}

        # First process the arguments to create variable substitutions
        vars_, kwargs = self._prepare_call(args, kwargs)

        # Check boundary conditions
        if {"u", "v"}.issubset(set(free_symbols.keys())):
            if ("u", 0) in kwargs.items():
                return SymPyFuncWrapper(sympy.S.Zero)
            if ("u", 1) in kwargs.items():
                return SymPyFuncWrapper(sympy.S.One)

        # Apply substitutions
        func = self._func.subs(vars_)

        # Wrap the result in CD2Wrapper to maintain behavior in chained calls
        result = CD2Wrapper(func)

        # If we've made a substitution for u, check if it's a boundary value
        if "u" in kwargs and isinstance(kwargs["u"], (int, float)):
            if kwargs["u"] == 0:
                return SymPyFuncWrapper(sympy.S.Zero)
            if kwargs["u"] == 1:
                return SymPyFuncWrapper(sympy.S.One)

        return result
