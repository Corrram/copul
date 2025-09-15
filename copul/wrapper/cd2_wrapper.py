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
        import numpy as _np

        free_symbols = {str(f): f for f in self._func.free_symbols}

        # Prepare substitutions
        vars_, kwargs = self._prepare_call(args, kwargs)

        # Boundary handling ONLY for scalar u (arrays are handled later by numpy_func)
        if {"u", "v"}.issubset(set(free_symbols.keys())):
            u_val = kwargs.get("u", None)
            if u_val is not None and (
                _np.isscalar(u_val) or isinstance(u_val, (int, float))
            ):
                try:
                    u_float = float(u_val)
                except Exception:
                    u_float = None
                if u_float == 0.0:
                    return SymPyFuncWrapper(sympy.S.Zero)
                if u_float == 1.0:
                    return SymPyFuncWrapper(sympy.S.One)

        # Apply substitutions
        func = self._func.subs(vars_)

        # Wrap again to keep CD2 semantics on further calls
        result = CD2Wrapper(func)

        # Redundant scalar boundary guard (kept for full backward compatibility)
        if "u" in kwargs and (
            _np.isscalar(kwargs["u"]) or isinstance(kwargs["u"], (int, float))
        ):
            try:
                u_float = float(kwargs["u"])
            except Exception:
                u_float = None
            if u_float == 0.0:
                return SymPyFuncWrapper(sympy.S.Zero)
            if u_float == 1.0:
                return SymPyFuncWrapper(sympy.S.One)

        return result
