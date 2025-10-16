import sympy
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CD2Wrapper(SymPyFuncWrapper):
    """
    Wrapper for ∂C/∂v (second coordinate). Bivariate boundaries:
      - CD2(0, v) = 0
      - CD2(1, v) = 1

    These boundary rules must hold even after partial substitution of v
    (i.e., when only u / u1 remains free).
    """

    def __call__(self, *args, **kwargs):
        # Map positional/keyword args to {Symbol -> value}
        vars_, kwargs = self._prepare_call(args, kwargs)

        free_syms = list(self._func.free_symbols)
        {str(s): s for s in free_syms}
        provided = {str(s): v for s, v in vars_.items()}

        # ---- Boundary rule should trigger whenever u (or u1) is provided,
        #      even if v was already substituted away. ----

        # Standard names (u, v)
        if "u" in provided:
            try:
                u_val = float(provided["u"])
                if u_val == 0.0:
                    return SymPyFuncWrapper(sympy.S.Zero)
                if u_val == 1.0:
                    return SymPyFuncWrapper(sympy.S.One)
            except Exception:
                pass  # symbolic value — skip boundary fast-path

        # Index names (u1, u2)
        if "u1" in provided:
            try:
                u_val = float(provided["u1"])
                if u_val == 0.0:
                    return SymPyFuncWrapper(sympy.S.Zero)
                if u_val == 1.0:
                    return SymPyFuncWrapper(sympy.S.One)
            except Exception:
                pass

        # Fallback: apply substitutions and keep wrapper semantics
        func = self._func.subs(vars_)
        return CD2Wrapper(func)
