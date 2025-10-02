import sympy
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CD1Wrapper(SymPyFuncWrapper):
    """
    Wrapper for ∂C/∂u (first coordinate). Bivariate boundaries:
      - CD1(u, 0) = 0
      - CD1(u, 1) = 1

    These boundary rules must hold even after partial substitution of u,
    i.e., when only v (or u2) remains free.
    """

    def __call__(self, *args, **kwargs):
        # Resolve positional/keyword substitutions into a dict {Symbol -> value}
        vars_, kwargs = self._prepare_call(args, kwargs)

        free_syms = list(self._func.free_symbols)
        by_name   = {str(s): s for s in free_syms}
        provided  = {str(s): v for s, v in vars_.items()}

        # ---- Boundary rule should trigger whenever v (or u2) is provided,
        #      even if u was already substituted away. ----
        # Prefer provided values; if not provided but v (or u2) is still free,
        # we'll fall through and substitute normally.

        # Case 1: standard names (u, v)
        if "v" in provided:
            try:
                v_val = float(provided["v"])
                if v_val == 0.0:
                    return SymPyFuncWrapper(sympy.S.Zero)
                if v_val == 1.0:
                    return SymPyFuncWrapper(sympy.S.One)
            except Exception:
                pass  # symbolic v value: skip boundary

        # Case 2: index names (u1, u2)
        if "u2" in provided:
            try:
                v_val = float(provided["u2"])
                if v_val == 0.0:
                    return SymPyFuncWrapper(sympy.S.Zero)
                if v_val == 1.0:
                    return SymPyFuncWrapper(sympy.S.One)
            except Exception:
                pass

        # Fallback: apply substitutions and keep wrapper semantics
        func = self._func.subs(vars_)
        return CD1Wrapper(func)
