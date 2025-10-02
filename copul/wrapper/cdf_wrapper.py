import sympy
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CDFWrapper(SymPyFuncWrapper):
    """
    Wrapper for copula cumulative distribution functions.

    Boundary handling (dimension-agnostic):
      - C(..., 0, ...) = 0  if any coordinate is 0
      - C(1, ..., 1)   = 1  if all coordinates are 1 (for those present)

    Bivariate margins (u,v) or (u1,u2):
      - C(1, v) = v
      - C(u, 1) = u
    """

    def __call__(self, *args, **kwargs):
        # Map args/kwargs to a substitution dict using the base helper
        # vars_ is a dict {sympy.Symbol -> value}
        vars_, kwargs = self._prepare_call(args, kwargs)

        free_syms = list(self._func.free_symbols)
        free_by_name = {str(s): s for s in free_syms}

        # Build a "provided" map {sym_name -> numeric/sym value} from vars_
        provided = {str(s): v for s, v in vars_.items()}

        # ---------- Generic boundary checks (dimension-agnostic) ----------
        # ANY zero among provided -> 0
        for name, val in provided.items():
            try:
                if float(val) == 0.0:
                    return SymPyFuncWrapper(sympy.S.Zero)
            except Exception:
                # If not numeric (symbolic), skip this quick check
                pass

        # ALL ones (only if user provided values for ALL free u's) -> 1
        if free_syms and all(str(s) in provided for s in free_syms):
            all_one = True
            for s in free_syms:
                try:
                    if float(provided[str(s)]) != 1.0:
                        all_one = False
                        break
                except Exception:
                    all_one = False
                    break
            if all_one:
                return SymPyFuncWrapper(sympy.S.One)

        # ---------- Bivariate margin shortcuts ----------
        # Handle (u, v)
        if {"u", "v"}.issubset(free_by_name.keys()):
            if "u" in provided:
                try:
                    if float(provided["u"]) == 1.0:
                        # If v provided numerically -> return that number
                        if "v" in provided:
                            return SymPyFuncWrapper(sympy.Float(provided["v"]))
                        # Else return the symbol v
                        return SymPyFuncWrapper(free_by_name["v"])
                except Exception:
                    pass
            if "v" in provided:
                try:
                    if float(provided["v"]) == 1.0:
                        if "u" in provided:
                            return SymPyFuncWrapper(sympy.Float(provided["u"]))
                        return SymPyFuncWrapper(free_by_name["u"])
                except Exception:
                    pass

        # Handle (u1, u2)
        if {"u1", "u2"}.issubset(free_by_name.keys()):
            if "u1" in provided:
                try:
                    if float(provided["u1"]) == 1.0:
                        if "u2" in provided:
                            return SymPyFuncWrapper(sympy.Float(provided["u2"]))
                        return SymPyFuncWrapper(free_by_name["u2"])
                except Exception:
                    pass
            if "u2" in provided:
                try:
                    if float(provided["u2"]) == 1.0:
                        if "u1" in provided:
                            return SymPyFuncWrapper(sympy.Float(provided["u1"]))
                        return SymPyFuncWrapper(free_by_name["u1"])
                except Exception:
                    pass

        # ---------- Fall back: perform the substitution and return a CDFWrapper ----------
        func = self._func.subs(vars_)
        return CDFWrapper(func)
