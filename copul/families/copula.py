from abc import ABC

import numpy as np
import sympy


class Copula(ABC):
    params = None
    intervals = None
    log_cut_off = 4

    def __init__(self, n):
        self.u_symbols = sympy.symbols(f"u1:{n}")

    @classmethod
    def from_cdf(cls, cdf, params=None):
        sp_cdf = sympy.sympify(cdf)
        func_params = ["u1", "u2"]
        func_vars, free_symbols = cls._segregate_symbols(sp_cdf, func_params, params)
        n = len(func_vars)
        obj = cls._from_string(n, free_symbols)
        obj._cdf = sp_cdf.subs(func_vars[0], cls.u).subs(func_vars[1], cls.v)
        return obj

    @classmethod
    def _segregate_symbols(cls, func, func_params, copula_params):
        free_symbols = {str(x): x for x in func.free_symbols}
        if isinstance(func_params, str):
            func_params = [func_params]
        if len(free_symbols) == len(func_params):
            if copula_params:
                msg = "Params must be None if copula has only two free symbols"
                raise ValueError(msg)
            copula_vars = sorted([*free_symbols])
            symbols = sympy.symbols(copula_vars)
            for func_param, symbol in zip(func_params, symbols):
                setattr(cls, func_param, symbol)
            free_symbols = {}
        else:
            if isinstance(copula_params, str):
                copula_params = [copula_params]
            if copula_params is None and not set(func_params).issubset(free_symbols):
                msg = "Params must be a list if copula free symbols are not u and v"
                raise ValueError(msg)
            elif copula_params is None:
                copula_params = [x for x in free_symbols if x not in func_params]
            copula_vars = [str(x) for x in free_symbols if x not in copula_params]
            copula_vars.sort()
            for x in copula_vars:
                del free_symbols[x]
        return copula_vars, free_symbols

    @classmethod
    def _from_string(cls, n, free_symbols):
        obj = cls(n)
        obj._free_symbols = free_symbols  # Store free symbols for later use

        for key in free_symbols:
            setattr(obj, key, sympy.symbols(key, real=True))
            value = getattr(obj, key)
            obj.params.append(value)
            obj.intervals[value] = sympy.Interval(-np.inf, np.inf)
        return obj
