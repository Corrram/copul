import numpy as np
import sympy

from copul.families.bivcopula import BivCopula
from copul.families.copula import Copula


class CopulaBuilder:
    @classmethod
    def from_cdf(cls, cdf):
        sp_cdf = sympy.sympify(cdf)
        free_symbols = [str(symbol) for symbol in sp_cdf.free_symbols]
        # get greek letters from free symbols
        params = [symbol for symbol in free_symbols if cls._is_greek(symbol)]
        func_vars = [symbol for symbol in free_symbols if symbol not in params]
        n = len(func_vars)
        obj = cls._from_string(n, params)
        func_vars = sorted(func_vars)
        if n == 2:
            sp_cdf = sp_cdf.subs(func_vars[0], obj.u).subs(func_vars[1], obj.v)
        else:
            for i, symbol in enumerate(func_vars):
                sp_cdf = sp_cdf.subs(symbol, obj.u_symbols[i])
        for i, symbol in enumerate(params):
            sp_cdf = sp_cdf.subs(symbol, obj._free_symbols[symbol])
        obj._cdf = sp_cdf
        return obj

    @classmethod
    def _from_string(cls, n, params):
        if n == 2:
            obj = BivCopula()
        elif n > 2:
            obj = Copula(n)
        else:
            raise ValueError("n must be greater than 1")

        for key in params:
            setattr(obj, key, sympy.symbols(key, real=True))
            value = getattr(obj, key)
            obj.params.append(value)
            obj.intervals[value] = sympy.Interval(-np.inf, np.inf)
        obj._free_symbols = {symbol: getattr(obj, symbol) for symbol in params}
        return obj

    @staticmethod
    def _is_greek(character: str) -> bool:
        greek_letters = [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
            "nu",
            "xi",
            "omicron",
            "pi",
            "rho",
            "sigma",
            "tau",
            "upsilon",
            "phi",
            "chi",
            "psi",
            "omega",
        ]

        return character.lower() in greek_letters


def from_cdf(cdf):
    return CopulaBuilder.from_cdf(cdf)
