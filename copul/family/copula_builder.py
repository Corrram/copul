import numpy as np
import sympy

from copul.family.core.copula import Copula
from copul.family.core.biv_copula import BivCopula


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
        for symbol in params:
            sp_cdf = sp_cdf.subs(symbol, obj._free_symbols[symbol])
        obj._cdf_expr = sp_cdf
        return obj

    @classmethod
    def from_pdf(cls, pdf):
        """
        Build a copula object from a PDF expression.

        Parameters
        ----------
        pdf : str | sympy.Expr
            A symbolic expression for c(u, v) in the bivariate case or
            c(u1, ..., ud) in the d-variate case. Greek-letter symbols
            are treated as parameters; all other symbols are taken as
            the copula's function variables.
        """
        # Bind greek names (except 'pi') to Symbols so they aren't parsed as special functions
        greek_names = [
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
            "rho",
            "sigma",
            "tau",
            "upsilon",
            "phi",
            "chi",
            "psi",
            "omega",
        ]
        local_dict = {name: sympy.symbols(name, real=True) for name in greek_names}

        # Keep zero terms so u/v/x/y/... remain in free_symbols even if multiplied by 0
        sp_pdf = sympy.sympify(pdf, locals=local_dict, evaluate=False)

        # Identify symbols: greek -> params, the rest -> function variables
        free_symbols = [str(s) for s in sp_pdf.free_symbols]
        params = [name for name in free_symbols if cls._is_greek(name) and name != "pi"]
        func_vars = [name for name in free_symbols if name not in params]
        n = len(func_vars)

        if n < 2:
            raise ValueError("PDF must depend on at least two variables (u, v).")

        # Create a copula shell and map variables
        obj = cls._from_string(n, params)
        func_vars = sorted(func_vars)

        # Replace function variables with the copula's variables
        if n == 2:
            sp_pdf = sp_pdf.subs(func_vars[0], obj.u).subs(func_vars[1], obj.v)
            vars_on_obj = [obj.u, obj.v]
        else:
            for i, name in enumerate(func_vars):
                sp_pdf = sp_pdf.subs(name, obj.u_symbols[i])
            vars_on_obj = list(obj.u_symbols)

        # Replace parameter names with the object's parameter symbols
        for name in params:
            sp_pdf = sp_pdf.subs(name, obj._free_symbols[name])

        # Store the PDF
        obj._pdf_expr = sp_pdf

        # Also construct and store a CDF by integrating the PDF from 0 to each variable
        # Use dummy symbols for integration bounds, then substitute upper limits.
        cdf_expr = sp_pdf
        for var in vars_on_obj:
            s = sympy.symbols(f"__int_{str(var)}", real=True, nonnegative=True)
            cdf_expr = sympy.integrate(cdf_expr.subs(var, s), (s, 0, var))

        obj._cdf_expr = cdf_expr

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
            obj.intervals[str(value)] = sympy.Interval(-np.inf, np.inf)
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


def from_pdf(pdf):
    return CopulaBuilder.from_pdf(pdf)
