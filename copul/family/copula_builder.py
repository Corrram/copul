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

    @classmethod
    def from_cond_distr_1(cls, cond):
        """
        Build a bivariate copula from the conditional CDF F_{V|U=u}(v) = ∂C/∂u (u, v).
        """
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

        sp_cond = sympy.sympify(cond, locals=local_dict, evaluate=False)

        # split symbols into params vs. function vars
        free_syms = list(sp_cond.free_symbols)
        param_syms = [s for s in free_syms if cls._is_greek(str(s)) and str(s) != "pi"]
        func_syms = [s for s in free_syms if s not in param_syms]

        obj = cls._from_string(2, [str(s) for s in param_syms])

        # map variables robustly to (u, v)
        sub_map = {}
        for s in func_syms:
            if str(s) == "u":
                sub_map[s] = obj.u
            elif str(s) == "v":
                sub_map[s] = obj.v
        remaining = [s for s in func_syms if s not in sub_map]
        if obj.u not in sub_map.values() and remaining:
            sub_map[remaining.pop(0)] = obj.u
        if obj.v not in sub_map.values() and remaining:
            sub_map[remaining.pop(0)] = obj.v

        sp_cond = sp_cond.xreplace(sub_map)
        for ps in param_syms:
            sp_cond = sp_cond.subs(ps, obj._free_symbols[str(ps)])

        # keep u and v visible to wrappers expecting both symbols
        if obj.u not in sp_cond.free_symbols:
            sp_cond = sympy.Add(sp_cond, obj.u, -obj.u, evaluate=False)
        if obj.v not in sp_cond.free_symbols:
            sp_cond = sympy.Add(sp_cond, obj.v, -obj.v, evaluate=False)

        # C(u,v) = ∫_0^u F_{V|U=s}(v) ds
        s = sympy.symbols("__int_u", real=True, nonnegative=True)
        cdf_expr = sympy.integrate(sp_cond.subs(obj.u, s), (s, 0, obj.u))
        cdf_expr = sympy.Add(cdf_expr, obj.u**2 / 2, -(obj.u**2) / 2, evaluate=False)
        obj._cdf_expr = cdf_expr
        obj._pdf_expr = sympy.diff(cdf_expr, obj.u, obj.v)
        return obj

    @classmethod
    def from_cond_distr_2(cls, cond):
        """
        Build a bivariate copula from the conditional CDF F_{U|V=v}(u) = ∂C/∂v (u, v).
        """
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

        sp_cond = sympy.sympify(cond, locals=local_dict, evaluate=False)

        free_syms = list(sp_cond.free_symbols)
        param_syms = [s for s in free_syms if cls._is_greek(str(s)) and str(s) != "pi"]
        func_syms = [s for s in free_syms if s not in param_syms]

        obj = cls._from_string(2, [str(s) for s in param_syms])

        # map variables robustly to (u, v)
        sub_map = {}
        for s in func_syms:
            if str(s) == "u":
                sub_map[s] = obj.u
            elif str(s) == "v":
                sub_map[s] = obj.v
        remaining = [s for s in func_syms if s not in sub_map]
        if obj.u not in sub_map.values() and remaining:
            sub_map[remaining.pop(0)] = obj.u
        if obj.v not in sub_map.values() and remaining:
            sub_map[remaining.pop(0)] = obj.v

        sp_cond = sp_cond.xreplace(sub_map)
        for ps in param_syms:
            sp_cond = sp_cond.subs(ps, obj._free_symbols[str(ps)])

        # keep u and v visible to wrappers expecting both symbols
        if obj.u not in sp_cond.free_symbols:
            sp_cond = sympy.Add(sp_cond, obj.u, -obj.u, evaluate=False)
        if obj.v not in sp_cond.free_symbols:
            sp_cond = sympy.Add(sp_cond, obj.v, -obj.v, evaluate=False)

        # C(u,v) = ∫_0^v F_{U|V=t}(u) dt
        t = sympy.symbols("__int_v", real=True, nonnegative=True)
        cdf_expr = sympy.integrate(sp_cond.subs(obj.v, t), (t, 0, obj.v))
        cdf_expr = sympy.Add(cdf_expr, obj.v**2 / 2, -(obj.v**2) / 2, evaluate=False)
        obj._cdf_expr = cdf_expr
        obj._pdf_expr = sympy.diff(cdf_expr, obj.u, obj.v)
        return obj


def from_cdf(cdf):
    return CopulaBuilder.from_cdf(cdf)


def from_pdf(pdf):
    return CopulaBuilder.from_pdf(pdf)


def from_cond_distr_1(cond):
    """Build from F_{V|U=u}(v) = ∂C/∂u (u, v)."""
    return CopulaBuilder.from_cond_distr_1(cond)


def from_cond_distr_2(cond):
    """Build from F_{U|V=v}(u) = ∂C/∂v (u, v)."""
    return CopulaBuilder.from_cond_distr_2(cond)


if __name__ == "__main__":
    cond1_str = "Piecewise((v, v > 1/2), (1/2, (v <= 1/2) & (u < 2*v)), (0, True))"
    copula = from_cond_distr_1(cond1_str)
    copula.plot_cond_distr_1()
