import sympy

from copul.families.bivcopula import BivCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Raftery(BivCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sympy.symbols("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, 1, left_open=False, right_open=False)}

    def __call__(self, **kwargs):
        if "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return IndependenceCopula()(**kwargs)
        if "delta" in kwargs and kwargs["delta"] == 1:
            del kwargs["delta"]
            return UpperFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def cdf(self):
        u = self.u
        v = self.v
        d = self.delta
        cdf = sympy.Min(u, v) + (1 - d) / (1 + d) * (u * v) ** (1 / (1 - d)) * (
            1 - sympy.Max(u, v) ** (-(1 + d) / (1 - d))
        )
        return SymPyFuncWrapper(cdf)

    @property
    def pdf(self):
        pdf = self._b(sympy.Min(self.u, self.v), sympy.Max(self.u, self.v))
        return SymPyFuncWrapper(pdf)

    def _b(self, u, v):
        delta = self.delta
        return (
            (1 - delta**2) ** (-1)
            * u ** (delta / (1 - delta))
            * (delta * v ** (-1 / (1 - delta)) + v ** (delta / (1 - delta)))
        )

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return self.delta * (4 - 3 * self.delta) / (2 - self.delta) ** 2

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return 2 * self.delta / (3 - self.delta)

    @property
    def lambda_L(self):
        return 2 * self.delta / (1 + self.delta)

    @property
    def lambda_U(self):
        return 0

    def _squared_cond_distr_1(self, u, v):
        delta = self.delta
        term1 = (
            u
            * (u * v) ** (1 / (delta - 1))
            * (delta + 1)
            * sympy.Heaviside(-u + v)
            * sympy.Max(u, v)
        )
        term2 = (
            u
            * (delta + 1)
            * sympy.Heaviside(u - v)
            * sympy.Max(u, v) ** ((delta + 1) / (delta - 1))
        )
        term3 = (1 - sympy.Max(u, v) ** ((delta + 1) / (delta - 1))) * sympy.Max(u, v)
        full_expr = (term1 + term2 + term3) ** 2 / (
            u**2
            * (u * v) ** (2 / (delta - 1))
            * (delta + 1) ** 2
            * sympy.Max(u, v) ** 2
        )
        return full_expr

    def _xi_int_1(self, v):
        delta = self.delta
        u = self.u

        term1 = u * (u * v) ** (1 / (delta - 1)) * (delta + 1) * v
        term3 = (1 - v ** ((delta + 1) / (delta - 1))) * v
        func_u_lower_v = sympy.simplify(
            (term1 + term3) ** 2
            / (u**2 * (u * v) ** (2 / (delta - 1)) * (delta + 1) ** 2 * v**2)
        )

        term2 = u * (delta + 1) * u ** ((delta + 1) / (delta - 1))
        term3 = (1 - u ** ((delta + 1) / (delta - 1))) * u
        func_u_greater_v = sympy.simplify(
            (term2 + term3) ** 2
            / (u**2 * (u * v) ** (2 / (delta - 1)) * (delta + 1) ** 2 * u**2)
        )

        int2 = sympy.simplify(sympy.integrate(func_u_greater_v, (u, v, 1)))
        print("sub int2 sympy: ", int2)
        print("sub int2: ", sympy.latex(int2))

        int1 = sympy.simplify(sympy.integrate(func_u_lower_v, (u, 0, v)))
        print("sub int1 sympy: ", int1)
        print("sub int1: ", sympy.latex(int1))

        return sympy.simplify(int1 + int2)


# B9 = Raftery
