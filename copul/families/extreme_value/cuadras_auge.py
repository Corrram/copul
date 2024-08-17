import sympy

from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.upper_frechet import UpperFrechet
from copul.sympy_wrapper import SymPyFunctionWrapper


class CuadrasAuge(ExtremeValueCopula):
    """
    Cuadras-Auge copula, special case of the Marshall-Olkin copula.
    """

    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sympy.symbols("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, 1, left_open=False, right_open=False)}

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            self.delta = args[0]
        if "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return IndependenceCopula()(**kwargs)
        if "delta" in kwargs and kwargs["delta"] == 1:
            del kwargs["delta"]
            return UpperFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self):
        return self.delta == 0

    @property
    def pickands(self):
        return 1 - self.delta * sympy.Min(1 - self.t, self.t)

    @property
    def cdf(self):
        cdf = sympy.Min(self.u, self.v) ** self.delta * (self.u * self.v) ** (
            1 - self.delta
        )
        return SymPyFunctionWrapper(cdf)

    def cond_distr_1(self) -> SymPyFunctionWrapper:
        u = self.u
        v = self.v
        delta = self.delta
        cond_distr_1 = (
            v ** (1 - delta)
            * (
                delta * u * sympy.Heaviside(-u + v)
                - delta * sympy.Min(u, v)
                + sympy.Min(u, v)
            )
            * sympy.Min(u, v) ** (delta - 1)
            / u**delta
        )
        return SymPyFunctionWrapper(cond_distr_1)

    def _squared_cond_distr_1(self, v, u):
        delta = self.delta
        func = (
            (u * v) ** (2 - 2 * delta)
            * (delta * u * sympy.Heaviside(-u + v) - (delta - 1) * sympy.Min(u, v)) ** 2
            * sympy.Min(u, v) ** (2 * delta - 2)
            / u**2
        )
        return sympy.simplify(func)

    def _xi_int_1(self, v):
        delta = self.delta
        u = self.u
        func_u_lower_v = (
            (u * v) ** (2 - 2 * delta)
            * (delta * u - (delta - 1) * u) ** 2
            * u ** (2 * delta - 2)
            / u**2
        )
        func_u_greater_v = (delta - 1) ** 2 * v**2 / u ** (2 * delta)
        int1 = sympy.simplify(sympy.integrate(func_u_lower_v, (u, 0, v)))
        # int2 = sympy.simplify(sympy.integrate(func_u_greater_v, (u, v, 1)))
        int2 = sympy.integrate(func_u_greater_v, (u, v, 1))
        # int2 = -v**2*v**(1 - 2*delta)*(delta - 1)**2/(1 - 2*delta) + v**2*(delta - 1)**2/(1 - 2*delta)
        print("sub int1 sympy: ", int1)
        print("sub int1: ", sympy.latex(int1))
        print("sub int2 sympy: ", int2)
        print("sub int2: ", sympy.latex(int2))
        return sympy.simplify(int1 + int2)

    def xi(self):
        print("xi")
        int_1 = self._xi_int_1(self.v)
        print("int_1 sympy: ", int_1)
        print("int_1: ", sympy.latex(int_1))
        int_2 = self._xi_int_2()
        print("int_2 sympy: ", int_2)
        print("int_2: ", sympy.latex(int_2))
        xi = self._xi()
        print("xi sympy: ", xi)
        print("xi: ", sympy.latex(xi))
        return xi


B12 = CuadrasAuge
