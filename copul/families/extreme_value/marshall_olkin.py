import sympy

from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class MarshallOlkin(ExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return self.alpha_1 == self.alpha_2

    _alpha_1, _alpha_2 = sympy.symbols("alpha_1 alpha_2", nonnegative=True)
    params = [_alpha_1, _alpha_2]

    intervals = {
        "alpha_1": sympy.Interval(0, 1, left_open=False, right_open=False),
        "alpha_2": sympy.Interval(0, 1, left_open=False, right_open=False),
    }

    @property
    def is_absolutely_continuous(self):
        return (self._alpha_1 == 0) | (self._alpha_2 == 0)

    @property
    def alpha_1(self):
        if isinstance(self._alpha_1, property):
            return self._alpha_1.fget(self)
        return self._alpha_1

    @alpha_1.setter
    def alpha_1(self, value):
        self._alpha_1 = value

    @property
    def alpha_2(self):
        if isinstance(self._alpha_2, property):
            return self._alpha_2.fget(self)
        return self._alpha_2

    @alpha_2.setter
    def alpha_2(self, value):
        self._alpha_2 = value

    @property
    def pickands(self):
        func = sympy.Max(1 - self.alpha_1 * (1 - self.t), 1 - self.alpha_2 * self.t)
        return SymPyFunctionWrapper(func)

    @property
    def cdf(self):
        if self.alpha_1 == self.alpha_2 == 0:
            return SymPyFunctionWrapper(self.u * self.v)
        arg1 = self.v * self.u ** (1 - self.alpha_1)
        arg2 = self.u * self.v ** (1 - self.alpha_2)
        cdf = sympy.Min(arg1, arg2)
        return SymPyFunctionWrapper(cdf)

    def cond_distr_1(self):
        u = self.u
        v = self.v
        alpha_1 = self.alpha_1
        alpha2 = self.alpha_2
        heavy_expr = u * v ** (1 - alpha2) - u ** (1 - alpha_1) * v
        return (
            u * v ** (1 - alpha2) * sympy.Heaviside(-heavy_expr)
            - u ** (1 - alpha_1) * v * (alpha_1 - 1) * sympy.Heaviside(heavy_expr)
        ) / u

    def cond_distr_2(self):
        u = self.u
        v = self.v
        alpha1 = self.alpha_1
        alpha2 = self.alpha_2
        heavy_expr = -u * v ** (1 - alpha2) + u ** (1 - alpha1) * v
        cond_distr = (
            u * v ** (1 - alpha2) * (1 - alpha2) * sympy.Heaviside(heavy_expr)
            + u ** (1 - alpha1) * v * sympy.Heaviside(-heavy_expr)
        ) / v
        return SymPyFunctionWrapper(cond_distr)

    def _squared_cond_distr_1(self, u, v):
        alpha1 = self.alpha_1
        alpha2 = self.alpha_2
        return (
            u
            * v ** (1 - alpha2)
            * sympy.Heaviside(-u * v ** (1 - alpha2) + u ** (1 - alpha1) * v)
            - u ** (1 - alpha1)
            * v
            * (alpha1 - 1)
            * sympy.Heaviside(u * v ** (1 - alpha2) - u ** (1 - alpha1) * v)
        ) ** 2 / u**2

    # def xi(self):
    #     alpha_1 = self.alpha_1
    #     alpha_2 = self.alpha_2
    #     xi = (-(alpha_1**2) * alpha_2 + 2 * alpha_1 * alpha_2 - 3 * alpha_1 - alpha_2) / (
    #         3 * (2 * alpha_1 * alpha_2 - 3 * alpha_1 - alpha_2)
    #     )
    #     xi = sympy.simplify(6 * xi - 2)
    #     print("xi: ", xi)
    #     print("xi latex: ", sympy.latex(xi))
    #     return SymPyFunctionWrapper(xi)

    def _xi_int_1(self, v):
        u = self.u
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        integrand_1 = (u * v ** (1 - alpha_2)) ** 2 / u**2
        integrand_2 = (u ** (1 - alpha_1) * v * (alpha_1 - 1)) ** 2 / u**2
        int_1 = sympy.simplify(
            sympy.integrate(integrand_1, (u, 0, v ** (alpha_2 / alpha_1)))
        )
        int_2 = sympy.simplify(
            sympy.integrate(integrand_2, (u, v ** (alpha_2 / alpha_1), 1))
        )
        int_2 = (
            v**2
            * (alpha_1 - 1) ** 2
            * (v ** ((alpha_2 / alpha_1) - 2 * alpha_2) - 1)
            / (2 * alpha_1 - 1)
        )
        int_2 = sympy.simplify(int_2)
        print(sympy.latex(int_2))
        return sympy.simplify(int_1 + int_2)

    def _xi_int_1(self, v):
        u = self.u
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        integrand_1 = (u * v ** (1 - alpha_2)) ** 2 / u**2
        integrand_2 = (u ** (1 - alpha_1) * v * (alpha_1 - 1)) ** 2 / u**2
        print(sympy.latex(sympy.simplify(integrand_1)))
        print(sympy.latex(sympy.simplify(integrand_2)))
        int_1 = sympy.simplify(
            sympy.integrate(integrand_1, (u, 0, v ** (alpha_2 / alpha_1)))
        )
        int_2 = sympy.simplify(
            sympy.integrate(integrand_2, (u, v ** (alpha_2 / alpha_1), 1))
        )
        int_2 = sympy.simplify(int_2)
        print(sympy.latex(int_1))
        print(sympy.latex(int_2))
        return sympy.simplify(int_1 + int_2)

    # def _xi_int_2(self):
    #     v = self.v
    #     alpha_1 = self.alpha_1
    #     alpha_2 = self.alpha_2
    #     integrand = (
    #         v**2
    #         * (
    #             -(alpha_1**2)
    #             + alpha_1**2 * v ** (alpha_2 / alpha_1) / v ** (2 * alpha_2)
    #             + 2 * alpha_1
    #             - 1
    #         )
    #         / (2 * alpha_1 - 1)
    #     )
    #     return sympy.simplify(sympy.integrate(integrand, (v, 0, 1)))


MarshallOlkinDiag = lambda: MarshallOlkin()(alpha2=MarshallOlkin.alpha_1)
