from abc import ABC

import logging

import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy import optimize
from copul.families import concrete_expand_log, get_simplified_solution
from copul.families.abstract_copula import AbstractCopula
from copul.sympy_wrapper import SymPyFunctionWrapper

log = logging.getLogger(__name__)


class ArchimedeanCopula(AbstractCopula, ABC):
    _t_min = 0
    _t_max = 1
    y, t = sympy.symbols("y t", positive=True)
    theta = sympy.symbols("theta")
    theta_interval = None
    params = [theta]

    def __init__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
        super().__init__(**kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}({self.theta})"

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def intervals(self):
        return {"theta": self.theta_interval}

    @property
    def _generator(self):
        raise NotImplementedError

    @property
    def generator(self):
        return SymPyFunctionWrapper(self._generator)

    @property
    def theta_max(self):
        return self.theta_interval.closure.end

    @property
    def theta_min(self):
        return self.theta_interval.closure.inf

    @property
    def cdf(self):
        """Cumulative distribution function of the copula"""
        inv_gen_at_u = self._generator.subs(self.t, self.u)
        inv_gen_at_v = self._generator.subs(self.t, self.v)
        cdf = self.inv_generator.subs(self.y, inv_gen_at_u + inv_gen_at_v)
        return SymPyFunctionWrapper(get_simplified_solution(cdf.func))

    @property
    def pdf(self):
        """Probability density function of the copula"""
        first_diff = sympy.diff(self.cdf, self.u)
        return sympy.diff(first_diff, self.v)

    @property
    def inv_generator(self) -> SymPyFunctionWrapper:
        eq = sympy.Eq(self.y, self._generator)
        sol = sympy.solve([eq, self.theta > 0, self.y > 0], self.t)
        my_sol = sol[self.t] if isinstance(sol, dict) else sol[0]
        my_simplified_sol = get_simplified_solution(my_sol)
        return SymPyFunctionWrapper(my_simplified_sol)

    @property
    def first_deriv_of_inv_gen(self):
        diff = sympy.diff(self.inv_generator.func, self.y)
        return sympy.simplify(diff)

    def tau(self):
        inv_gen = self.generator.func
        log.debug("inv gen: ", inv_gen)
        log.debug("inv gen latex: ", sympy.latex(inv_gen))
        inv_gen_diff = sympy.diff(inv_gen, self.t)
        log.debug("inv gen diff: ", inv_gen_diff)
        log.debug("inv gen diff latex: ", sympy.latex(inv_gen_diff))
        frac = inv_gen / inv_gen_diff
        log.debug("frac: ", frac)
        log.debug("frac latex: ", sympy.latex(frac))
        integral = sympy.integrate(frac, (self.t, 0, 1))
        log.debug("integral: ", integral)
        log.debug("integral latex: ", sympy.latex(integral))
        tau = 1 + 4 * integral
        log.debug("tau: ", tau)
        log.debug("tau latex: ", sympy.latex(tau))
        return tau

    @property
    def second_deriv_of_inv_gen(self):
        first_diff = self.first_deriv_of_inv_gen
        second_diff = sympy.diff(first_diff, self.y)
        return sympy.simplify(second_diff)

    def ltd_char(self):
        return sympy.simplify(sympy.log(self.inv_generator.func))

    def diff2_ltd_char(self):
        beauty_func = self.ltd_char()
        diff2 = sympy.diff(beauty_func, self.y, 2)
        return sympy.simplify(diff2)

    @property
    def ci_char(self):
        minus_gen_deriv = -self.first_deriv_of_inv_gen
        beauty_deriv = concrete_expand_log(sympy.simplify(sympy.log(minus_gen_deriv)))
        return SymPyFunctionWrapper(beauty_deriv)

    def first_deriv_of_ci_char(self):
        chi_char_func = self.ci_char()
        return sympy.simplify(sympy.diff(chi_char_func, self.y))

    def second_deriv_of_ci_char(self):
        chi_char_func_deriv = self.first_deriv_of_ci_char()
        return sympy.simplify(sympy.diff(chi_char_func_deriv, self.y))

    def tp2_char(self, u, v):
        second_deriv = self.second_deriv_of_inv_gen.subs([(self.u, u), (self.v, v)])
        beauty_2deriv = concrete_expand_log(sympy.simplify(sympy.log(second_deriv)))
        print(sympy.latex(second_deriv))
        return SymPyFunctionWrapper(beauty_2deriv)

    def first_deriv_of_tp2_char(self):
        mtp2_char = self.tp2_char(self.u, self.v)
        return sympy.simplify(sympy.diff(mtp2_char.func, self.y))

    def second_deriv_of_tp2_char(self):
        return sympy.simplify(sympy.diff(self.tp2_char(self.u, self.v).func, self.y, 2))

    @property
    def log_der(self):
        minus_log_derivative = self.ci_char()
        first_deriv = self.first_deriv_of_ci_char()
        second_deriv = self.second_deriv_of_ci_char()
        return self._compute_log2_der_of(
            first_deriv, minus_log_derivative, second_deriv
        )

    @property
    def log2_der(self):
        log_second_derivative = self.tp2_char(self.u, self.v)
        first_deriv = self.first_deriv_of_tp2_char()
        second_deriv = self.second_deriv_of_tp2_char()
        return self._compute_log2_der_of(
            first_deriv, log_second_derivative, second_deriv
        )

    def _compute_log2_der_of(self, first_deriv, log_second_derivative, second_deriv):
        log_der_lambda = sympy.lambdify([(self.y, self.theta)], second_deriv)
        bounds = [(self._t_min, self._t_max), (self.theta_min, self.theta_max)]
        starting_point = np.array(
            [
                min(self._t_min + 0.5, self._t_max),
                min(self.theta_min + 0.5, self.theta_max),
            ]
        )
        min_val = optimize.minimize(log_der_lambda, starting_point, bounds=bounds)
        return (
            log_second_derivative,
            first_deriv,
            second_deriv,
            [round(val, 2) for val in min_val.x],
            round(log_der_lambda(min_val.x), 2),
        )

    def compute_gen_max(self):
        try:
            limit = sympy.limit(self._generator, self.t, 0)
        except TypeError:
            limit = sympy.limit(
                self._generator.subs(self.theta, (self.theta_max - self.theta_min) / 2),
                self.t,
                0,
            )
        return sympy.simplify(limit)

    def lambda_L(self):
        expr = self.inv_generator(y=2 * self.y).func / self.inv_generator(y=self.y).func
        return sympy.limit(expr, self.y, sympy.oo, dir="-")

    def lambda_U(self):
        expr = (1 - self.inv_generator(y=2 * self.y).func) / (
            1 - self.inv_generator(y=self.y).func
        )
        return sympy.simplify(2 - sympy.limit(expr, self.y, 0, dir="+"))

    def plot_generator(self, start=0, stop=1):
        generator = sympy.lambdify(self.t, self.generator.func)
        inv_generator = sympy.lambdify(self.y, self.inv_generator.func)
        x = np.linspace(start, stop, 1000)
        y = [generator(i) for i in x]
        z = [inv_generator(i) for i in x]
        plt.plot(x, y, label="Generator $\\varphi$")
        plt.plot(x, z, label="Inverse generator $\psi$")
        title = self._get_copula_title()
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
        return
