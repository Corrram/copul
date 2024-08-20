import itertools
import warnings
import logging
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy
from sympy import Derivative, Subs, log

from copul.cdf_wrapper import CDFWrapper
from copul.families.abstract_copula import AbstractCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


plt.rc("text", usetex=True)  # Enable LaTeX rendering
plt.rc("font", size=12)  # You can adjust this value as needed

log_ = logging.getLogger(__name__)


class ExtremeValueCopula(AbstractCopula):
    _t_min = 0
    _t_max = 1
    t = sympy.symbols("t", positive=True)
    pickands = SymPyFunctionWrapper(sympy.Function("A")(t))
    intervals = None

    def deriv_pickand_at_0(self):
        diff = sympy.simplify(sympy.diff(self.pickands, self.t))
        diff_at_0 = sympy.limit(diff, self.t, 0)
        return diff_at_0

    def sample_parameters(self, n=1):
        return {
            k: list(np.random.uniform(max(-10, v.start), min(10, v.end), n))
            for k, v in self.intervals.items()
        }

    @property
    def is_ci(self):
        return True

    @property
    def cdf(self):
        """Cumulative distribution function of the copula"""
        cop = (self.u * self.v) ** self.pickands(
            sympy.ln(self.v) / sympy.ln(self.u * self.v)
        ).func
        cop = self._get_simplified_solution(cop)
        return CDFWrapper(cop)

    @property
    def pdf(self):
        """Probability density function of the copula"""
        _xi_1, u, v = sympy.symbols("_xi_1 u v")
        pickands = self.pickands.func
        t = self.t
        pdf = (
            (u * v) ** pickands.subs(t, log(v) / log(u * v))
            * (
                -(
                    (log(v) - log(u * v))
                    * Subs(
                        Derivative(pickands.subs(t, _xi_1), _xi_1),
                        _xi_1,
                        log(v) / log(u * v),
                    )
                    - pickands.subs(t, log(v) / log(u * v)) * log(u * v)
                )
                * (
                    pickands.subs(t, log(v) / log(u * v)) * log(u * v)
                    - log(v)
                    * Subs(
                        Derivative(pickands.subs(t, _xi_1), _xi_1),
                        _xi_1,
                        log(v) / log(u * v),
                    )
                )
                * log(u * v)
                + (log(v) - log(u * v))
                * log(v)
                * Subs(
                    Derivative(pickands.subs(t, _xi_1), (_xi_1, 2)),
                    _xi_1,
                    log(v) / log(u * v),
                )
            )
            / (u * v * log(u * v) ** 3)
        )
        pdf = self._get_simplified_solution(pdf)
        return SymPyFunctionWrapper(pdf)

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        integrand = self._rho_int_1()  # nelsen 5.15
        log_.debug(f"integrand: {integrand}")
        log_.debug(f"integrand latex: {sympy.latex(integrand)}")
        rho = self._rho()
        log_.debug(f"rho: {rho}")
        log_.debug(f"rho latex: {sympy.latex(rho)}")
        return rho

    def _rho_int_1(self):
        return sympy.simplify((self.pickands.func + 1) ** (-2))

    def _rho(self):
        return sympy.simplify(
            12 * sympy.integrate(self._rho_int_1(), (self.t, 0, 1)) - 3
        )

    def kendalls_tau(self, *args, **kwargs):  # nelsen 5.15
        self._set_params(args, kwargs)
        t = self.t
        diff2_pickands = sympy.diff(self.pickands, t, 2)
        integrand = t * (1 - t) / self.pickands.func * diff2_pickands.func
        integrand = sympy.simplify(integrand)
        log_.debug("integrand: ", integrand)
        log_.debug("integrand latex: ", sympy.latex(integrand))
        integral = sympy.integrate(integrand, (t, 0, 1))
        tau = sympy.simplify(integral)
        log_.debug("tau: ", tau)
        log_.debug("tau latex: ", sympy.latex(tau))
        return tau

    def minimize_func(self, sympy_func):
        parameters = self.intervals.keys()

        def func(x):
            x1_float, x2_float, y1_float, y2_float = x[:4]
            par_dict = dict(zip(parameters, x[4:]))
            return sympy_func.subs(
                {"x1": x1_float, "x2": x2_float, "y1": y1_float, "y2": y2_float}
                | par_dict
            ).evalf()

        b = [0, 1]
        bounds = [b, b, b, b]
        parameter_bounds = [
            [self.intervals[par].inf, self.intervals[par].sup] for par in parameters
        ]
        bounds += parameter_bounds
        start_parameters = [
            min(self.intervals[par].inf + 0.5, self.intervals[par].sup)
            for par in parameters
        ]
        i = 0
        x0 = None
        while i < 4:
            x0 = np.concatenate((np.random.rand(4), start_parameters))
            try:
                solution = scipy.optimize.minimize(func, x0, bounds=bounds)
                return solution, x0
            except TypeError:
                i += 1
                log_.debug(i)
                continue
        return None, x0

    @staticmethod
    def _get_function_graph(func, par):
        par_str = ", ".join(f"$\\{key}={value}$" for key, value in par.items())
        par_str = par_str.replace("oo", "\\infty")
        lambda_func = sympy.lambdify("t", func)
        x = np.linspace(0, 1, 1000)
        y = [lambda_func(i) for i in x]
        plt.plot(x, y, label=par_str)

    def plot_pickands(self, subs=None, **kwargs):
        if kwargs:
            subs = kwargs
        if subs is None:
            subs = {}
        subs = {
            getattr(self, k) if isinstance(k, str) else k: v for k, v in subs.items()
        }
        for key, value in subs.items():
            if not isinstance(value, list):
                subs[key] = [value]
        plot_vals = self._mix_params(subs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for plot_val in plot_vals:
                subs_dict = {str(k): v for k, v in plot_val.items()}
                pickands = self(**subs_dict).pickands
                self._get_function_graph(pickands.func, plot_val)

        @contextmanager
        def suppress_warnings():
            warnings.filterwarnings("ignore")
            yield
            warnings.filterwarnings("default")

        params = {param: getattr(self, param) for param in [*self.intervals]}
        defined_params = {
            k: v for k, v in params.items() if not isinstance(v, sympy.Symbol)
        }
        dict_str = ", ".join(
            f"\\{key}={value}" for key, value in defined_params.items()
        )
        x_label = f"$t$"
        plt.xlabel(x_label)

        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1.03)
        plt.title(f"{self.__class__.__name__}")
        plt.ylabel("$A(t)$")
        plt.legend()
        with suppress_warnings():
            plt.show()
        # filepath = f"{self._package_path}/images/{self.__class__.__name__}_pickand.png"
        # plt.savefig(filepath)

    @staticmethod
    def _mix_params(params):
        cross_prod_keys = [
            key
            for key, value in params.items()
            if isinstance(value, (str, list, property))
        ]
        values_to_cross_product = [
            val if isinstance(val, list) else [val] for val in params.values()
        ]
        cross_prod = list(itertools.product(*values_to_cross_product))
        return [
            dict(zip(cross_prod_keys, cross_prod[i])) for i in range(len(cross_prod))
        ]

    def minimize_func_empirically(self, func, parameters):
        b = [0.01, 0.99]
        bounds = [b, b, b, b]
        parameter_bounds = [
            [max(self.intervals[par].inf, -10), min(self.intervals[par].sup, 10)]
            for par in parameters
        ]
        bounds += parameter_bounds
        linspaces = [
            np.linspace(start=float(b[0]), stop=float(b[1]), num=5) for b in bounds
        ]
        meshgrid = np.meshgrid(*linspaces)
        func_vals = func(*meshgrid)
        return min(func_vals)

    @staticmethod
    def _get_simplified_solution(sol):
        simplified_sol = sympy.simplify(sol)
        if isinstance(simplified_sol, sympy.core.containers.Tuple):
            return simplified_sol[0]
        else:
            return simplified_sol.evalf()
