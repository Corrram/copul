import copy
import inspect
import logging
import pathlib
import types

import numpy as np
import sympy
from matplotlib import pyplot as plt, rcParams

from copul.cd1_wrapper import CD1Wrapper
from copul.cd2_wrapper import CD2Wrapper
from copul.families.copula_graphs import CopulaGraphs
from copul.families.rank_correlation_plotter import RankCorrelationPlotter
from copul.families.cis_verifier import CISVerifier
from copul.families.copula_sampler import CopulaSampler
from copul.families.tp2_verifier import TP2Verifier
from copul.sympy_wrapper import SymPyFunctionWrapper

log = logging.getLogger(__name__)


class Copula:
    params = []
    u, v = sympy.symbols("u v", positive=True)
    intervals = {}
    log_cut_off = 4
    _package_path = pathlib.Path(__file__).parent.parent
    _cdf = None
    _free_symbols = {}

    def __init__(self, *args, **kwargs):
        if args and len(args) <= len(self.params):
            for i in range(len(args)):
                kwargs[str(self.params[i])] = args[i]
        self._are_class_vars(kwargs)
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(self, k, v)
        self.params = [param for param in self.params if str(param) not in kwargs]
        self.intervals = {
            k: v for k, v in self.intervals.items() if str(k) not in kwargs
        }

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        new_copula = copy.copy(self)
        new_copula._are_class_vars(kwargs)
        for i in range(len(args)):
            kwargs[str(self.params[i])] = args[i]
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(new_copula, k, v)
        new_copula.params = [param for param in self.params if str(param) not in kwargs]
        new_copula.intervals = {
            k: v for k, v in self.intervals.items() if str(k) not in kwargs
        }
        return new_copula

    @classmethod
    def _from_string(cls, free_symbols):
        obj = cls()
        obj._free_symbols = free_symbols  # Store free symbols for later use

        for key in free_symbols:
            setattr(obj, key, sympy.symbols(key, real=True))
            value = getattr(obj, key)
            obj.params.append(value)
            obj.intervals[value] = sympy.Interval(-np.inf, np.inf)
        return obj

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def parameters(self):
        return self.intervals

    def _set_params(self, args, kwargs):
        if args and len(args) <= len(self.params):
            for i in range(len(args)):
                kwargs[str(self.params[i])] = args[i]
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _are_class_vars(self, kwargs):
        class_vars = set(dir(self))
        assert set(kwargs).issubset(
            class_vars
        ), f"keys: {set(kwargs)}, free symbols: {class_vars}"

    def slice_interval(self, param, interval_start=None, interval_end=None):
        if not isinstance(param, str):
            param = str(param)
        left_open = self.intervals[param].left_open
        right_open = self.intervals[param].right_open
        if interval_start is None:
            interval_start = self.intervals[param].inf
        else:
            left_open = False
        if interval_end is None:
            interval_end = self.intervals[param].sup
        else:
            right_open = False
        self.intervals[param] = sympy.Interval(
            interval_start, interval_end, left_open, right_open
        )

    @property
    def cdf(self, u=None, v=None):
        expr = self._cdf
        for key, value in self._free_symbols.items():
            expr = expr.subs(value, getattr(self, key))
        return SymPyFunctionWrapper(expr)(u, v)

    @classmethod
    def from_cdf(cls, cdf, params=None):
        sp_cdf = sympy.sympify(cdf)
        func_vars, free_symbols = cls._segregate_symbols(sp_cdf, ["u", "v"], params)
        obj = cls._from_string(free_symbols)
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

    @property
    def is_absolutely_continuous(self) -> bool:
        raise NotImplementedError("This method must be implemented in a subclass")

    @property
    def is_symmetric(self) -> bool:
        raise NotImplementedError("This method must be implemented in a subclass")

    def rvs(self, n=1):
        """Sample a value from the copula"""

        cond_distr = self.cond_distr_2
        sig = inspect.signature(cond_distr)
        params = set(sig.parameters.keys()) & set(self.intervals)
        if params:
            func2_ = cond_distr
        else:
            func_ = cond_distr().func
            func2_ = sympy.lambdify([self.u, self.v], func_, ["numpy"])
        results = self._sample_val(func2_, n)
        return results

    def _sample_val(self, function, n=1):
        sampler = CopulaSampler(self)
        result = np.array([sampler.sample_val(function) for _ in range(n)])
        log.debug(sampler.err_counter)
        return result

    @property
    def pdf(self):
        result = sympy.simplify(sympy.diff(self.cond_distr_2().func, self.u))
        return SymPyFunctionWrapper(result)

    def cond_distr_1(self, u=None, v=None):
        result = CD1Wrapper(sympy.diff(self.cdf, self.u))
        return result(u, v)

    def cond_distr_2(self, u=None, v=None):
        result = CD2Wrapper(sympy.diff(self.cdf, self.v))
        return result(u, v)

    def chatterjees_xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        log.debug("xi")
        cond_distri_1 = sympy.simplify(self.cond_distr_1())
        log.debug("cond_distr_1 sympy: ", cond_distri_1)
        log.debug("cond_distr_1: ", sympy.latex(cond_distri_1))
        squared_cond_distr_1 = self._squared_cond_distr_1(self.u, self.v)
        log.debug("squared_cond_distr_1 sympy: ", squared_cond_distr_1)
        log.debug("squared_cond_distr_1: ", sympy.latex(squared_cond_distr_1))
        int_1 = self._xi_int_1(self.v)
        log.debug("int_1 sympy: ", int_1)
        log.debug("int_1: ", sympy.latex(int_1))
        int_2 = self._xi_int_2()
        log.debug("int_2 sympy: ", int_2)
        log.debug("int_2: ", sympy.latex(int_2))
        xi = self._xi()
        log.debug("xi sympy: ", xi)
        log.debug("xi: ", sympy.latex(xi))
        return SymPyFunctionWrapper(xi)

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # log.debug("rho")
        # if isinstance(self.cdf, SymPyFunctionWrapper):
        #     cdf = sympy.simplify(self.cdf.func)
        # else:
        #     cdf = self.cdf
        # log.debug("cdf sympy: ", cdf)
        # log.debug("cdf latex: ", sympy.latex(cdf))
        # int_1 = self._rho_int_1()
        # log.debug("int_1 sympy: ", int_1)
        # log.debug("int_1 latex: ", sympy.latex(int_1))
        rho = self._rho()
        log.debug("rho sympy: ", rho)
        log.debug("rho latex: ", sympy.latex(rho))
        return rho

    def _rho(self):
        return sympy.simplify(12 * self._rho_int_2() - 3)

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # log.debug("tau")
        # if isinstance(self.cdf, SymPyFunctionWrapper):
        #     integrand = self.cdf.func * self.pdf
        # else:
        #     integrand = self.cdf * self.pdf
        # log.debug("integrand sympy: ", integrand)
        # log.debug("integrand latex: ", sympy.latex(integrand))
        # int_1 = self._tau_int_1()
        # log.debug("int_1 sympy: ", int_1)
        # log.debug("int_1 latex: ", sympy.latex(int_1))
        # int_2 = self._tau_int_2()
        # log.debug("int_2 sympy: ", int_2)
        # log.debug("int_2 latex: ", sympy.latex(int_2))
        tau = self._tau()
        log.debug("tau sympy: ", tau)
        log.debug("tau latex: ", sympy.latex(tau))
        return tau

    def _tau(self):
        return 4 * self._tau_int_2() - 1

    def _xi(self):
        return sympy.simplify(6 * self._xi_int_2() - 2)

    def _xi_int_2(self):
        integrand = self._xi_int_1(self.v)
        return sympy.simplify(sympy.integrate(integrand, (self.v, 0, 1)))

    def _rho_int_2(self):
        return sympy.simplify(sympy.integrate(self._rho_int_1(), (self.v, 0, 1)))

    def _tau_int_2(self):
        return sympy.simplify(sympy.integrate(self._tau_int_1(), (self.v, 0, 1)))

    def _xi_int_1(self, v):
        squared_cond_distr_1 = self._squared_cond_distr_1(self.u, v)
        return sympy.simplify(sympy.integrate(squared_cond_distr_1, (self.u, 0, 1)))

    def _rho_int_1(self):
        return sympy.simplify(sympy.integrate(self.cdf.func, (self.u, 0, 1)))

    def _tau_int_1(self):
        return sympy.simplify(sympy.integrate(self.cdf.func * self.pdf, (self.u, 0, 1)))

    def _squared_cond_distr_1(self, u, v):
        return sympy.simplify(self.cond_distr_1().func ** 2)

    def plot(self, *args, **kwargs):
        if not args and not kwargs:
            return self.plot_cdf()
        for i, function in enumerate(args):
            if len(args) > 1:
                kwargs[f"Function {i + 1}"] = function
            else:
                kwargs[""] = function
        free_symbol_dict = {str(s): getattr(self, str(s)) for s in self.params}
        for function_name, function in kwargs.items():
            if function.__name__ in ["cond_distr_1", "cond_distr_2"]:
                try:
                    function = function()
                except TypeError:
                    pass
            if not free_symbol_dict:
                self._plot3d(function, title=f"{function_name}", zlabel="")
            elif len([*free_symbol_dict]) == 1:
                param_str = [*free_symbol_dict][0]
                param_ = free_symbol_dict[param_str]
                interval = self.intervals[str(param_)]
                lower_bound = float(max(-10, interval.left))
                if interval.left_open:
                    lower_bound += 0.01
                upper_bound = float(min(interval.right, 10))
                if interval.right_open:
                    upper_bound -= 0.01
                x = np.linspace(lower_bound, upper_bound, 100)
                y = np.array([function.subs(str(param_), x_i) for x_i in x])
                try:
                    plt.plot(x, y, label=f"{function_name}")
                except TypeError as e:
                    if "complex" not in str(e):
                        raise e
                    y_list = [
                        function.subs(str(param_), x_i).evalf().as_real_imag()[0]
                        for x_i in x
                    ]
                    y = np.array(y_list)
                    plt.plot(x, y, label=f"{function_name}")
        if free_symbol_dict:
            plt.legend()
            title = CopulaGraphs(self).get_copula_title()
            plt.title(f"{title} {', '.join([*kwargs])}")
            plt.grid(True)
            plt.show()
            plt.draw()
            plt.close()
            # pathlib.Path("images").mkdir(exist_ok=True)
            # fig1 = plt.gcf()
            # filepath = f"{self._package_path}/images/{self.__class__.__name__}.png"
            # fig1.savefig(filepath)

    def scatter_plot(self, n=1_000):
        data_ = self.rvs(n)
        plt.scatter(data_[:, 0], data_[:, 1], s=rcParams["lines.markersize"] ** 2)
        title = CopulaGraphs(self).get_copula_title()
        plt.title(title)
        plt.xlabel("u")
        plt.ylabel("v")
        plt.grid(True)
        plt.show()
        plt.close()
        # filepath = f"{self._package_path}/images/{type(self).__name__}_scatter.png"
        # plt.savefig(filepath)

    def plot_cdf(self, data=None, title=None, zlabel=None):
        if title is None:
            title = CopulaGraphs(self).get_copula_title()
        if zlabel is None:
            zlabel = ""
        if data is None:
            return self._plot3d(self.cdf, title=title, zlabel=zlabel, zlim=(0, 1))
        else:
            self._plot_cdf_from_data(data)

    @staticmethod
    def _plot_cdf_from_data(data):
        # Estimate the 2D histogram (which we'll use as a CDF)
        bins = [50, 50]  # Number of bins in each dimension
        hist, xedges, yedges = np.histogram2d(
            data[:, 0], data[:, 1], bins=bins, density=True
        )

        # Calculate the CDF from the histogram
        cdf = np.cumsum(np.cumsum(hist, axis=0), axis=1)
        cdf /= cdf[-1, -1]

        # Create a grid for plotting
        x, y = np.meshgrid(
            (xedges[1:] + xedges[:-1]) / 2, (yedges[1:] + yedges[:-1]) / 2
        )

        # Plot the 3D CDF
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, cdf, cmap="viridis")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("CDF")
        plt.show()

    def plot_rank_correlations(
        self,
        n_obs=10_000,
        n_params=20,
        plot_var=False,
        ylim=(-1, 1),
        params=None,
        log_cut_off=None,
    ):
        plotter = RankCorrelationPlotter(self, log_cut_off)
        plotter.plot_rank_correlations(n_obs, n_params, params, plot_var, ylim)

    def plot_pdf(self):
        free_symbol_dict = {str(s): getattr(self, str(s)) for s in self.params}
        pdf = self(**free_symbol_dict).pdf
        title = CopulaGraphs(self).get_copula_title()
        return self._plot3d(pdf, title=title, zlabel="PDF")

    def plot_cond_distr_1(self):
        free_symbol_dict = {str(s): getattr(self, str(s)) for s in self.params}
        cond_distr_1 = self(**free_symbol_dict).cond_distr_1
        title = CopulaGraphs(self).get_copula_title()
        return self._plot3d(
            cond_distr_1, title=title, zlabel="Conditional Distribution 1"
        )

    def plot_cond_distr_2(self):
        free_symbol_dict = {str(s): getattr(self, str(s)) for s in self.params}
        cond_distr_2 = self(**free_symbol_dict).cond_distr_2
        title = CopulaGraphs(self).get_copula_title()
        return self._plot3d(
            cond_distr_2, title=title, zlabel="Conditional Distribution 2"
        )

    def _plot3d(self, func, title, zlabel, zlim=None):
        try:
            parameters = inspect.signature(func).parameters
        except TypeError:
            pass
        else:
            if isinstance(func, types.MethodType) and len(parameters) == 0:
                func = func()
        if isinstance(func, SymPyFunctionWrapper):
            f = sympy.lambdify((self.u, self.v), func.func)
        elif isinstance(func, sympy.Expr):
            f = sympy.lambdify((self.u, self.v), func)
        else:
            f = func

        # Create a meshgrid
        x = np.linspace(0.01, 0.99, 100)
        y = np.linspace(0.01, 0.99, 100)
        # Compute Z values for each pair of (X, Y)
        Z = np.zeros((len(y), len(x)))  # Initialize a matrix for Z values
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = f(x[i], y[j])

        # Create a 3D plot
        X, Y = np.meshgrid(x, y)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap="viridis")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel(zlabel)
        if zlim is not None:
            ax.set_zlim(*zlim)
        plt.title(title)
        plt.show()

    def lambda_L(self):
        return sympy.limit(self.cdf(v=self.u).func / self.u, self.u, 0, dir="+")

    def lambda_U(self):
        expr = (1 - self.cdf(v=self.u).func) / (1 - self.u)
        return sympy.simplify(2 - sympy.limit(expr, self.u, 1, dir="-"))

    def is_tp2(self, range_min=None, range_max=None):
        return TP2Verifier(range_min, range_max).is_tp2(self)

    def is_cis(self, cond_distr=1):
        return CISVerifier(cond_distr).is_cis(self)
