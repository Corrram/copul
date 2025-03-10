import inspect
import logging
import pathlib
import types

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib import rcParams

from copul.copula_sampler import CopulaSampler
from copul.families.cis_verifier import CISVerifier
from copul.families.copula_graphs import CopulaGraphs
from copul.families.core_copula import CoreCopula
from copul.families.rank_correlation_plotter import RankCorrelationPlotter
from copul.families.tp2_verifier import TP2Verifier
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

log = logging.getLogger(__name__)


class BivCopula(CoreCopula):
    u, v = sp.symbols("u v", positive=True)
    log_cut_off = 4
    _package_path = pathlib.Path(__file__).parent.parent

    def __init__(self, *args, **kwargs):
        super().__init__(2)
        self.u_symbols = [self.u, self.v]
        self.dimension = 2
        self._are_class_vars(kwargs)
        for i in range(len(args)):
            kwargs[str(self.params[i])] = args[i]
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

    @staticmethod
    def _segregate_symbols(expr, func_var_name=None, params=None):
        """
        Separate function variables from parameters in a sympy expression.

        Parameters:
        -----------
        expr : sympy expression
            The expression to analyze
        func_var_name : str, optional
            Name of the expected function variable (e.g., 't')
        params : list or None, optional
            List of explicitly provided parameter symbols

        Returns:
        --------
        tuple
            (function_variables, parameters)
        """
        # Get all symbols from the expression
        all_symbols = list(expr.free_symbols)

        if not all_symbols:
            return [], params or []

        # If parameters were explicitly provided
        if params is not None:
            # Ensure params is a list
            if not isinstance(params, list):
                params = [params]

            # Convert string parameters to sympy symbols if needed
            param_symbols = []
            for p in params:
                if isinstance(p, str):
                    param_symbols.append(sp.symbols(p, positive=True))
                else:
                    param_symbols.append(p)

            # Function variables are all symbols that are not parameters
            func_vars = [s for s in all_symbols if s not in param_symbols]
            return func_vars, param_symbols

        # If no parameters provided, try to identify based on function_var_name
        if func_var_name:
            # Find the function variable by name
            func_vars = [s for s in all_symbols if str(s) == func_var_name]
            if not func_vars and all_symbols:
                # If no exact match found, try case-insensitive match
                func_vars = [
                    s for s in all_symbols if str(s).lower() == func_var_name.lower()
                ]

            # All other symbols are parameters
            params = [s for s in all_symbols if s not in func_vars]
            return func_vars, params

        # If we can't determine, assume first symbol is function variable, rest are parameters
        return [all_symbols[0]], all_symbols[1:]

    @classmethod
    def _from_string(cls, params=None):
        """
        Create a new instance with parameters defined by string names.

        Parameters:
        -----------
        params : list of sympy symbols or strings
            The parameters for the copula

        Returns:
        --------
        A new instance with parameters set as symbolic variables
        """
        obj = cls()

        # Ensure params is iterable
        if params is None:
            return obj

        # Convert params to list if it's a single item
        if not isinstance(params, list):
            params = [params]

        # Initialize or reset parameters
        obj.params = []

        # Process each parameter
        for param in params:
            # Convert string to symbol if needed
            if isinstance(param, str):
                param = sp.symbols(param, positive=True)

            # Add to params list
            obj.params.append(param)

            # Add as attribute
            param_name = str(param)
            setattr(obj, param_name, param)

            # Initialize _free_symbols if needed
            if not hasattr(obj, "_free_symbols"):
                obj._free_symbols = {}

            # Add to free_symbols dictionary
            obj._free_symbols[param_name] = param

        return obj

    def rvs(self, n=1, random_state=None):
        """Sample a value from the copula"""
        return CopulaSampler(self, random_state=random_state).rvs(n)

    @property
    def pdf(self):
        result = sp.simplify(sp.diff(self.cond_distr_2().func, self.u))
        return SymPyFuncWrapper(result)

    def cond_distr_1(self, u=None, v=None):
        result = CD1Wrapper(sp.diff(self.cdf, self.u))
        return result(u, v)

    def cond_distr_2(self, u=None, v=None):
        result = CD2Wrapper(sp.diff(self.cdf, self.v))
        return result(u, v)

    def chatterjees_xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        log.debug("xi")
        cond_distri_1 = sp.simplify(self.cond_distr_1())
        log.debug("cond_distr_1 sympy: ", cond_distri_1)
        log.debug("cond_distr_1: ", sp.latex(cond_distri_1))
        squared_cond_distr_1 = self._squared_cond_distr_1(self.u, self.v)
        log.debug("squared_cond_distr_1 sympy: ", squared_cond_distr_1)
        log.debug("squared_cond_distr_1: ", sp.latex(squared_cond_distr_1))
        int_1 = self._xi_int_1(self.v)
        log.debug("int_1 sympy: ", int_1)
        log.debug("int_1: ", sp.latex(int_1))
        int_2 = self._xi_int_2()
        log.debug("int_2 sympy: ", int_2)
        log.debug("int_2: ", sp.latex(int_2))
        xi = self._xi()
        log.debug("xi sympy: ", xi)
        log.debug("xi: ", sp.latex(xi))
        return SymPyFuncWrapper(xi)

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # log.debug("rho")
        # if isinstance(self.cdf, SymPyFunctionWrapper):
        #     cdf = sp.simplify(self.cdf.func)
        # else:
        #     cdf = self.cdf
        # log.debug("cdf sympy: ", cdf)
        # log.debug("cdf latex: ", sp.latex(cdf))
        # int_1 = self._rho_int_1()
        # log.debug("int_1 sympy: ", int_1)
        # log.debug("int_1 latex: ", sp.latex(int_1))
        rho = self._rho()
        log.debug("rho sympy: ", rho)
        log.debug("rho latex: ", sp.latex(rho))
        return rho

    def _rho(self):
        return sp.simplify(12 * self._rho_int_2() - 3)

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # log.debug("tau")
        # if isinstance(self.cdf, SymPyFunctionWrapper):
        #     integrand = self.cdf.func * self.pdf
        # else:
        #     integrand = self.cdf * self.pdf
        # log.debug("integrand sympy: ", integrand)
        # log.debug("integrand latex: ", sp.latex(integrand))
        # int_1 = self._tau_int_1()
        # log.debug("int_1 sympy: ", int_1)
        # log.debug("int_1 latex: ", sp.latex(int_1))
        # int_2 = self._tau_int_2()
        # log.debug("int_2 sympy: ", int_2)
        # log.debug("int_2 latex: ", sp.latex(int_2))
        tau = self._tau()
        log.debug("tau sympy: ", tau)
        log.debug("tau latex: ", sp.latex(tau))
        return tau

    def _tau(self):
        return 4 * self._tau_int_2() - 1

    def _xi(self):
        return sp.simplify(6 * self._xi_int_2() - 2)

    def _xi_int_2(self):
        integrand = self._xi_int_1(self.v)
        return sp.simplify(sp.integrate(integrand, (self.v, 0, 1)))

    def _rho_int_2(self):
        return sp.simplify(sp.integrate(self._rho_int_1(), (self.v, 0, 1)))

    def _tau_int_2(self):
        return sp.simplify(sp.integrate(self._tau_int_1(), (self.v, 0, 1)))

    def _xi_int_1(self, v):
        squared_cond_distr_1 = self._squared_cond_distr_1(self.u, v)
        return sp.simplify(sp.integrate(squared_cond_distr_1, (self.u, 0, 1)))

    def _rho_int_1(self):
        return sp.simplify(sp.integrate(self.cdf.func, (self.u, 0, 1)))

    def _tau_int_1(self):
        return sp.simplify(sp.integrate(self.cdf.func * self.pdf, (self.u, 0, 1)))

    def _squared_cond_distr_1(self, u, v):
        return sp.simplify(self.cond_distr_1().func ** 2)

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
        if isinstance(func, SymPyFuncWrapper):
            f = sp.lambdify((self.u, self.v), func.func)
        elif isinstance(func, sp.Expr):
            f = sp.lambdify((self.u, self.v), func)
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
        return sp.limit(self.cdf(v=self.u).func / self.u, self.u, 0, dir="+")

    def lambda_U(self):
        expr = (1 - self.cdf(v=self.u).func) / (1 - self.u)
        return sp.simplify(2 - sp.limit(expr, self.u, 1, dir="-"))

    def is_tp2(self, range_min=None, range_max=None):
        return TP2Verifier(range_min, range_max).is_tp2(self)

    def is_cis(self, cond_distr=1):
        return CISVerifier(cond_distr).is_cis(self)
