import inspect
import logging
import pathlib
import types

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from copul.numerics import to_numpy_callable
from copul.schur_order.cis_verifier import CISVerifier
from copul.family.copula_graphs import CopulaGraphs
from copul.family.rank_correlation_plotter import RankCorrelationPlotter
from copul.family.tp2_verifier import TP2Verifier
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdi_wrapper import CDiWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

log = logging.getLogger(__name__)


class BivCoreCopula:
    """
    Base class for bivariate copulas using symbolic expressions.

    This class extends CoreCopula for the bivariate (2-dimensional) case.
    It provides functionality for sampling, evaluation of the PDF, conditional
    distributions, and various dependence measures such as Chatterjee's xi,
    Spearman's rho, and Kendall's tau. Additionally, plotting utilities are
    included for visualizing the copula's functions.
    """

    u, v = sp.symbols("u v", positive=True)
    log_cut_off = 4
    _package_path = pathlib.Path(__file__).parent.parent
    params: list = []
    intervals: dict = {}

    def __init__(self):
        """
        Initialize a bivariate copula instance.

        This constructor sets up the copula in 2 dimensions, processes the
        provided parameters (both positional and keyword), and initializes the
        free parameters and their intervals.

        Parameters
        ----------
        *args : tuple
            Positional parameters corresponding to copula parameters.
        **kwargs : dict
            Keyword parameters to override default symbolic parameters.
        """
        self.dim = 2
        self.u_symbols = [self.u, self.v]

    def __str__(self):
        """
        Return a string representation of the bivariate copula.

        Returns
        -------
        str
            The class name of the copula.
        """
        return self.__class__.__name__

    @staticmethod
    def _segregate_symbols(expr, func_var_name=None, params=None):
        """
        Separate function variables from parameter symbols in an expression.

        Parameters
        ----------
        expr : sympy.Expr
            The symbolic expression to analyze.
        func_var_name : str, optional
            The expected name of the function variable (e.g., 't').
        params : list or None, optional
            A list of explicitly provided parameter symbols or names.

        Returns
        -------
        tuple
            A tuple (function_variables, parameters) where:
            - function_variables is a list of sympy symbols not in params.
            - parameters is a list of sympy symbols representing the parameters.
        """
        all_symbols = list(expr.free_symbols)

        if not all_symbols:
            return [], params or []

        if params is not None:
            if not isinstance(params, list):
                params = [params]
            param_symbols = []
            for p in params:
                if isinstance(p, str):
                    param_symbols.append(sp.symbols(p, positive=True))
                else:
                    param_symbols.append(p)
            func_vars = [s for s in all_symbols if s not in param_symbols]
            return func_vars, param_symbols

        if func_var_name:
            func_vars = [s for s in all_symbols if str(s) == func_var_name]
            if not func_vars and all_symbols:
                func_vars = [
                    s for s in all_symbols if str(s).lower() == func_var_name.lower()
                ]
            params = [s for s in all_symbols if s not in func_vars]
            return func_vars, params

        return [all_symbols[0]], all_symbols[1:]

    @classmethod
    def _from_string(cls, params=None):
        """
        Construct a new copula instance with parameters defined by strings.

        This factory method creates an instance of the copula with symbolic
        parameters corresponding to the given names.

        Parameters
        ----------
        params : list of sympy.Symbol or str, optional
            The parameter names or symbols to be used for the copula.

        Returns
        -------
        BivCopula
            A new instance with the specified symbolic parameters.
        """
        obj = cls()
        if params is None:
            return obj
        if not isinstance(params, list):
            params = [params]
        obj.params = []
        for param in params:
            if isinstance(param, str):
                param = sp.symbols(param, positive=True)
            obj.params.append(param)
            param_name = str(param)
            setattr(obj, param_name, param)
            if not hasattr(obj, "_free_symbols"):
                obj._free_symbols = {}
            obj._free_symbols[param_name] = param
        return obj

    @property
    def pdf(self):
        """
        Evaluate the probability density function (PDF) of the copula.

        The PDF is computed by differentiating the conditional distribution.

        Returns
        -------
        SymPyFuncWrapper
            A wrapper around the simplified symbolic expression for the PDF.
        """
        result = sp.simplify(sp.diff(self.cond_distr_2().func, self.u))
        return SymPyFuncWrapper(result)

    def cond_distr_1(self, u=None, v=None):
        """
        Compute the first conditional distribution F_{U_{-1}|U_1}(u_{-1}|u_1).

        This method differentiates the CDF with respect to the first variable.

        Parameters
        ----------
        u : optional
            The u-coordinate; if provided, passed to the wrapper.
        v : optional
            The v-coordinate; if provided, passed to the wrapper.

        Returns
        -------
        callable
            A callable (wrapped via CD1Wrapper) representing the conditional distribution.
        """
        result = self.cdf().diff(self.u)
        return result(u, v)

    def cond_distr_2(self, u=None, v=None):
        """
        Compute the second conditional distribution F_{U_{-2}|U_2}(u_{-2}|u_2).

        This method differentiates the CDF with respect to the second variable.

        Parameters
        ----------
        u : optional
            The u-coordinate; if provided, passed to the wrapper.
        v : optional
            The v-coordinate; if provided, passed to the wrapper.

        Returns
        -------
        callable
            A callable (wrapped via CD2Wrapper) representing the conditional distribution.
        """
        result = CD2Wrapper(sp.diff(self.cdf().func, self.v))
        return result(u, v)

    def chatterjees_xi(self, *args, **kwargs):
        """
        Compute Chatterjee's xi correlation measure.

        This method sets the parameters, computes intermediate integrals,
        and returns the simplified expression for xi.

        Returns
        -------
        SymPyFuncWrapper
            A wrapper around the symbolic expression for Chatterjee's xi.
        """
        self._set_params(args, kwargs)
        log.debug("xi")
        cond_distri_1 = sp.simplify(self.cond_distr_1())
        log.debug("cond_distr_1 sympy: %s", cond_distri_1)
        log.debug("cond_distr_1 latex: %s", sp.latex(cond_distri_1))
        squared_cond_distr_1 = self._squared_cond_distr_1(self.u, self.v)
        log.debug("squared_cond_distr_1 sympy: %s", squared_cond_distr_1)
        log.debug("squared_cond_distr_1 latex: %s", sp.latex(squared_cond_distr_1))
        int_1 = self._xi_int_1(self.v)
        log.debug("int_1 sympy: %s", int_1)
        log.debug("int_1 latex: %s", sp.latex(int_1))
        int_2 = self._xi_int_2()
        log.debug("int_2 sympy: %s", int_2)
        log.debug("int_2 latex: %s", sp.latex(int_2))
        xi = self._xi()
        log.debug("xi sympy: %s", xi)
        log.debug("xi latex: %s", sp.latex(xi))
        return SymPyFuncWrapper(xi)

    def spearmans_rho(self, *args, **kwargs):
        """
        Compute Spearman's rho correlation measure.

        This method sets the parameters, computes the corresponding integral,
        and returns the simplified expression for Spearman's rho.

        Returns
        -------
        sympy.Expr
            The symbolic expression for Spearman's rho.
        """
        self._set_params(args, kwargs)
        rho = self._rho()
        log.debug("rho sympy: %s", rho)
        log.debug("rho latex: %s", sp.latex(rho))
        return rho

    def _rho(self):
        """
        Internal method to compute Spearman's rho.

        Returns
        -------
        sympy.Expr
            The simplified symbolic expression for Spearman's rho.
        """
        return sp.simplify(12 * self._rho_int_2() - 3)

    def kendalls_tau(self, *args, **kwargs):
        """
        Compute Kendall's tau correlation measure.

        This method sets the parameters, computes necessary integrals,
        and returns the simplified expression for Kendall's tau.

        Returns
        -------
        sympy.Expr
            The symbolic expression for Kendall's tau.
        """
        self._set_params(args, kwargs)
        tau = self._tau()
        log.debug("tau sympy: %s", tau)
        log.debug("tau latex: %s", sp.latex(tau))
        return tau

    def blests_nu(self, *args, **kwargs):
        """
        Compute Blest's rank correlation ν.

        Uses the copula form
            ν(C) = 24 ∫_0^1 ∫_0^1 (1 - u) C(u, v) du dv - 2
        which is linear in C and generally symbolic-friendly.

        Returns
        -------
        sympy.Expr
            The symbolic expression for Blest's ν.
        """
        self._set_params(args, kwargs)
        nu = self._nu()
        log.debug("nu sympy: %s", nu)
        log.debug("nu latex: %s", sp.latex(nu))
        return nu

    def _nu(self):
        """
        Internal method to compute Blest's ν via C(u,v).
        """
        return sp.simplify(24 * self._nu_int_2() - 2)

    def _nu_int_2(self):
        """
        Outer integral over v for Blest's ν.
            ∫_0^1 [ ∫_0^1 (1 - u) C(u, v) du ] dv
        """
        return sp.simplify(sp.integrate(self._nu_int_1(), (self.v, 0, 1)))

    def _nu_int_1(self):
        """
        Inner integral over u for Blest's ν.
            ∫_0^1 (1 - u) C(u, v) du
        """
        return sp.simplify(sp.integrate((1 - self.u) * self.cdf().func, (self.u, 0, 1)))

    def _tau(self):
        """
        Internal method to compute Kendall's tau.

        Returns
        -------
        sympy.Expr
            The simplified symbolic expression for Kendall's tau.
        """
        return 4 * self._tau_int_2() - 1

    def _xi(self):
        """
        Internal method to compute Chatterjee's xi.

        Returns
        -------
        sympy.Expr
            The simplified symbolic expression for xi.
        """
        return sp.simplify(6 * self._xi_int_2() - 2)

    def _xi_int_2(self):
        """
        Compute the inner integral for xi over variable v.

        Returns
        -------
        sympy.Expr
            The simplified result of integrating the inner integrand with respect to v.
        """
        integrand = self._xi_int_1(self.v)
        return sp.simplify(sp.integrate(integrand, (self.v, 0, 1)))

    def _rho_int_2(self):
        """
        Compute the inner integral for Spearman's rho over variable v.

        Returns
        -------
        sympy.Expr
            The simplified result of integrating the inner integrand with respect to v.
        """
        return sp.simplify(sp.integrate(self._rho_int_1(), (self.v, 0, 1)))

    def _tau_int_2(self):
        """
        Compute the inner integral for Kendall's tau over variable v.

        Returns
        -------
        sympy.Expr
            The simplified result of integrating the inner integrand with respect to v.
        """
        return sp.simplify(sp.integrate(self._tau_int_1(), (self.v, 0, 1)))

    def _xi_int_1(self, v):
        """
        Compute the integrand for xi with respect to variable u.

        Parameters
        ----------
        v : sympy.Symbol or numeric
            The variable for the second dimension.

        Returns
        -------
        sympy.Expr
            The simplified expression representing the integrand.
        """
        squared_cond_distr_1 = self._squared_cond_distr_1(self.u, v)
        return sp.simplify(sp.integrate(squared_cond_distr_1, (self.u, 0, 1)))

    def _rho_int_1(self):
        """
        Compute the integrand for Spearman's rho with respect to variable u.

        Returns
        -------
        sympy.Expr
            The simplified expression of the integrand.
        """
        return sp.simplify(sp.integrate(self.cdf().func, (self.u, 0, 1)))

    def _tau_int_1(self):
        """
        Compute the integrand for Kendall's tau with respect to variable u.

        Returns
        -------
        sympy.Expr
            The simplified expression of the integrand.
        """
        return sp.simplify(sp.integrate(self.cdf().func * self.pdf, (self.u, 0, 1)))

    def _squared_cond_distr_1(self, u, v):
        """
        Compute the square of the first conditional distribution.

        Parameters
        ----------
        u : sympy.Symbol or numeric
            The first variable.
        v : sympy.Symbol or numeric
            The second variable.

        Returns
        -------
        sympy.Expr
            The simplified squared expression of the first conditional distribution.
        """
        return sp.simplify(self.cond_distr_1().func ** 2)

    def plot(self, *args, **kwargs):
        """
        Plot one or more functions related to the copula.

        If no functions are provided as arguments, the CDF is plotted by default.
        Additional functions can be passed as positional arguments and are labeled
        automatically in the legend.

        Parameters
        ----------
        *args
            One or more functions to be plotted.
        **kwargs
            Additional keyword arguments where keys become labels for the functions.

        Notes
        -----
        The function uses the free symbolic parameters to determine plotting ranges.
        """
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
            elif len(free_symbol_dict) == 1:
                param_str = list(free_symbol_dict.keys())[0]
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
            plt.title(f"{title} {', '.join(list(kwargs.keys()))}")
            plt.grid(True)
            plt.show()
            plt.draw()
            plt.close()

    @staticmethod
    def _plot_cdf_from_data(data):
        """
        Plot a 3D surface of the CDF computed from given data.

        This function calculates a 2D histogram from the data, computes the cumulative
        sum to obtain the CDF, and then creates a 3D surface plot.

        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n, 2) used to estimate the CDF.
        """
        bins = [50, 50]
        hist, xedges, yedges = np.histogram2d(
            data[:, 0], data[:, 1], bins=bins, density=True
        )
        cdf = np.cumsum(np.cumsum(hist, axis=0), axis=1)
        cdf /= cdf[-1, -1]
        x, y = np.meshgrid(
            (xedges[1:] + xedges[:-1]) / 2, (yedges[1:] + yedges[:-1]) / 2
        )
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
        approximate=False,
    ):
        """
        Plot rank correlations for the copula.

        This method creates a RankCorrelationPlotter instance for the current copula,
        and uses it to plot various rank correlation measures (e.g. Chatterjee's xi).
        Optionally, variance bands can be plotted along with the main correlation curve.

        Parameters
        ----------
        n_obs : int, optional
            Number of observations to simulate per evaluation point (default is 10,000).
        n_params : int, optional
            Number of parameter values to evaluate (default is 20).
        plot_var : bool, optional
            Whether to plot variance bands in the correlation plot (default is False).
        ylim : tuple of float, optional
            Y-axis limits for the plot (default is (-1, 1)).
        params : dict or None, optional
            Dictionary of additional parameters for mixed-parameter plotting.
            If None, only the main parameter is used (default is None).
        log_cut_off : int, float, tuple, or None, optional
            Cut-off value(s) for applying a logarithmic scale to the x-axis.
            If provided, the plot will use a log scale (default is None).
        approximate : bool, optional
            Whether to use approximate sampling via checkerboard copulas.

        Returns
        -------
        None
            This method displays the plot using matplotlib.
        """
        plotter = RankCorrelationPlotter(self, log_cut_off, approximate)
        plotter.plot_rank_correlations(n_obs, n_params, params, ylim)

    def plot_cdf(
        self,
        data=None,
        title=None,
        zlabel=None,
        *,
        plot_type="3d",
        log_z=False,
        **kwargs,
    ) -> plt.Figure:
        """Plot the cumulative distribution function (CDF).

        Parameters
        ----------
        plot_type : {"3d", "contour"}
            Choose visualisation style.
        log_z : bool, optional (contour only)
            Use logarithmic colour scaling (``matplotlib.colors.LogNorm``).
        """
        if plot_type not in {"3d", "contour"}:
            raise ValueError("plot_type must be '3d' or 'contour'")

        if title is None:
            title = CopulaGraphs(self).get_copula_title()
        if zlabel is None:
            zlabel = ""

        if data is not None and plot_type == "contour":
            raise ValueError(
                "Contour plots from external data are not supported – set plot_type='3d' or omit *data*."
            )

        if data is None:
            if plot_type == "3d":
                return self._plot3d(self.cdf, title=title, zlabel=zlabel, zlim=(0, 1))
            return self._plot_contour(
                self.cdf, title=title, zlabel=zlabel, zlim=(0, 1), log_z=log_z, **kwargs
            )
        return self._plot_cdf_from_data(data)

    def plot_pdf(self, *args, plot_type="3d", log_z=False, **kwargs) -> plt.Figure:
        """Plot the probability density function (PDF).

        Parameters
        ----------
        plot_type : {"3d", "contour"}
        log_z : bool, optional (contour only)
        """
        if plot_type not in {"3d", "contour"}:
            raise ValueError("plot_type must be '3d' or 'contour'")

        free_symbol_dict = {str(s): getattr(self, str(s)) for s in self.params}
        pdf = self(**free_symbol_dict).pdf if free_symbol_dict else self.pdf
        title = kwargs.pop("title", CopulaGraphs(self).get_copula_title())

        if plot_type == "3d":
            return self._plot3d(pdf, title=title, zlabel="PDF", **kwargs)
        return self._plot_contour(pdf, title=title, zlabel="PDF", log_z=log_z, **kwargs)

    def plot_cond_distr_1(self, *, plot_type="3d", log_z=False, **kwargs) -> plt.Figure:
        if plot_type not in {"3d", "contour", "functions"}:
            raise ValueError("plot_type must be '3d', 'contour', or 'functions'")
        cond_distr_1 = self.cond_distr_1
        if "title" not in kwargs:
            title = CopulaGraphs(self).get_copula_title()
        else:
            title = kwargs.pop("title")
        if "zlabel" not in kwargs:
            zlabel = "Conditional Distribution 1"
        else:
            zlabel = kwargs.pop("zlabel")
        if "xlabel" in kwargs:
            xlabel = kwargs.pop("xlabel")
        else:
            xlabel = "u"

        if plot_type == "3d":
            return self._plot3d(cond_distr_1, title=title, zlabel=zlabel)
        elif plot_type == "functions":
            return self._plot_functions(
                cond_distr_1, title=title, zlabel=zlabel, xlabel=xlabel, **kwargs
            )
        return self._plot_contour(
            cond_distr_1, title=title, zlabel=zlabel, log_z=log_z, **kwargs
        )

    def plot_cond_distr_2(self, *, plot_type="3d", log_z=False, **kwargs):
        if plot_type not in {"3d", "contour"}:
            raise ValueError("plot_type must be '3d' or 'contour'")
        cond_distr_2 = self.cond_distr_2
        title = CopulaGraphs(self).get_copula_title()
        if plot_type == "3d":
            return self._plot3d(
                cond_distr_2, title=title, zlabel="Conditional Distribution 2"
            )
        return self._plot_contour(
            cond_distr_2,
            title=title,
            zlabel="Conditional Distribution 2",
            log_z=log_z,
            **kwargs,
        )

    def _plot3d(self, func, title, zlabel, zlim=None) -> plt.Figure:
        """
        Generate a 3D plot of a given function of two variables.

        The function can be a SymPy expression wrapped in a SymPyFuncWrapper,
        or any callable that accepts (u, v).
        """
        intervals = {k: v for k, v in self.intervals.items()}

        try:
            _parameters = inspect.signature(func).parameters
        except TypeError:
            pass
        else:
            if isinstance(func, types.MethodType):
                try:
                    func = func()
                except (ValueError, TypeError):
                    pass

        if isinstance(func, (SymPyFuncWrapper, CD1Wrapper, CD2Wrapper, CDiWrapper)):
            f = to_numpy_callable(func.func, (self.u, self.v), ae=True)
        elif isinstance(func, sp.Expr):
            f = to_numpy_callable(func, (self.u, self.v), ae=True)
        else:
            f = func

        x = np.linspace(0.01, 0.99, 100)
        y = np.linspace(0.01, 0.99, 100)
        Z = np.zeros((len(y), len(x)))
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = f(x[i], y[j])
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel(zlabel)
        if zlim is not None:
            ax.set_zlim(*zlim)
        plt.title(title)
        plt.show()
        self.intervals = intervals
        return fig

    def _plot_functions(self, func, title, zlabel, xlabel="u", **kwargs) -> plt.Figure:
        """
        Evaluate the given bivariate function at v = 0.1, 0.2, …, 0.9 and
        draw the resulting 1-D curves in one (square) figure.
        """
        # -------- resolve `func` exactly as in the original -----------------
        try:
            inspect.signature(func).parameters
            if isinstance(func, types.MethodType):
                func = func()
        except (ValueError, TypeError):
            pass

        if isinstance(func, (SymPyFuncWrapper, CD1Wrapper, CD2Wrapper, CDiWrapper)):
            f = to_numpy_callable(func.func, (self.u, self.v), ae=True)
        elif isinstance(func, sp.Expr):
            f = to_numpy_callable(func, (self.u, self.v), ae=True)
        else:
            f = func
        # -------------------------------------------------------------------
        u_vals = np.linspace(0.01, 0.99, 200)  # x grid
        v_vals = np.linspace(0.1, 0.9, 9)  # the fixed v’s

        # --- make a square figure and explicit axes ------------------------
        fig, ax = plt.subplots(figsize=(6, 6))  # width = height

        for v_i in v_vals:
            try:
                y_vals = f(u_vals, v_i)
            except Exception:  # scalar-only function
                y_vals = np.array([f(u, v_i) for u in u_vals])
            ax.plot(u_vals, y_vals, label=f"$v = {v_i:.1f}$", linewidth=2.5, **kwargs)

        # labels, grid, title ------------------------------------------------
        ax.set_xlabel(xlabel)
        if zlabel is not None:
            ax.set_ylabel(zlabel)
        if title is not None:
            ax.set_title(f"{title} — {zlabel}")

        ax.grid(True)
        # legend *outside* the axes to keep the square uncluttered
        leg = ax.legend(loc="upper right", frameon=True, fontsize=10)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.8)

        fig.tight_layout()  # respect the outside legend
        plt.show()
        return fig

    def _plot_contour(
        self, func, title, zlabel, *, levels=100, zlim=None, log_z=False, **kwargs
    ) -> plt.Figure:
        r"""Create a filled contour plot.

        If ``log_z`` is *True*, the colour map is based on log1p(Z) etc.
        """
        intervals_backup = dict(self.intervals)

        try:
            _ = inspect.signature(func).parameters
            if isinstance(func, types.MethodType):
                func = func()
        except (TypeError, ValueError):
            pass

        if isinstance(func, (SymPyFuncWrapper, CD1Wrapper, CD2Wrapper, CDiWrapper)):
            f = to_numpy_callable(func.func, (self.u, self.v), ae=True)
        elif isinstance(func, sp.Expr):
            f = to_numpy_callable(func, (self.u, self.v), ae=True)
        else:
            f = func

        grid_size = kwargs.pop("grid_size", None)
        if grid_size is None:
            grid_size = 2 * levels
        x = np.linspace(0.005, 0.995, grid_size)
        y = np.linspace(0.005, 0.995, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(f)(X, Y)

        if zlim is not None:
            Z = np.clip(Z, zlim[0], zlim[1])

        cmap = kwargs.pop("cmap", "viridis")
        cmap = plt.cm.get_cmap(cmap).copy()
        fig, ax = plt.subplots()

        if log_z:
            Z_mask = np.ma.masked_less(Z, 0.0)
            Z_for_norm = np.ma.masked_less(Z_mask + 1.0, 1e-12)
            vmin = (zlim[0] + 1) if zlim else Z_for_norm.min()
            vmax = (zlim[1] + 1) if zlim else Z_for_norm.max()
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            if isinstance(levels, int):
                levels = np.geomspace(vmin, vmax, levels * 5)
            cf = ax.contourf(X, Y, Z_for_norm, levels=levels, cmap=cmap, norm=norm)
            cbar = fig.colorbar(cf, ax=ax)
            ticks = cbar.get_ticks()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t - 1:g}" for t in ticks])
        else:
            if isinstance(levels, int):
                levels = levels
            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
            cbar = fig.colorbar(cf, ax=ax)

        cbar.set_label(zlabel)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(title)
        fig.tight_layout()
        plt.show()
        self.intervals = intervals_backup
        return fig

    def lambda_L(self):
        """
        Compute the lower tail dependence coefficient.

        Returns
        -------
        sympy.Expr
            The symbolic expression for the lower tail dependence.
        """
        return sp.limit(self.cdf(v=self.u).func / self.u, self.u, 0, dir="+")

    def lambda_U(self):
        """
        Compute the upper tail dependence coefficient.

        Returns
        -------
        sympy.Expr
            The simplified symbolic expression for the upper tail dependence.
        """
        expr = (1 - self.cdf(v=self.u).func) / (1 - self.u)
        return sp.simplify(2 - sp.limit(expr, self.u, 1, dir="-"))

    def is_tp2(self, range_min=None, range_max=None):
        """
        Check if the copula satisfies the TP2 (Total Positivity of order 2) property.

        Parameters
        ----------
        range_min : numeric, optional
            Minimum value of the range for testing (default is None).
        range_max : numeric, optional
            Maximum value of the range for testing (default is None).

        Returns
        -------
        bool
            True if the copula is TP2, False otherwise.
        """
        return TP2Verifier(range_min, range_max).is_tp2(self)

    def is_cis(self, cond_distr=1):
        """
        Check if the copula satisfies the CIS (Conditional Increasing in Sequence) property.

        Parameters
        ----------
        cond_distr : int, optional
            Specifies which conditional distribution to use (default is 1).

        Returns
        -------
        bool
            True if the copula is CIS, False otherwise.
        """
        return CISVerifier(cond_distr).is_cis(self)

    def blomqvists_beta(self) -> float:
        """
        Blomqvist’s β   :=  4·C(½,½) – 1
        """
        return 4.0 * self.cdf(u=0.5, v=0.5) - 1.0

    # ------------------------------------------------------------------
    # Gini’s gamma  γ(C) = 4[∫₀¹ C(t,t) dt + ∫₀¹ C(t,1-t) dt] − 2
    # ------------------------------------------------------------------

    def gini_gamma(self, *args, **kwargs):
        r"""
        Compute Gini’s gamma concordance coefficient.

        .. math::

           \gamma(C) = 4\!\left[\int_0^1 C(t,t)\,dt
                        + \int_0^1 C(t,1-t)\,dt\right] - 2

        Range: :math:`[-1, 1]`.
        :math:`\gamma = 1` for the upper Fréchet bound (comonotonic),
        :math:`\gamma = -1` for the lower Fréchet bound (countermonotonic),
        :math:`\gamma = 0` for independence.

        Returns
        -------
        sympy.Expr
            The symbolic expression for Gini’s gamma.
        """
        self._set_params(args, kwargs)
        return self._gini_gamma()

    def _gini_gamma(self):
        t = sp.Symbol("t", positive=True)
        cdf_expr = self.cdf().func
        cdf_diag = cdf_expr.subs([(self.u, t), (self.v, t)])
        cdf_anti = cdf_expr.subs([(self.u, t), (self.v, 1 - t)])
        int1 = sp.simplify(sp.integrate(cdf_diag, (t, 0, 1)))
        int2 = sp.simplify(sp.integrate(cdf_anti, (t, 0, 1)))
        return sp.simplify(4 * (int1 + int2) - 2)

    # ------------------------------------------------------------------
    # Spearman’s footrule  ψ(C) = 6·∫₀¹ C(t,t) dt − 2
    # ------------------------------------------------------------------

    def spearman_footrule(self, *args, **kwargs):
        r"""
        Compute Spearman’s footrule coefficient :math:`\psi`.

        .. math::

           \psi(C) = 6 \int_0^1 C(t,t)\,dt - 2

        Equivalently, :math:`\psi(C) = 6\,\mathbb{E}[\min(U,V)] - 2`
        where :math:`(U,V) \sim C`.

        Range: :math:`[-\tfrac{1}{2}, 1]`.
        :math:`\psi = 1` for the upper Fréchet bound,
        :math:`\psi = -\tfrac{1}{2}` for the lower Fréchet bound,
        :math:`\psi = 0` for independence.

        Returns
        -------
        sympy.Expr
            The symbolic expression for Spearman’s footrule coefficient.
        """
        self._set_params(args, kwargs)
        return self._spearman_footrule()

    def _spearman_footrule(self):
        t = sp.Symbol("t", positive=True)
        cdf_expr = self.cdf().func
        cdf_diag = cdf_expr.subs([(self.u, t), (self.v, t)])
        integral = sp.simplify(sp.integrate(cdf_diag, (t, 0, 1)))
        return sp.simplify(6 * integral - 2)

    # ------------------------------------------------------------------
    # Tail concentration functions
    # ------------------------------------------------------------------

    def lower_tail_concentration(self, t_val=None):
        r"""
        Lower tail concentration function :math:`L(t) = C(t,t)\,/\,t`.

        This function interpolates between :math:`L(t)\to\lambda_L` as
        :math:`t\to 0^+` (lower tail dependence coefficient) and
        :math:`L(1) = C(1,1) = 1`.  A plot of :math:`L(t)` vs :math:`t`
        visualises how tail dependence builds up.

        Parameters
        ----------
        t_val : float or None
            If given, evaluate numerically at this point in :math:`(0,1)`.
            If ``None``, return the SymPy expression in the symbol ``t``.

        Returns
        -------
        sympy.Expr or float
        """
        t = sp.Symbol("t", positive=True)
        cdf_expr = self.cdf().func
        cdf_diag = cdf_expr.subs([(self.u, t), (self.v, t)])
        L_expr = cdf_diag / t
        if t_val is not None:
            return float(L_expr.subs(t, t_val))
        return L_expr

    def upper_tail_concentration(self, t_val=None):
        r"""
        Upper tail concentration function
        :math:`R(t) = (1 - 2t + C(t,t))\,/\,(1-t)`.

        This function satisfies :math:`R(t)\to\lambda_U` as :math:`t\to 1^-`
        (upper tail dependence coefficient) and :math:`R(0) = 1`.

        Parameters
        ----------
        t_val : float or None
            If given, evaluate numerically at this point in :math:`(0,1)`.
            If ``None``, return the SymPy expression in the symbol ``t``.

        Returns
        -------
        sympy.Expr or float
        """
        t = sp.Symbol("t", positive=True)
        cdf_expr = self.cdf().func
        cdf_diag = cdf_expr.subs([(self.u, t), (self.v, t)])
        R_expr = (1 - 2 * t + cdf_diag) / (1 - t)
        if t_val is not None:
            return float(R_expr.subs(t, t_val))
        return R_expr

    def tail_dependence_function(self, t, lower=True):
        r"""Tail dependence function (TDF) evaluated numerically.

        The *lower* TDF is defined as

        .. math::

           b_L(t) = \lim_{s\to 0^+}
                    \frac{C\bigl(s(1-t),\, st\bigr)}{s},
                    \qquad t\in[0,1],

        and the *upper* TDF as

        .. math::

           b_U(t) = \lim_{s\to 0^+}
                    \frac{\hat C\bigl(s(1-t),\, st\bigr)}{s}

        where :math:`\hat C` is the survival copula.

        The diagonal value :math:`b(1/2)` equals :math:`\lambda / 2`
        where :math:`\lambda` is the corresponding tail dependence
        coefficient.  The full function describes *how* probability
        mass concentrates in the tail.

        Parameters
        ----------
        t : float or array_like
            Point(s) in :math:`[0,1]`.
        lower : bool
            ``True`` for the lower TDF, ``False`` for the upper TDF.

        Returns
        -------
        float or numpy.ndarray

        References
        ----------
        Joe & Li (2011), *Tail Risk of Multivariate Regular Variation*,
        Methodology and Computing in Applied Probability 13, 671--693.
        """
        t = np.asarray(t, dtype=float)
        eps = 1e-7

        if hasattr(self, "cdf_vectorized"):
            def _cdf_scalar(u, v):
                return float(self.cdf_vectorized(
                    np.array([u]), np.array([v])
                )[0])
        else:
            def _cdf_scalar(u, v):
                return float(self.cdf(u=u, v=v))

        def _eval(ti):
            if ti <= 0 or ti >= 1:
                return 0.0
            u_s = eps * (1.0 - ti)
            v_s = eps * ti
            if lower:
                return _cdf_scalar(u_s, v_s) / eps
            else:
                return (u_s + v_s - 1.0 + _cdf_scalar(1.0 - u_s, 1.0 - v_s)) / eps

        if t.ndim == 0:
            return _eval(float(t))
        return np.array([_eval(float(ti)) for ti in t.ravel()]).reshape(t.shape)

    def tail_order(self):
        r"""Tail order :math:`\kappa` (lower and upper).

        The tail order describes the polynomial rate at which
        :math:`C(t,t)` vanishes as :math:`t\to 0^+`:

        .. math::

           C(t,t) \sim t^{\kappa_L} \ell(t), \qquad t \to 0^+.

        Analogously, :math:`\hat C(t,t) \sim t^{\kappa_U}\ell(t)` as
        :math:`t\to 0^+` defines the upper tail order.

        - :math:`\kappa = 1` ⟹ tail dependence (:math:`\lambda > 0`).
        - :math:`1 < \kappa < 2` ⟹ intermediate tail dependence.
        - :math:`\kappa = 2` ⟹ tail independence (same rate as Π).

        Computed here by a log--log regression on small quantiles.

        Returns
        -------
        dict
            ``{"lower": kappa_L, "upper": kappa_U}``

        References
        ----------
        Ledford & Tawn (1996), *Statistics for near independence in
        multivariate extreme values*, Biometrika 83, 169--187.
        """
        ts = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        log_t = np.log(ts)

        if hasattr(self, "cdf_vectorized"):
            c_diag = self.cdf_vectorized(ts, ts)
        else:
            c_diag = np.array([float(self.cdf(u=t, v=t)) for t in ts])

        # Lower tail order: C(t,t) ~ t^kappa_L
        pos_mask = c_diag > 0
        if np.sum(pos_mask) >= 2:
            kappa_L = float(np.polyfit(log_t[pos_mask], np.log(c_diag[pos_mask]), 1)[0])
        else:
            kappa_L = float("inf")

        # Upper tail order: Chat(t,t) ~ t^kappa_U
        c_surv = 1.0 - 2.0 * ts + c_diag  # Ĉ(t,t)
        pos_mask_u = c_surv > 0
        if np.sum(pos_mask_u) >= 2:
            kappa_U = float(np.polyfit(log_t[pos_mask_u], np.log(c_surv[pos_mask_u]), 1)[0])
        else:
            kappa_U = float("inf")

        return {"lower": kappa_L, "upper": kappa_U}

    def plot_tail_concentration(self, n_pts: int = 200) -> "plt.Figure":
        r"""
        Plot the lower and upper tail concentration functions on one figure.

        The lower curve :math:`L(t)=C(t,t)/t` and the upper curve
        :math:`R(t)=(1-2t+C(t,t))/(1-t)` are drawn against :math:`t\in(0,1)`.
        Horizontal dashed lines mark the tail dependence coefficients
        :math:`\lambda_L` and :math:`\lambda_U`.

        Parameters
        ----------
        n_pts : int
            Number of grid points (default 200).

        Returns
        -------
        matplotlib.figure.Figure
        """
        t_sym = sp.Symbol("t", positive=True)
        cdf_expr = self.cdf().func
        cdf_diag = cdf_expr.subs([(self.u, t_sym), (self.v, t_sym)])

        L_expr = cdf_diag / t_sym
        R_expr = (1 - 2 * t_sym + cdf_diag) / (1 - t_sym)

        f_L = to_numpy_callable(L_expr, [t_sym], ae=True)
        f_R = to_numpy_callable(R_expr, [t_sym], ae=True)

        t_vals = np.linspace(0.01, 0.99, n_pts)
        with np.errstate(divide="ignore", invalid="ignore"):
            L_vals = np.vectorize(f_L)(t_vals)
            R_vals = np.vectorize(f_R)(t_vals)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_vals, L_vals, label=r"$L(t) = C(t,t)/t$", linewidth=2)
        ax.plot(t_vals, R_vals, label=r"$R(t) = (1-2t+C(t,t))/(1-t)$", linewidth=2)

        try:
            lam_L = float(self.lambda_L())
            lam_U = float(self.lambda_U())
            ax.axhline(
                lam_L,
                color="C0",
                linestyle="--",
                alpha=0.6,
                label=rf"$\lambda_L={lam_L:.3f}$",
            )
            ax.axhline(
                lam_U,
                color="C1",
                linestyle="--",
                alpha=0.6,
                label=rf"$\lambda_U={lam_U:.3f}$",
            )
        except Exception:
            pass

        ax.set_xlabel("$t$")
        ax.set_ylabel("Concentration")
        title = CopulaGraphs(self).get_copula_title()
        ax.set_title(f"{title} — Tail concentration")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        return fig

    # ------------------------------------------------------------------
    # Concordance ordering
    # ------------------------------------------------------------------

    def concordance_order(
        self, other: "BivCoreCopula", n_grid: int = 20, tol: float = 1e-9
    ) -> bool:
        r"""
        Numerically check whether *self* is concordance-ordered below *other*.

        Returns ``True`` if :math:`C_1(u,v) \le C_2(u,v)` for all
        :math:`(u,v)` on a uniform :math:`n\_grid \times n\_grid` grid over
        :math:`(0.05, 0.95)^2`, i.e. whether
        :math:`C_1 \preceq_c C_2` in the concordance partial order.

        Parameters
        ----------
        other : BivCoreCopula
            The copula to compare against.
        n_grid : int
            Number of evaluation points per axis (default 20).
        tol : float
            Numerical tolerance for the inequality (default 1e-9).

        Returns
        -------
        bool
            ``True`` if *self* ≤_c *other* on the grid, ``False`` otherwise.
        """
        grid = np.linspace(0.05, 0.95, n_grid)
        for u in grid:
            for v in grid:
                u_f, v_f = float(u), float(v)
                c1 = float(self.cdf(u=u_f, v=v_f))
                c2 = float(other.cdf(u=u_f, v=v_f))
                if c1 > c2 + tol:
                    return False
        return True

    # ==================================================================
    # Schweizer–Wolff sigma
    # ==================================================================

    def schweizer_wolff_sigma(self, *args, **kwargs):
        r"""
        Schweizer–Wolff's :math:`\sigma` dependence measure.

        .. math::

           \sigma(C)
             = 12 \iint_{[0,1]^2} \lvert C(u,v) - uv \rvert\,du\,dv

        Unlike Spearman's :math:`\rho` (which uses the signed deviation
        :math:`C - \Pi`), this measure takes the absolute value and therefore
        captures *any* departure from independence, including non-monotone
        dependence.

        Range: :math:`[0, 1]`.
        :math:`\sigma = 0` iff :math:`C = \Pi` (independence).

        When the integral cannot be evaluated symbolically (because
        :math:`|C - \Pi|` introduces piecewise terms that SymPy cannot
        always simplify), the method falls back to numerical quadrature on
        a 50 × 50 grid.

        Returns
        -------
        sympy.Expr or float
        """
        self._set_params(args, kwargs)
        return self._schweizer_wolff_sigma()

    def _schweizer_wolff_sigma(self):
        cdf_expr = self.cdf().func
        Pi = self.u * self.v  # independence copula
        integrand = sp.Abs(cdf_expr - Pi)

        # Try symbolic integration first
        try:
            inner = sp.integrate(integrand, (self.v, 0, 1))
            result = sp.simplify(12 * sp.integrate(inner, (self.u, 0, 1)))
            # If SymPy returns an unevaluated Integral, fall through
            if not result.has(sp.Integral):
                return result
        except Exception:
            pass

        # Numerical fallback
        return self._schweizer_wolff_sigma_numerical()

    def _schweizer_wolff_sigma_numerical(self, n_grid: int = 50) -> float:
        r"""Evaluate :math:`\sigma` by midpoint-rule quadrature."""
        cdf_expr = self.cdf().func
        f_cdf = to_numpy_callable(cdf_expr, [self.u, self.v])
        h = 1.0 / n_grid
        mid = np.linspace(h / 2, 1 - h / 2, n_grid)
        uu, vv = np.meshgrid(mid, mid, indexing="ij")
        C_vals = np.vectorize(f_cdf)(uu, vv)
        Pi_vals = uu * vv
        return float(12 * np.mean(np.abs(C_vals - Pi_vals)))

    # ==================================================================
    # Hoeffding's D  (dependence index)
    # ==================================================================

    def hoeffdings_d(self, *args, **kwargs):
        r"""
        Hoeffding's :math:`D` dependence measure (also called the
        *dependence index* or :math:`\Phi^2`).

        .. math::

           D(C) = 90 \iint_{[0,1]^2} \bigl[C(u,v) - uv\bigr]^2\,du\,dv

        This is the squared :math:`L_2` analogue of Spearman's :math:`\rho`
        (which uses an :math:`L_1` integral of the signed deviation).
        It measures any departure from independence, monotone or not.

        Range: :math:`[0, 1]`.
        :math:`D = 0` iff :math:`C = \Pi` (independence).

        Falls back to numerical quadrature when SymPy cannot evaluate the
        double integral in closed form.

        Returns
        -------
        sympy.Expr or float

        References
        ----------
        Hoeffding, W. (1940). "Masstabinvariante Korrelationstheorie."
        Schweizer, B. & Wolff, E. F. (1981). "On Nonparametric Measures of
        Dependence for Random Variables." *Ann. Statist.* 9(4).
        """
        self._set_params(args, kwargs)
        return self._hoeffdings_d()

    def _hoeffdings_d(self):
        cdf_expr = self.cdf().func
        Pi = self.u * self.v
        integrand = (cdf_expr - Pi) ** 2

        try:
            inner = sp.integrate(integrand, (self.v, 0, 1))
            result = sp.simplify(90 * sp.integrate(inner, (self.u, 0, 1)))
            if not result.has(sp.Integral):
                return result
        except Exception:
            pass

        return self._hoeffdings_d_numerical()

    def _hoeffdings_d_numerical(self, n_grid: int = 50) -> float:
        r"""Evaluate :math:`D` by midpoint-rule quadrature."""
        cdf_expr = self.cdf().func
        f_cdf = to_numpy_callable(cdf_expr, [self.u, self.v])
        h = 1.0 / n_grid
        mid = np.linspace(h / 2, 1 - h / 2, n_grid)
        uu, vv = np.meshgrid(mid, mid, indexing="ij")
        C_vals = np.vectorize(f_cdf)(uu, vv)
        Pi_vals = uu * vv
        return float(90 * np.mean((C_vals - Pi_vals) ** 2))

    # ==================================================================
    # Lp concordance distance  (generalisation of rho, sigma, D)
    # ==================================================================

    # Normalisation constants k(p) so that the measure equals 1 at the
    # Fréchet upper bound M(u,v)=min(u,v).
    _LP_NORM_CONSTANTS = {1: 12, 2: 90, 3: 560, 4: 3150, 5: 16632}

    def lp_concordance(self, p: int = 2, *args, **kwargs):
        r"""
        :math:`L_p` concordance distance from independence.

        .. math::

           \delta_p(C) = k(p)\,
             \iint_{[0,1]^2} \lvert C(u,v) - uv \rvert^{p}\,du\,dv

        where :math:`k(p)` is chosen so that :math:`\delta_p(M) = 1` for the
        Fréchet upper bound :math:`M(u,v) = \min(u,v)`.

        Special cases:

        ============ ======== ======================================
        :math:`p`    :math:`k(p)` Equivalent measure
        ============ ======== ======================================
        1            12       Schweizer–Wolff :math:`\sigma`
        2            90       Hoeffding :math:`D`
        ============ ======== ======================================

        Parameters
        ----------
        p : int
            The exponent (default 2).  Pre-tabulated for *p* = 1 … 5.

        Returns
        -------
        sympy.Expr or float
        """
        self._set_params(args, kwargs)
        return self._lp_concordance(p)

    def _lp_concordance(self, p: int):
        k = self._LP_NORM_CONSTANTS.get(p)
        if k is None:
            raise ValueError(
                f"Normalisation constant k({p}) not tabulated.  "
                f"Supported p values: {sorted(self._LP_NORM_CONSTANTS)}"
            )

        cdf_expr = self.cdf().func
        Pi = self.u * self.v
        integrand = sp.Abs(cdf_expr - Pi) ** p

        # Symbolic attempt
        try:
            inner = sp.integrate(integrand, (self.v, 0, 1))
            result = sp.simplify(k * sp.integrate(inner, (self.u, 0, 1)))
            if not result.has(sp.Integral):
                return result
        except Exception:
            pass

        # Numerical fallback
        return self._lp_concordance_numerical(p, k)

    def _lp_concordance_numerical(self, p: int, k: int, n_grid: int = 50) -> float:
        cdf_expr = self.cdf().func
        f_cdf = to_numpy_callable(cdf_expr, [self.u, self.v])
        h = 1.0 / n_grid
        mid = np.linspace(h / 2, 1 - h / 2, n_grid)
        uu, vv = np.meshgrid(mid, mid, indexing="ij")
        C_vals = np.vectorize(f_cdf)(uu, vv)
        Pi_vals = uu * vv
        return float(k * np.mean(np.abs(C_vals - Pi_vals) ** p))

    # ==================================================================
    # Mutual information  I(C)  (copula entropy)
    # ==================================================================

    def mutual_information(self, *args, **kwargs):
        r"""
        Copula-based mutual information (negative copula entropy).

        .. math::

           I(C) = -\iint_{[0,1]^2} c(u,v)\,\ln c(u,v)\,du\,dv

        where :math:`c(u,v) = \partial^2 C/\partial u\,\partial v` is the
        copula density.  This equals the mutual information of a pair
        :math:`(X,Y)` whose copula is :math:`C`, regardless of the marginals.

        Range: :math:`[0, \infty)`.
        :math:`I = 0` iff :math:`C = \Pi` (independence).

        Because the integrand involves :math:`\ln(c)`, symbolic evaluation
        rarely succeeds; the method defaults to numerical quadrature.

        Parameters
        ----------
        n_grid : int
            Number of grid points per axis for quadrature (default 80).

        Returns
        -------
        float
        """
        self._set_params(args, kwargs)
        n_grid = kwargs.pop("n_grid", 80)
        return self._mutual_information_numerical(n_grid)

    def _mutual_information_numerical(self, n_grid: int = 80) -> float:
        from copul.exceptions import PropertyUnavailableException

        pdf_expr = self.pdf
        if callable(pdf_expr) and not isinstance(pdf_expr, sp.Basic):
            # pdf might be a wrapper callable
            pass
        try:
            f_pdf = to_numpy_callable(pdf_expr, [self.u, self.v])
        except (TypeError, AttributeError):
            raise PropertyUnavailableException(
                "mutual_information requires a symbolic PDF expression."
            )

        h = 1.0 / n_grid
        mid = np.linspace(h / 2, 1 - h / 2, n_grid)
        uu, vv = np.meshgrid(mid, mid, indexing="ij")
        with np.errstate(divide="ignore", invalid="ignore"):
            c_vals = np.vectorize(f_pdf)(uu, vv)
            c_vals = np.maximum(c_vals, 0.0)  # clip negative numerical noise
            log_c = np.where(c_vals > 1e-300, np.log(c_vals), 0.0)
            integrand = c_vals * log_c
        return float(-np.mean(integrand))
