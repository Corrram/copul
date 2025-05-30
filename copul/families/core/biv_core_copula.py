import inspect
import logging
import pathlib
import types

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors 

from copul.schur_order.cis_verifier import CISVerifier
from copul.families.copula_graphs import CopulaGraphs
from copul.families.rank_correlation_plotter import RankCorrelationPlotter
from copul.families.tp2_verifier import TP2Verifier
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

    Attributes
    ----------
    u, v : sympy.Symbol
        Symbols representing the two dimensions of the copula.
    log_cut_off : int
        Default cut-off used for logarithmic scales.
    _package_path : pathlib.Path
        Path to the parent package directory.
    """

    u, v = sp.symbols("u v", nonnegative=True)
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
        result = self.cdf.diff(self.u)
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
        result = CD2Wrapper(sp.diff(self.cdf, self.v))
        return result(u, v)

    def xi(self, *args, **kwargs):
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

    def rho(self, *args, **kwargs):
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

    def tau(self, *args, **kwargs):
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
        return sp.simplify(sp.integrate(self.cdf.func, (self.u, 0, 1)))

    def _tau_int_1(self):
        """
        Compute the integrand for Kendall's tau with respect to variable u.

        Returns
        -------
        sympy.Expr
            The simplified expression of the integrand.
        """
        return sp.simplify(sp.integrate(self.cdf.func * self.pdf, (self.u, 0, 1)))

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
        plotter.plot_rank_correlations(n_obs, n_params, params, plot_var, ylim)

    def plot_cdf(self, data=None, title=None, zlabel=None, *, plot_type="3d", log_z=False, **kwargs):
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
            raise ValueError("Contour plots from external data are not supported – set plot_type='3d' or omit *data*.")

        if data is None:
            if plot_type == "3d":
                return self._plot3d(self.cdf, title=title, zlabel=zlabel, zlim=(0, 1))
            return self._plot_contour(self.cdf, title=title, zlabel=zlabel, zlim=(0, 1), log_z=log_z, **kwargs)
        return self._plot_cdf_from_data(data)

    def plot_pdf(self, *args, plot_type="3d", log_z=False, **kwargs):
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

    def plot_cond_distr_1(self, *, plot_type="3d", log_z=False, **kwargs):
        if plot_type not in {"3d", "contour"}:
            raise ValueError("plot_type must be '3d' or 'contour'")
        cond_distr_1 = self.cond_distr_1
        title = CopulaGraphs(self).get_copula_title()
        if plot_type == "3d":
            return self._plot3d(cond_distr_1, title=title, zlabel="Conditional Distribution 1")
        return self._plot_contour(cond_distr_1, title=title, zlabel="Conditional Distribution 1", log_z=log_z, **kwargs)

    def plot_cond_distr_2(self, *, plot_type="3d", log_z=False, **kwargs):
        if plot_type not in {"3d", "contour"}:
            raise ValueError("plot_type must be '3d' or 'contour'")
        cond_distr_2 = self.cond_distr_2
        title = CopulaGraphs(self).get_copula_title()
        if plot_type == "3d":
            return self._plot3d(cond_distr_2, title=title, zlabel="Conditional Distribution 2")
        return self._plot_contour(cond_distr_2, title=title, zlabel="Conditional Distribution 2", log_z=log_z, **kwargs)

    def _plot3d(self, func, title, zlabel, zlim=None):
        """
        Generate a 3D plot of a given function of two variables.

        The function can be a SymPy expression wrapped in a SymPyFuncWrapper,
        or any callable that accepts (u, v).

        Parameters
        ----------
        func : callable or SymPyFuncWrapper or sp.Expr
            The function to plot.
        title : str
            Title for the plot.
        zlabel : str
            Label for the z-axis.
        zlim : tuple, optional
            Limits for the z-axis as (min, max).

        Returns
        -------
        None
        """
        intervals = {k: v for k, v in self.intervals.items()}
        try:
            _parameters = inspect.signature(func).parameters
        except TypeError:
            pass
        else:
            if isinstance(func, types.MethodType):  #  and len(parameters) == 0:
                func = func()
        if isinstance(func, (SymPyFuncWrapper, CD1Wrapper, CD2Wrapper, CDiWrapper)):
            f = sp.lambdify((self.u, self.v), func.func)
        elif isinstance(func, sp.Expr):
            f = sp.lambdify((self.u, self.v), func)
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
        self.intervals = intervals  # otherwise it may have been modified by the plot


    def _plot_contour(self, func, title, zlabel, *, levels=50, zlim=None, log_z=False, **kwargs):
        r"""Create a filled contour plot.

        If ``log_z`` is *True*, the colour map is based on
        :math:`\log(1+z)` (simply `numpy.log1p`).  This means zeros map to
        *exactly* 0 on the log scale instead of being excluded.  The colour bar
        still shows the **original values** (by subtracting 1 again in its tick
        labels) so that the legend remains interpretable in the natural scale.
        """
        intervals_backup = dict(self.intervals)
        # Resolve callable --------------------------------------------------
        try:
            _ = inspect.signature(func).parameters
            if isinstance(func, types.MethodType):
                func = func()
        except TypeError:
            pass
        if isinstance(func, (SymPyFuncWrapper, CD1Wrapper, CD2Wrapper, CDiWrapper)):
            f = sp.lambdify((self.u, self.v), func.func)
        elif isinstance(func, sp.Expr):
            f = sp.lambdify((self.u, self.v), func)
        else:
            f = func
        # Evaluate grid -----------------------------------------------------
        if "grid_size" in kwargs:
            grid_size = kwargs.pop("grid_size")
        else:
            grid_size = 100
        x = np.linspace(0.005, 0.995, grid_size)
        y = np.linspace(0.005, 0.995, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(f)(X, Y)
        if zlim is not None:
            Z = np.clip(Z, zlim[0], zlim[1])
        # Colour scaling ----------------------------------------------------
        cmap = plt.cm.get_cmap("viridis").copy()
        if log_z:
            # mask negatives, shift by +1 for log1p
            Z_mask = np.ma.masked_less(Z, 0.0)
            Z_for_norm = np.ma.masked_less(Z_mask + 1.0, 1e-12)
            vmin = (zlim[0] + 1) if zlim else Z_for_norm.min()
            vmax = (zlim[1] + 1) if zlim else Z_for_norm.max()
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            # use more levels for smoother transitions
            if isinstance(levels, int):
                levels = np.geomspace(vmin, vmax, levels * 5)
            cf = plt.contourf(X, Y, Z_for_norm, levels=levels, cmap=cmap, norm=norm)
            cbar = plt.colorbar(cf)
            ticks = cbar.get_ticks()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t - 1:g}" for t in ticks])
        else:
            if isinstance(levels, int):
                levels = levels
            cf = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
            cbar = plt.colorbar(cf)
        cbar.set_label(zlabel)
        plt.xlabel("u")
        plt.ylabel("v")
        plt.title(title)
        plt.show()
        self.intervals = intervals_backup


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
