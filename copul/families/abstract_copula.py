import copy
import inspect
import logging
import pathlib
import types
from abc import ABC, abstractmethod

import numpy as np
import sympy
from matplotlib import pyplot as plt

from copul.families.chatterjee_plotter import ChatterjeePlotter
from copul.families.cis_verifier import CISVerifier
from copul.families.copula_sampler import CopulaSampler
from copul.families.tp2_verifier import TP2Verifier
from copul.sympy_wrapper import SymPyFunctionWrapper

log = logging.getLogger(__name__)


class AbstractCopula(ABC):
    params = None
    u, v = sympy.symbols("u v", positive=True)
    intervals = None
    log_cut_off = 4

    def __init__(self, **kwargs):
        self._are_class_vars(kwargs)
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(self, k, v)
        self.params = [param for param in self.params if str(param) not in kwargs]

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, **kwargs):
        new_copula = copy.copy(self)
        new_copula._are_class_vars(kwargs)
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(new_copula, k, v)
        new_copula.params = [param for param in self.params if str(param) not in kwargs]
        return new_copula

    @property
    def name(self):
        return self.__class__.__name__

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
    @abstractmethod
    def cdf(self) -> SymPyFunctionWrapper:
        pass

    @property
    @abstractmethod
    def is_absolutely_continuous(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_symmetric(self) -> bool:
        pass

    def rvs(self, n=1):
        """Sample a value from the copula"""

        cond_distr = self.cond_distr_2
        sig = inspect.signature(cond_distr)
        params = list(sig.parameters.values())
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
        print(sampler.err_counter)
        return result

    @property
    def pdf(self):
        return sympy.simplify(sympy.diff(self.cond_distr_2(), self.u))

    def cond_distr_1(self) -> SymPyFunctionWrapper:
        return SymPyFunctionWrapper(sympy.diff(self.cdf, self.u))

    def cond_distr_2(self) -> SymPyFunctionWrapper:
        return SymPyFunctionWrapper(sympy.diff(self.cdf, self.v))

    def xi(self):
        print("xi")
        cond_distri_1 = sympy.simplify(self.cond_distr_1())
        print("cond_distr_1 sympy: ", cond_distri_1)
        print("cond_distr_1: ", sympy.latex(cond_distri_1))
        # sample_int = sympy.simplify(sympy.integrate(cond_distri_1, self.u))
        # print("sample_int sympy: ", sample_int)
        # print("sample_int: ", sympy.latex(sample_int))
        squared_cond_distr_1 = self._squared_cond_distr_1(self.u, self.v)
        print("squared_cond_distr_1 sympy: ", squared_cond_distr_1)
        print("squared_cond_distr_1: ", sympy.latex(squared_cond_distr_1))
        int_1 = self._xi_int_1(self.v)
        print("int_1 sympy: ", int_1)
        print("int_1: ", sympy.latex(int_1))
        int_2 = self._xi_int_2()
        print("int_2 sympy: ", int_2)
        print("int_2: ", sympy.latex(int_2))
        xi = self._xi()
        print("xi sympy: ", xi)
        print("xi: ", sympy.latex(xi))
        return SymPyFunctionWrapper(xi)

    def rho(self):
        print("rho")
        if isinstance(self.cdf, SymPyFunctionWrapper):
            cdf = sympy.simplify(self.cdf.func)
        else:
            cdf = self.cdf
        print("cdf sympy: ", cdf)
        print("cdf latex: ", sympy.latex(cdf))
        int_1 = self._rho_int_1()
        print("int_1 sympy: ", int_1)
        print("int_1 latex: ", sympy.latex(int_1))
        rho = self._rho()
        print("rho sympy: ", rho)
        print("rho latex: ", sympy.latex(rho))
        return rho

    def _rho(self):
        return sympy.simplify(12 * self._rho_int_2() - 3)

    def tau(self):
        print("tau")
        if isinstance(self.cdf, SymPyFunctionWrapper):
            integrand = self.cdf.func * self.pdf
        else:
            integrand = self.cdf * self.pdf
        print("integrand sympy: ", integrand)
        print("integrand latex: ", sympy.latex(integrand))
        int_1 = self._tau_int_1()
        print("int_1 sympy: ", int_1)
        print("int_1 latex: ", sympy.latex(int_1))
        int_2 = self._tau_int_2()
        print("int_2 sympy: ", int_2)
        print("int_2 latex: ", sympy.latex(int_2))
        tau = self._tau()
        print("tau sympy: ", tau)
        print("tau latex: ", sympy.latex(tau))
        return tau

    def _tau(self, *args):
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
            plt.title(f"{self.__class__.__name__} {', '.join([*kwargs])}")
            plt.grid(True)
            pathlib.Path("images").mkdir(exist_ok=True)
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(f"images/{self.__class__.__name__}.png")

    def plot_cdf(self, data=None, title=None, zlabel=None):
        if title is None:
            title = f"{type(self).__name__} Copula"
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

    def plot_chatterjee(
        self,
        n_obs,
        n_params,
        plot_var=False,
        ylim=(-1, 1),
        params=None,
        log_cut_off=None,
    ):
        plotter = ChatterjeePlotter(self, log_cut_off)
        plotter.plot_chatterjee(n_obs, n_params, params, plot_var, ylim)

    def plot_pdf(self):
        free_symbol_dict = {str(s): getattr(self, str(s)) for s in self.params}
        pdf = self(**free_symbol_dict).pdf
        return self._plot3d(pdf, title=f"{type(self).__name__} Copula", zlabel="PDF")

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
