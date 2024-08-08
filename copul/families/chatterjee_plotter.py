import itertools
import logging
import pathlib
import pickle

import numpy as np
import scipy
import sympy
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from copul import chatterjee

log = logging.getLogger(__name__)


class ChatterjeePlotter:
    def __init__(self, copul, log_cut_off=None):
        self.copul = copul
        self.log_cut_off = log_cut_off

    def plot_chatterjee(
        self, n_obs=10_000, n_params=20, params=None, plot_var=False, ylim=(-1, 1)
    ):
        log.info(f"Plotting Chatterjee graph for {type(self.copul).__name__} copula")
        mixed_params = self._mix_params(params) if params is not None else {}
        if self.log_cut_off is not None:
            log_scale = True
        else:
            log_scale = False
        if not mixed_params:
            filename_suffix = ""
            self._plot_correlation_for(
                n_obs, n_params, self.copul, plot_var, log_scale=log_scale
            )
            const_params = {*self.copul.intervals} - set(
                {str(param) for param in self.copul.params}
            )
            legend_suffix = ""
            for p in const_params:
                legend_suffix += ", "
                param = getattr(self, str(p))
                if isinstance(param, (property, sympy.Symbol)):
                    legend_suffix += f"$\\{p}=\\{param}$"
                else:
                    legend_suffix += f"$\\{p}={param}$"
                legend_suffix = " (with " + legend_suffix[2:] + ")"
                legend_suffix = legend_suffix.replace("),", ",")
        else:
            filename_suffix = "_xi"
            legend_suffix = ""
        for mixed_param in mixed_params:
            new_copula = self.copul(**mixed_param)
            label = ", ".join(
                f"$\\{k}=\\{v}$" if isinstance(v, (property, str)) else f"$\\{k}={v}$"
                for k, v in mixed_param.items()
            )
            self._construct_xi_graph_for(
                n_obs, n_params, new_copula, plot_var, label, log_scale
            )
            plt.ylabel(r"$\xi$")
        plt.legend()  # legend with n_obs and n_params
        if params is None:
            x_param = self.copul.params[0]
        else:
            x_param = [
                param for param in self.copul.params if str(param) not in [*params]
            ][0]
        x_label = f"$\\{x_param}${legend_suffix}"
        plt.xlabel(x_label)
        plt.ylim(0, 1) if mixed_params else plt.ylim(*ylim)
        plt.title(f"{self.__class__.__name__} Copula")
        plt.grid(True)
        pathlib.Path("images").mkdir(exist_ok=True)
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig(f"images/{self.__class__.__name__}{filename_suffix}.png")
        pathlib.Path("images/functions").mkdir(exist_ok=True)

    def _construct_xi_graph_for(
        self, n_obs, n_params, new_copula, plot_var, label=r"$\xi$", log_scale=False
    ):
        params = new_copula.get_params(n_params, log_scale=log_scale)
        data_points = []
        for param in params:
            data = new_copula(**{str(new_copula.params[0]): param}).rvs(n_obs)
            xi = chatterjee.xi_ncalculate(data[:, 0], data[:, 1])
            if plot_var:
                xivar = chatterjee.xi_nvarcalculate(data[:, 0], data[:, 1])
                y_err = 3.291 * np.sqrt(xivar / n_obs)
            else:
                y_err = 0
            data_points.append((param, xi, y_err))
        data_points = np.array(data_points)
        x = data_points[:, 0]
        y = data_points[:, 1]
        y_errs = data_points[:, 2]
        cs = CubicSpline(x, y) if len(x) > 1 else lambda x_: x_
        # Create a dense set of x-values for plotting
        if log_scale:
            left_boundary = float(self.copul.intervals[str(self.copul.params[0])].inf)
            if isinstance(self.log_cut_off, tuple):
                x_dense = np.logspace(*self.log_cut_off, 500) + left_boundary
            else:
                x_dense = (
                    np.logspace(-self.log_cut_off, self.log_cut_off, 500)
                    + left_boundary
                )
        else:
            x_dense = np.linspace(params[0], params[-1], 500)
        # Compute the corresponding y-values
        y_dense = cs(x_dense)
        # Plot the results
        plt.scatter(x, y, label=label)
        cs_label = "Cubic Spline" if label == r"$\xi$" else None
        plt.plot(x_dense, y_dense, label=cs_label)
        if log_scale:
            plt.xscale("log")
        plt.fill_between(x, y - y_errs, y + y_errs, alpha=0.2)
        self._save_data_and_splines(cs, data_points)

    def _plot_correlation_for(
        self, n_obs, n_params, new_copula, plot_var, log_scale=False
    ):
        params = self.get_params(n_params, log_scale=log_scale)
        data_points = []
        for param in params:
            specific_copula = new_copula(**{str(new_copula.params[0]): param})
            data = specific_copula.rvs(n_obs)
            xi = chatterjee.xi_ncalculate(data[:, 0], data[:, 1])
            rho = scipy.stats.spearmanr(data[:, 0], data[:, 1])
            tau = scipy.stats.kendalltau(data[:, 0], data[:, 1])
            if plot_var:
                xivar = chatterjee.xi_nvarcalculate(data[:, 0], data[:, 1])
                y_err = 3.291 * np.sqrt(xivar / n_obs)
            else:
                y_err = 0
            data_points.append((param, xi, y_err, rho[0], tau[0]))
        data_points = np.array(data_points)
        x = data_points[:, 0]
        y = data_points[:, 1]
        # y_errs = data_points[:, 2]
        y_rho = data_points[:, 3]
        y_tau = data_points[:, 4]
        cs = CubicSpline(x, y) if len(x) > 1 else lambda x_: x_
        cs_rho = CubicSpline(x, y_rho) if len(x) > 1 else lambda x_: x_
        cs_tau = CubicSpline(x, y_tau) if len(x) > 1 else lambda x_: x_
        # Create a dense set of x-values for plotting
        if log_scale:
            inf = float(self.copul.intervals[str(self.copul.params[0])].inf)
            if isinstance(self.log_cut_off, tuple):
                x_dense = np.logspace(*self.log_cut_off, 500) + inf
            else:
                x_dense = np.logspace(-self.log_cut_off, self.log_cut_off, 500) + inf
        else:
            x_dense = np.linspace(params[0], params[-1], 500)
            inf = 0
        # Compute the corresponding y-values
        y_dense = cs(x_dense)
        y_rho_dense = cs_rho(x_dense)
        y_tau_dense = cs_tau(x_dense)
        # Plot the results
        plt.scatter(x - inf, y, label="Chatterjee's xi", marker="o")
        plt.scatter(x - inf, y_rho, label="Spearman's rho", marker="^")
        plt.scatter(x - inf, y_tau, label="Kendall's tau", marker="s")
        plt.plot(x_dense - inf, y_dense)
        plt.plot(x_dense - inf, y_rho_dense)
        plt.plot(x_dense - inf, y_tau_dense)
        if log_scale:
            plt.xscale("log")
        if log_scale and inf != 0.0:
            ticks = plt.xticks()[0]
            infimum = int(inf) if inf.is_integer() else inf
            new_ticklabels = [f"${infimum} + 10^{{{int(np.log10(t))}}}$" for t in ticks]
            plt.xticks(ticks, new_ticklabels)
            plt.xlim(x[0] - inf, x[-1] - inf)
        # plt.fill_between(x - inf, y - y_errs, y + y_errs, alpha=0.2)

        self._save_data_and_splines(cs, data_points)

    def _save_data_and_splines(self, cs, data_points):
        pathlib.Path("images/functions").mkdir(exist_ok=True, parents=True)
        if isinstance(cs, CubicSpline):
            with open(f"images/functions/{self.__class__.__name__}.pkl", "wb") as f:
                pickle.dump(cs, f)
        with open(f"images/functions/{self.__class__.__name__}Data.pkl", "wb") as f:
            pickle.dump(data_points, f)

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

    def get_params(self, n_params, log_scale=False):
        interval = self.copul.intervals[str(self.copul.params[0])]
        if isinstance(interval, sympy.FiniteSet):
            return np.array([float(val) for val in interval])
        cut_off = self.log_cut_off if log_scale else 10
        if log_scale:
            inf = float(interval.inf)
            if isinstance(cut_off, tuple):
                param_array = np.logspace(*cut_off, n_params) + inf
            else:
                param_array = np.logspace(-cut_off, cut_off, n_params) + inf
        else:
            if isinstance(cut_off, tuple):
                left_border = float(max(interval.inf, cut_off[0]))
                right_border = float(min(cut_off[1], interval.sup))
            else:
                left_border = float(max(-cut_off, interval.inf))
                right_border = float(min(cut_off, interval.sup))
            if interval.left_open:
                left_border += 0.01
            if interval.right_open:
                right_border -= 0.01
            param_array = np.linspace(left_border, right_border, n_params)
        return param_array
