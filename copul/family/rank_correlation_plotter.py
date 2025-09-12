"""
Module for plotting rank correlations of copulas.

This module provides tools to visualize various rank correlation measures
for copulas with different parameter settings.
"""

import itertools
import logging
import pathlib
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import sympy
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from copul.chatterjee import xi_ncalculate
from copul.family.copula_graphs import CopulaGraphs

# Set up logger
log = logging.getLogger(__name__)


@dataclass
class CorrelationData:
    """Class to store correlation data for various metrics."""

    params: np.ndarray
    xi: np.ndarray
    rho: Optional[np.ndarray] = None
    tau: Optional[np.ndarray] = None
    footrule: Optional[np.ndarray] = None
    ginis_gamma: Optional[np.ndarray] = None
    blomqvists_beta: Optional[np.ndarray] = None


class RankCorrelationPlotter:
    """
    Class for plotting rank correlations of copulas.

    This class provides functionality to plot Chatterjee's xi, Spearman's rho,
    Kendall's tau, and other rank correlations for various copula parameters.
    """

    def __init__(
        self,
        copula: Any,
        log_cut_off: Optional[Union[float, Tuple[float, float]]] = None,
        approximate: bool = True,
        xlim: int = 10,
    ):
        """
        Initialize RankCorrelationPlotter.

        Args:
            copula: Copula object to analyze.
            log_cut_off: Cut-off for log scale. Tuple for [10^a, 10^b] or single value for 10^val.
            approximate: Whether to use approximate sampling.
        """
        self.copul = copula
        self.log_cut_off = log_cut_off
        self._approximate = approximate
        self.xlim = xlim
        self.images_dir = pathlib.Path("images")
        self.functions_dir = self.images_dir / "functions"
        self.functions_dir.mkdir(parents=True, exist_ok=True)

    def plot_rank_correlations(
        self,
        n_obs: int = 10_000,
        n_params: int = 20,
        params: Optional[Dict[str, Any]] = None,
        ylim: Tuple[float, float] = (-1, 1),
    ) -> None:
        """
        Plot rank correlations for a given copula with various parameter values.

        Args:
            n_obs: Number of observations for each copula.
            n_params: Number of parameter values to evaluate.
            params: Dictionary of fixed parameter values to create multiple plots.
            ylim: Y-axis limits for the plot.
        """
        log.info(f"Plotting correlation graph for {type(self.copul).__name__} copula")
        log_scale = self.log_cut_off is not None
        mixed_params = self._mix_params(params) if params is not None else []

        if not mixed_params:
            # Plot all correlations for the primary parameter
            self._compute_and_plot(
                self.copul, n_obs, n_params, log_scale, plot_all=True
            )
        else:
            # Plot only Chatterjee's xi for each combination of fixed parameters
            for param_set in mixed_params:
                try:
                    new_copula = self.copul(**param_set)
                    label = self._format_param_label(param_set)
                    self._compute_and_plot(
                        new_copula,
                        n_obs,
                        n_params,
                        log_scale,
                        plot_all=False,
                        label_prefix=label,
                    )
                except Exception as e:
                    log.error(f"Error plotting for parameters {param_set}: {e}")

        self._finalize_plot(params, ylim, is_mixed=bool(mixed_params))

    def _compute_and_plot(
        self,
        copula: Any,
        n_obs: int,
        n_params: int,
        log_scale: bool,
        plot_all: bool,
        label_prefix: str = "",
    ):
        """Helper to compute, plot, and save correlation data."""
        param_values = self._get_parameter_values(copula, n_params, log_scale)
        data = self._compute_correlations(
            copula, param_values, n_obs, compute_all=plot_all
        )
        splines = self._plot_curves(data, log_scale, plot_all, label_prefix)
        self._save_data(copula, data, splines)

    def _compute_correlations(
        self, copula: Any, param_values: np.ndarray, n_obs: int, compute_all: bool
    ) -> CorrelationData:
        """
        Compute correlations for a range of parameter values efficiently.

        This version pre-computes ranks to avoid redundant calculations.
        """
        results = {
            "xi": [],
            "rho": [],
            "tau": [],
            "footrule": [],
            "ginis_gamma": [],
            "blomqvists_beta": [],
        }

        for param in param_values:
            try:
                specific_copula = copula(**{str(copula.params[0]): param})
                data_sample = specific_copula.rvs(n_obs, approximate=self._approximate)
                x, y = data_sample[:, 0], data_sample[:, 1]

                # --- OPTIMIZATION: Compute ranks ONCE per sample ---
                rank_x = scipy.stats.rankdata(x)
                rank_y = scipy.stats.rankdata(y)

                # --- Calculate all correlation measures ---
                results["xi"].append(xi_ncalculate(x, y))
                results["tau"].append(scipy.stats.kendalltau(x, y)[0])

                if compute_all:
                    # Spearman's Rho is the Pearson correlation of the ranks. This is faster.
                    rho, _ = scipy.stats.pearsonr(rank_x, rank_y)
                    results["rho"].append(rho)

                    # Use pre-computed ranks for our custom functions
                    results["footrule"].append(self.spearman_footrule(rank_x, rank_y))
                    results["ginis_gamma"].append(self.ginis_gamma(rank_x, rank_y))

                    # Blomqvist's beta does not use ranks, so pass original data
                    results["blomqvists_beta"].append(self.blomqvist_beta(x, y))

            except Exception as e:
                log.warning(f"Error computing correlations for param {param}: {e}")
                # Append NaN to all lists on failure
                for key in results:
                    if (
                        key in ["rho", "footrule", "ginis_gamma", "blomqvists_beta"]
                        and not compute_all
                    ):
                        continue
                    results[key].append(np.nan)

        return CorrelationData(
            params=param_values,
            xi=np.array(results["xi"]),
            rho=np.array(results["rho"]) if compute_all else None,
            tau=np.array(results["tau"]) if compute_all else None,
            footrule=np.array(results["footrule"]) if compute_all else None,
            ginis_gamma=np.array(results["ginis_gamma"]) if compute_all else None,
            blomqvists_beta=(
                np.array(results["blomqvists_beta"]) if compute_all else None
            ),
        )

    def _plot_curves(
        self, data: CorrelationData, log_scale: bool, plot_all: bool, label_prefix: str
    ) -> Dict[str, CubicSpline]:
        """Plot correlation data points and their cubic spline interpolations."""
        plot_config = {
            "xi": ("Chatterjee's xi", "o"),
            "rho": ("Spearman's rho", "^"),
            "tau": ("Kendall's tau", "s"),
            "footrule": ("Footrule", "x"),
            "ginis_gamma": ("Gini's Gamma", "d"),
            "blomqvists_beta": ("Blomqvist's Beta", "*"),
        }
        if not plot_all:
            plot_config = {"xi": ("$\\xi$", "o")}

        splines = {}
        inf = float(self.copul.intervals[str(self.copul.params[0])].inf)

        for name, (label, marker) in plot_config.items():
            y_values = getattr(data, name)
            if y_values is None:
                continue

            mask = ~np.isnan(y_values)
            x, y = data.params[mask], y_values[mask]

            if len(x) < 2:
                continue

            # Adjust x-axis for log scale with non-zero infimum
            x_plot = x - inf if log_scale and inf != 0.0 else x

            full_label = f"{label_prefix} {label}".strip()
            plt.scatter(x_plot, y, label=full_label, marker=marker)

            # Create and plot spline
            cs = CubicSpline(x, y)
            splines[name] = cs

            if log_scale:
                x_dense = self._get_dense_log_x_values(inf)
                x_dense_plot = x_dense - inf
            else:
                x_dense = np.linspace(x.min(), x.max(), 500)
                x_dense_plot = x_dense

            plt.plot(x_dense_plot, cs(x_dense))

        if log_scale:
            self._format_log_x_axis(inf, x_plot)

        return splines

    def _finalize_plot(
        self,
        params: Optional[Dict[str, Any]],
        ylim: Tuple[float, float],
        is_mixed: bool,
    ) -> None:
        """Add final titles, labels, and grid to the plot, then show and save it."""
        plt.legend()

        # Determine the varying parameter for the x-axis label
        all_params = {str(p) for p in self.copul.params}
        fixed_params = set(params.keys()) if params else set()
        varying_params = all_params - fixed_params
        x_param = varying_params.pop() if varying_params else self.copul.params[0]

        legend_suffix = self._generate_legend_suffix(fixed_params)
        plt.xlabel(f"$\\{x_param}${legend_suffix}")

        # Set y-axis label and limits
        plt.ylabel("Correlation")
        if is_mixed:
            plt.ylabel(r"$\xi$")
            plt.ylim(0, 1)
        else:
            plt.ylim(*ylim)

        title = CopulaGraphs(self.copul, False).get_copula_title()
        plt.title(title)
        plt.grid(True)
        plt.savefig(self.images_dir / f"{title}_rank_correlations.png")
        plt.show()

    def _format_log_x_axis(self, inf: float, x_plot: np.ndarray) -> None:
        """Set x-axis to log scale and format ticks if infimum is non-zero."""
        plt.xscale("log")
        if inf != 0.0:
            ticks = plt.xticks()[0]
            infimum_str = f"{int(inf)}" if inf.is_integer() else f"{inf:.2f}"
            new_labels = [
                f"${infimum_str} + 10^{{{int(np.log10(t))}}}$" for t in ticks if t > 0
            ]
            valid_ticks = [t for t in ticks if t > 0]
            plt.xticks(valid_ticks, new_labels)
            plt.xlim(x_plot.min(), x_plot.max())

    def _generate_legend_suffix(self, fixed_params: set) -> str:
        """Generate a legend suffix showing constant parameter values."""
        const_params = {*self.copul.intervals} - {
            str(p) for p in self.copul.params
        } | fixed_params
        if not const_params:
            return ""

        parts = []
        for p in const_params:
            val = getattr(self.copul, p, None) or self.copul.defaults.get(p)
            if val is not None:
                val_str = f"\\{val}" if isinstance(val, (property, str)) else f"{val}"
                parts.append(f"$\\{p}={val_str}$")

        return f" (with {', '.join(parts)})" if parts else ""

    @staticmethod
    def _format_param_label(param_dict: Dict[str, Any]) -> str:
        """Format parameter dictionary for a plot label."""
        return ", ".join(
            f"$\\{k}=\\{v}$" if isinstance(v, (property, str)) else f"$\\{k}={v}$"
            for k, v in param_dict.items()
        )

    def _get_parameter_values(
        self, copula: Any, n_params: int, log_scale: bool
    ) -> np.ndarray:
        """Generate an array of parameter values over the copula's valid interval."""
        if hasattr(copula, "get_params"):
            return copula.get_params(n_params, log_scale=log_scale)
        return self.get_params(n_params, log_scale)

    def _save_data(
        self, copula: Any, data: CorrelationData, splines: Dict[str, CubicSpline]
    ):
        """Save computed correlation data and spline objects to files."""
        try:
            base_name = CopulaGraphs(copula, False).get_copula_title()
            with open(self.functions_dir / f"{base_name}_data.pkl", "wb") as f:
                pickle.dump(data, f)
            with open(self.functions_dir / f"{base_name}_splines.pkl", "wb") as f:
                pickle.dump(splines, f)
        except Exception as e:
            log.warning(f"Failed to save data: {e}")

    @staticmethod
    def _mix_params(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameter values for multiple plots."""
        if not params:
            return []

        keys = list(params.keys())
        value_lists = [v if isinstance(v, list) else [v] for v in params.values()]

        combinations = list(itertools.product(*value_lists))
        return [dict(zip(keys, combo)) for combo in combinations]

    def get_params(self, n_params: int, log_scale: bool = False) -> np.ndarray:
        """Generate parameter values based on the copula's defined interval."""
        interval = self.copul.intervals[str(self.copul.params[0])]
        if isinstance(interval, sympy.FiniteSet):
            return np.array([float(val) for val in interval])

        inf, sup = float(interval.inf), float(interval.sup)

        if log_scale:
            # base = max(1e-4, inf)
            if isinstance(self.log_cut_off, tuple):
                return np.logspace(*self.log_cut_off, n_params) + inf
            return np.logspace(-self.log_cut_off, self.log_cut_off, n_params) + inf

        cut_off = self.log_cut_off or self.xlim
        if isinstance(cut_off, tuple):
            left, right = max(inf, cut_off[0]), min(sup, cut_off[1])
        else:
            left, right = max(-cut_off, inf), min(cut_off, sup)

        if getattr(interval, "left_open", False):
            left += 1e-2
        if getattr(interval, "right_open", False):
            right -= 1e-2

        return np.linspace(left, right, n_params)

    def _get_dense_log_x_values(self, left_boundary: float) -> np.ndarray:
        """Generate dense x values for a smooth curve on a logarithmic scale."""
        if isinstance(self.log_cut_off, tuple):
            return np.logspace(*self.log_cut_off, 500) + left_boundary
        return np.logspace(-self.log_cut_off, self.log_cut_off, 500) + left_boundary

    @staticmethod
    def spearman_footrule(rank_x: np.ndarray, rank_y: np.ndarray):
        """Calculate Spearman's footrule coefficient from pre-computed ranks."""
        n = len(rank_x)
        d_sum = np.sum(np.abs(rank_x - rank_y))
        return 1 + (3.0 / n) - (3.0 / (n * n)) * d_sum

    @staticmethod
    def ginis_gamma(rank_x: np.ndarray, rank_y: np.ndarray):
        """Calculate Gini's gamma correlation from pre-computed ranks."""
        n = len(rank_x)
        # Convert ranks to pseudo-observations
        u = rank_x / n
        v = rank_y / n

        integral_1 = np.mean(1 - np.maximum(u, v))
        integral_2 = np.mean(np.maximum(0, 1 - u - v))
        return 4 * (integral_1 + integral_2) - 2

    @staticmethod
    def blomqvist_beta(x: np.ndarray, y: np.ndarray):
        """Calculate Blomqvist's beta correlation."""
        n = len(x)
        med_x, med_y = np.median(x), np.median(y)
        quad_agree = np.sum(((x <= med_x) & (y <= med_y)) | ((x > med_x) & (y > med_y)))
        return 2.0 * quad_agree / n - 1.0


def plot_rank_correlations(
    copula: Any,
    n_obs: int = 10_000,
    n_params: int = 20,
    params: Optional[Dict[str, Any]] = None,
    xlim: Optional[Tuple[float, float]] = 10,
    ylim: Tuple[float, float] = (-1, 1),
    log_cut_off: Optional[Union[float, Tuple[float, float]]] = None,
    approximate: bool = False,
) -> None:
    """
    Convenience function to plot rank correlations for a copula.

    Args:
        copula: Copula object to analyze.
        n_obs: Number of observations per point.
        n_params: Number of parameter values to evaluate.
        params: Dictionary of parameter values to mix.
        xlim: X-axis limits for the plot.
        ylim: Y-axis limits.
        log_cut_off: Cut-off value(s) for logarithmic scale.
        approximate: Whether to use approximate sampling.
    """
    plotter = RankCorrelationPlotter(copula, log_cut_off, approximate, xlim)
    plotter.plot_rank_correlations(n_obs, n_params, params, ylim)


if __name__ == "__main__":
    # Example usage
    import copul

    families = [e.name for e in copul.family_list.Families]

    params_dict = {
        "CLAYTON": {"log_cut_off": (-1.5, 1.5)},
        "NELSEN1": {"log_cut_off": (-1.5, 1.5)},
        "BIV_CLAYTON": {"log_cut_off": (-1.5, 1.5)},
        "NELSEN2": {"log_cut_off": (-1.5, 1.5)},
        "FRANK": {"xlim": (-20, 20)},
        "JOE": {"log_cut_off": (-1.5, 1.5)},
        "NELSEN8": {"log_cut_off": (-2, 3)},
        "NELSEN13": {"log_cut_off": (-2, 2)},
        "GENEST_GHOUDI": {"log_cut_off": (-2, 1)},
        "NELSEN16": {"log_cut_off": (-3.5, 3.5)},
        "NELSEN18": {"log_cut_off": (-2, 2)},
        "NELSEN21": {"log_cut_off": (-2, 1)},
        "BB5": {"params": {"theta": 2}, "log_cut_off": (-1.5, 1.5), "ylim": (0, 1)},
        "GALAMBOS": {"log_cut_off": (-1, 1)},
        "GUMBEL_HOUGAARD": {"log_cut_off": (-1, 1)},
        "HUESLER_REISS": {"log_cut_off": (-1.5, 1.5)},
        "JOEEV": {"params": {"alpha_1": 0.9, "alpha_2": 0.9}, "log_cut_off": (-1, 2)},
        "TAWN": {"params": {"alpha_1": 0.9, "alpha_2": 0.9}, "log_cut_off": (-2, 2)},
        "PLACKETT": {"log_cut_off": (-3, 3)},
    }

    main_families = ["NELSEN1", "FRANK", "GUMBEL_HOUGAARD", "JOE", "GAUSSIAN"]
    main_families = ["JOE", "GUMBEL_HOUGAARD", "GAUSSIAN"]
    # main_families = ["GAUSSIAN"]
    for family in main_families:
        print(f"Plotting rank correlations for {family} copula...")
        copula_class = copul.family_list.Families.create(family)
        run_params = params_dict.get(family, {})

        # Instantiate copula with fixed params if they exist
        if "params" in run_params:
            copula_instance = copula_class(**run_params.pop("params"))
        else:
            copula_instance = copula_class()

        # Call the standalone function with the copula instance
        plot_rank_correlations(
            copula=copula_instance,
            n_obs=1_000_000,
            n_params=50,
            approximate=False,
            **run_params,
        )
