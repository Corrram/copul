"""
Module for plotting rank correlations of copulas.

This module provides tools to visualize various rank correlation measures
for copulas with different parameter settings.
"""

import itertools
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import sympy
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from copul.chatterjee import xi_ncalculate, xi_nvarcalculate
from copul.families.copula_graphs import CopulaGraphs

# Set up logger
log = logging.getLogger(__name__)


@dataclass
class CorrelationData:
    """
    Class to store correlation data for various metrics.

    Attributes:
        params: Parameter values used for computation
        xi: Chatterjee's xi correlation values
        xi_var: Variance of xi estimates (if computed)
        rho: Spearman's rho correlation values
        tau: Kendall's tau correlation values
    """

    params: np.ndarray
    xi: np.ndarray
    xi_var: Optional[np.ndarray] = None
    rho: Optional[np.ndarray] = None
    tau: Optional[np.ndarray] = None
    footrule: Optional[np.ndarray] = None
    ginis_gamma: Optional[np.ndarray] = None
    blomqvists_beta: Optional[np.ndarray] = None

    def with_error_bands(self, n_obs: int) -> Dict[str, np.ndarray]:
        """
        Compute error bands for correlations.

        Args:
            n_obs: Number of observations used to compute correlations

        Returns:
            Dictionary mapping correlation type to error band width
        """
        result = {}
        if self.xi_var is not None:
            result["xi"] = 3.291 * np.sqrt(self.xi_var / n_obs)
        return result


class RankCorrelationPlotter:
    """
    Class for plotting rank correlations of copulas.

    This class provides functionality to plot Chatterjee's xi correlation,
    Spearman's rho, and Kendall's tau for various copula parameter values.
    """

    def __init__(
        self,
        copula: Any,
        log_cut_off: Optional[Union[float, Tuple[float, float]]] = None,
        approximate=True,
    ):
        """
        Initialize RankCorrelationPlotter.

        Args:
            copula: Copula object to analyze
            log_cut_off: Cut-off value(s) for logarithmic scale
                If single value, defines upper bound as 10^cut_off
                If tuple (a, b), defines range as [10^a, 10^b]
                If None, linear scale is used by default
        """
        self.copul = copula
        self.log_cut_off = log_cut_off
        self._approximate = approximate

        # Create directory for saving plots
        self.images_dir = pathlib.Path("images")
        self.functions_dir = self.images_dir / "functions"

    def plot_rank_correlations(
        self,
        n_obs: int = 10_000,
        n_params: int = 20,
        params: Optional[Dict[str, Any]] = None,
        plot_var: bool = False,
        ylim: Tuple[float, float] = (-1, 1),
    ) -> None:
        """
        Plot rank correlations for a given copula with various parameter values.

        Args:
            n_obs: Number of observations for each copula
            n_params: Number of parameter values to evaluate
            params: Dictionary of parameter values to mix
                Keys are parameter names, values can be:
                - Single value
                - List of values
                - String
                - Property
            plot_var: Whether to plot variance bands
            ylim: Y-axis limits for the plot
        """
        log.info(f"Plotting Chatterjee graph for {type(self.copul).__name__} copula")

        # Determine if we should use logarithmic scale
        log_scale = self.log_cut_off is not None

        # Mix parameters if provided
        mixed_params = self._mix_params(params) if params is not None else {}

        # Generate legend suffix
        legend_suffix = self._generate_legend_suffix() if not mixed_params else ""

        # Plot for main parameter if no mixed parameters
        if not mixed_params:
            self._plot_correlation_for(
                n_obs, n_params, self.copul, plot_var, log_scale=log_scale
            )

        # Plot for each mixed parameter set
        for mixed_param in mixed_params:
            try:
                new_copula = self.copul(**mixed_param)
                label = self._format_param_label(mixed_param)
                self._construct_xi_graph_for(
                    n_obs, n_params, new_copula, plot_var, label, log_scale
                )
                plt.ylabel(r"$\xi$")
            except Exception as e:
                log.error(f"Error plotting for parameters {mixed_param}: {e}")

        # Final plot formatting
        self._finalize_plot(params, legend_suffix, ylim, mixed_params)

    def _generate_legend_suffix(self) -> str:
        """
        Generate a legend suffix with constant parameters.

        Returns:
            String with formatted parameter values
        """
        const_params = {*self.copul.intervals} - set(
            {str(param) for param in self.copul.params}
        )
        legend_suffix = ""

        for p in const_params:
            param = getattr(self, str(p))
            if isinstance(param, (property, sympy.Symbol)):
                param_str = f"$\\{p}=\\{param}$"
            else:
                param_str = f"$\\{p}={param}$"

            if not legend_suffix:
                legend_suffix = f" (with {param_str})"
            else:
                legend_suffix += f", {param_str}"

        return legend_suffix.replace("),", ",")

    def _format_param_label(self, param_dict: Dict[str, Any]) -> str:
        """
        Format parameter dictionary for plot label.

        Args:
            param_dict: Dictionary of parameter values

        Returns:
            Formatted string for plot label
        """
        return ", ".join(
            f"$\\{k}=\\{v}$" if isinstance(v, (property, str)) else f"$\\{k}={v}$"
            for k, v in param_dict.items()
        )

    def _finalize_plot(
        self,
        params: Optional[Dict[str, Any]],
        legend_suffix: str,
        ylim: Tuple[float, float],
        mixed_params: List[Dict[str, Any]],
    ) -> None:
        """
        Add final formatting to the plot.

        Args:
            params: Dictionary of mixed parameters
            legend_suffix: Suffix for x-axis label
            ylim: Y-axis limits
            mixed_params: List of mixed parameter dictionaries
        """
        plt.legend()

        # Set x-axis label with the main parameter
        if params is None:
            x_param = self.copul.params[0]
        else:
            try:
                x_param = [
                    param for param in self.copul.params if str(param) not in [*params]
                ][0]
            except IndexError:
                x_param = self.copul.params[0]

        x_label = f"$\\{x_param}${legend_suffix}"
        plt.xlabel(x_label)

        # Set y-axis limits
        plt.ylim(0, 1) if mixed_params else plt.ylim(*ylim)

        # Set title
        title = CopulaGraphs(self.copul, False).get_copula_title()
        plt.title(title)
        plt.grid(True)
        plt.savefig(self.images_dir / f"{title}_rank_correlations.png")
        plt.show()
        plt.draw()

    def _construct_xi_graph_for(
        self,
        n_obs: int,
        n_params: int,
        copula: Any,
        plot_var: bool,
        label: str = r"$\xi$",
        log_scale: bool = False,
    ) -> None:
        """
        Construct and plot Chatterjee's xi correlation graph.

        Args:
            n_obs: Number of observations per point
            n_params: Number of parameter values to evaluate
            copula: Copula object to analyze
            plot_var: Whether to plot variance bands
            label: Label for the plot
            log_scale: Whether to use logarithmic scale for x-axis
        """
        # Get parameter values
        param_values = self._get_parameter_values(copula, n_params, log_scale)

        # Compute correlations
        data_points = self._compute_xi_correlations(
            copula, param_values, n_obs, plot_var
        )

        # Plot results
        self._plot_xi_correlation(data_points, label, log_scale, plot_var, n_obs)

        # Save data for future reference
        self._save_data_and_splines(data_points)

    def _compute_xi_correlations(
        self, copula: Any, param_values: np.ndarray, n_obs: int, compute_var: bool
    ) -> CorrelationData:
        """
        Compute Chatterjee's xi correlations for parameter values.

        Args:
            copula: Copula object to analyze
            param_values: Array of parameter values
            n_obs: Number of observations per point
            compute_var: Whether to compute variance

        Returns:
            CorrelationData object with computed correlations
        """
        xi_values = np.zeros(len(param_values))
        xi_var_values = np.zeros(len(param_values)) if compute_var else None

        for i, param in enumerate(param_values):
            try:
                # Create copula instance with the parameter value
                specific_copula = copula(**{str(copula.params[0]): param})

                # Generate random sample
                data = specific_copula.rvs(n_obs, approximate=self._approximate)

                # Calculate Chatterjee's xi
                xi = xi_ncalculate(data[:, 0], data[:, 1])
                xi_values[i] = xi

                # Calculate variance if requested
                if compute_var and xi_var_values is not None:
                    xi_var = xi_nvarcalculate(data[:, 0], data[:, 1])
                    xi_var_values[i] = xi_var
            except Exception as e:
                log.warning(f"Error computing xi for parameter {param}: {e}")
                xi_values[i] = np.nan
                if compute_var and xi_var_values is not None:
                    xi_var_values[i] = np.nan

        return CorrelationData(param_values, xi_values, xi_var_values)

    def _plot_xi_correlation(
        self,
        data: CorrelationData,
        label: str,
        log_scale: bool,
        plot_var: bool,
        n_obs: int,
    ) -> None:
        """
        Plot xi correlation with optional error bands.

        Args:
            data: Correlation data to plot
            label: Label for the plot
            log_scale: Whether to use logarithmic scale
            plot_var: Whether to plot variance bands
            n_obs: Number of observations (for error bands)
        """
        # Remove NaN values
        mask = ~np.isnan(data.xi)
        x = data.params[mask]
        y = data.xi[mask]

        # Scatter plot of actual data points
        plt.scatter(x, y, label=label)

        # Create spline interpolation if we have enough points
        if len(x) > 1:
            cs = CubicSpline(x, y)

            # Dense x values for smooth curve
            if log_scale:
                left_boundary = float(
                    self.copul.intervals[str(self.copul.params[0])].inf
                )
                x_dense = self._get_dense_log_x_values(left_boundary)
            else:
                x_dense = np.linspace(x.min(), x.max(), 500)

            # Compute y values and plot
            y_dense = cs(x_dense)
            cs_label = "Cubic Spline" if label == r"$\xi$" else None
            plt.plot(x_dense, y_dense, label=cs_label)

            # Set logarithmic scale if requested
            if log_scale:
                plt.xscale("log")

                # Adjust tick labels if non-zero lower bound
                inf = float(self.copul.intervals[str(self.copul.params[0])].inf)
                if log_scale and inf != 0.0:
                    ticks = plt.xticks()[0]
                    infimum = int(inf) if inf.is_integer() else inf
                    new_ticklabels = [
                        f"${infimum} + 10^{{{int(np.log10(t))}}}$" for t in ticks
                    ]
                    plt.xticks(ticks, new_ticklabels)
                    plt.xlim(x[0] - inf, x[-1] - inf)

        # Add error bands if requested
        if plot_var and data.xi_var is not None:
            error_bands = data.with_error_bands(n_obs)
            if "xi" in error_bands:
                y_err = error_bands["xi"][mask]
                plt.fill_between(x, y - y_err, y + y_err, alpha=0.2)

    def _get_dense_log_x_values(self, left_boundary: float) -> np.ndarray:
        """
        Generate dense x values for logarithmic scale.

        Args:
            left_boundary: Lower bound for x values

        Returns:
            Array of x values
        """
        if isinstance(self.log_cut_off, tuple):
            return np.logspace(*self.log_cut_off, 500) + left_boundary
        else:
            return np.logspace(-self.log_cut_off, self.log_cut_off, 500) + left_boundary

    @staticmethod
    def spearman_footrule(x, y):
        """
        Calculates Spearman's footrule-based estimator of ψ(C) = 6 ∫ C(u,u) du − 2.

        Args:
            x (array-like): First variable.
            y (array-like): Second variable.

        Returns:
            float: The Spearman's footrule ψ estimate.
        """
        n = len(x)
        if n != len(y):
            raise ValueError("Input arrays must have the same length.")
        if n < 2:
            return np.nan

        # Calculate ranks
        rank_x = scipy.stats.rankdata(x)
        rank_y = scipy.stats.rankdata(y)

        # Sum of absolute differences
        d_sum = np.sum(np.abs(rank_x - rank_y))

        # ψ(C) finite-sample estimator
        psi = 1 + (3.0 / n) - (3.0 / (n * n)) * d_sum
        return psi

    @staticmethod
    def ginis_gamma(x, y):
        """
        Calculates Gini's gamma correlation estimate based on the empirical copula.

        The formula is:
        γ(C) = 4 ∫[0,1] C(u,u)du + 4 ∫[0,1] C(u,1-u)du - 2

        Args:
            x (array-like): First variable.
            y (array-like): Second variable.

        Returns:
            float: The Gini's gamma estimate.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)

        if n != len(y):
            raise ValueError("Input arrays must have the same length.")
        if n < 2:
            return np.nan

        # Compute pseudo-observations from ranks (empirical copula values)
        u = scipy.stats.rankdata(x, method="ordinal") / n
        v = scipy.stats.rankdata(y, method="ordinal") / n

        # Estimate the two integrals using their non-parametric sample estimators

        # Estimate for ∫ C(u,u)du is mean(1 - max(u_i, v_i))
        integral_1 = np.mean(1 - np.maximum(u, v))

        # Estimate for ∫ C(u,1-u)du is mean(max(0, 1 - u_i - v_i))
        integral_2 = np.mean(np.maximum(0, 1 - u - v))

        gamma = 4 * integral_1 + 4 * integral_2 - 2
        return gamma

    @staticmethod
    def blomqvist_beta(x, y):
        """
        Calculates Blomqvist's beta correlation coefficient.

        Definition: β(C) = 4 C(1/2, 1/2) - 1.

        Empirical version:
            Count proportion of pairs (x_i, y_i)
            that fall into the same quadrant w.r.t. their medians.

        Args:
            x (array-like): First variable.
            y (array-like): Second variable.

        Returns:
            float: Blomqvist's beta estimate.
        """
        n = len(x)
        if n != len(y):
            raise ValueError("Input arrays must have the same length.")
        if n < 2:
            return np.nan

        med_x = np.median(x)
        med_y = np.median(y)

        # Count quadrant agreements
        same_quadrant = np.sum(
            ((x <= med_x) & (y <= med_y)) | ((x > med_x) & (y > med_y))
        )

        beta = 2.0 * same_quadrant / n - 1.0
        return beta

    def _plot_correlation_for(
        self,
        n_obs: int,
        n_params: int,
        copula: Any,
        plot_var: bool,
        log_scale: bool = False,
    ) -> None:
        """
        Plot multiple correlation measures for a copula.

        Args:
            n_obs: Number of observations per point
            n_params: Number of parameter values to evaluate
            copula: Copula object to analyze
            plot_var: Whether to plot variance bands
            log_scale: Whether to use logarithmic scale
        """
        # Get parameter values
        param_values = self.get_params(n_params, log_scale=log_scale)

        # Initialize arrays for results
        xi_values = np.zeros(len(param_values))
        rho_values = np.zeros(len(param_values))
        tau_values = np.zeros(len(param_values))
        footrule_values = np.zeros(len(param_values))
        ginis_gamma_values = np.zeros(len(param_values))
        blomqvists_beta_values = np.zeros(len(param_values))
        xi_var_values = np.zeros(len(param_values)) if plot_var else None

        # Compute correlations for each parameter value
        for i, param in enumerate(param_values):
            log.info(f"Computing correlations for parameter {param}")
            try:
                # Create copula instance with the parameter value
                specific_copula = copula(**{str(copula.params[0]): param})

                # Generate random sample
                data = specific_copula.rvs(n_obs, approximate=self._approximate)

                # Calculate rank correlations
                xi_values[i] = xi_ncalculate(data[:, 0], data[:, 1])
                rho_values[i] = scipy.stats.spearmanr(data[:, 0], data[:, 1])[0]
                tau_values[i] = scipy.stats.kendalltau(data[:, 0], data[:, 1])[0]
                footrule_values[i] = self.spearman_footrule(data[:, 0], data[:, 1])
                ginis_gamma_values[i] = self.ginis_gamma(data[:, 0], data[:, 1])
                blomqvists_beta_values[i] = self.blomqvist_beta(data[:, 0], data[:, 1])

                # Calculate variance if requested
                if plot_var and xi_var_values is not None:
                    xi_var_values[i] = xi_nvarcalculate(data[:, 0], data[:, 1])
            except Exception as e:
                log.warning(f"Error computing correlations for parameter {param}: {e}")
                xi_values[i] = np.nan
                rho_values[i] = np.nan
                tau_values[i] = np.nan
                footrule_values[i] = np.nan
                ginis_gamma_values[i] = np.nan
                blomqvists_beta_values[i] = np.nan
                if plot_var and xi_var_values is not None:
                    xi_var_values[i] = np.nan

        # Create CorrelationData object
        data = CorrelationData(
            param_values,
            xi_values,
            xi_var_values,
            rho_values,
            tau_values,
            footrule_values,
            ginis_gamma_values,
            blomqvists_beta_values,
        )

        # Plot results
        self._plot_all_correlations(data, log_scale, plot_var, n_obs)

    def _plot_all_correlations(
        self, data: CorrelationData, log_scale: bool, plot_var: bool, n_obs: int
    ) -> None:
        """
        Plot all correlation measures.

        Args:
            data: Correlation data to plot
            log_scale: Whether to use logarithmic scale
            plot_var: Whether to plot variance bands
            n_obs: Number of observations (for error bands)
        """
        # Remove NaN values
        mask = ~np.isnan(data.xi)
        x = data.params[mask]
        y_xi = data.xi[mask] if data.xi is not None else None
        y_rho = data.rho[mask] if data.rho is not None else None
        y_tau = data.tau[mask] if data.tau is not None else None
        y_footrule = data.footrule[mask] if data.footrule is not None else None
        y_ginis_gamma = data.ginis_gamma[mask] if data.ginis_gamma is not None else None
        y_blomqvists_beta = (
            data.blomqvists_beta[mask] if data.blomqvists_beta is not None else None
        )

        # Adjust x values if using log scale
        inf = float(self.copul.intervals[str(self.copul.params[0])].inf)
        x_adjusted = x - inf if log_scale and inf != 0.0 else x

        # Plot Rank Correlations
        if y_xi is not None:
            plt.scatter(x_adjusted, y_xi, label="Chatterjee's xi", marker="o")
        if y_rho is not None:
            plt.scatter(x_adjusted, y_rho, label="Spearman's rho", marker="^")
        if y_tau is not None:
            plt.scatter(x_adjusted, y_tau, label="Kendall's tau", marker="s")
        if y_footrule is not None:
            plt.scatter(x_adjusted, y_footrule, label="Footrule", marker="x")
        if y_ginis_gamma is not None:
            plt.scatter(x_adjusted, y_ginis_gamma, label="Gini's Gamma", marker="d")
        if y_blomqvists_beta is not None:
            plt.scatter(
                x_adjusted, y_blomqvists_beta, label="Blomqvist's Beta", marker="*"
            )

        # Create spline interpolations if we have enough points
        if len(x) > 1:
            # Create dense x values
            if log_scale:
                x_dense = self._get_dense_log_x_values(inf)
                x_dense_adjusted = x_dense - inf
            else:
                x_dense = np.linspace(x.min(), x.max(), 500)
                x_dense_adjusted = x_dense

            # Plot Chatterjee's xi spline
            cs_xi = CubicSpline(x, y_xi)
            y_xi_dense = cs_xi(x_dense)
            plt.plot(x_dense_adjusted, y_xi_dense)

            # Plot Spearman's rho spline if available
            if y_rho is not None:
                cs_rho = CubicSpline(x, y_rho)
                y_rho_dense = cs_rho(x_dense)
                plt.plot(x_dense_adjusted, y_rho_dense)

            # Plot Kendall's tau spline if available
            if y_tau is not None:
                cs_tau = CubicSpline(x, y_tau)
                y_tau_dense = cs_tau(x_dense)
                plt.plot(x_dense_adjusted, y_tau_dense)

            if y_footrule is not None:
                cs_footrule = CubicSpline(x, y_footrule)
                y_footrule_dense = cs_footrule(x_dense)
                plt.plot(x_dense_adjusted, y_footrule_dense)

            if y_ginis_gamma is not None:
                cs_ginis_gamma = CubicSpline(x, y_ginis_gamma)
                y_ginis_gamma_dense = cs_ginis_gamma(x_dense)
                plt.plot(x_dense_adjusted, y_ginis_gamma_dense)

            if y_blomqvists_beta is not None:
                cs_blomqvists_beta = CubicSpline(x, y_blomqvists_beta)
                y_blomqvists_beta_dense = cs_blomqvists_beta(x_dense)
                plt.plot(x_dense_adjusted, y_blomqvists_beta_dense)

            # Set logarithmic scale if requested
            if log_scale:
                plt.xscale("log")

                # Adjust tick labels if non-zero lower bound
                if inf != 0.0:
                    ticks = plt.xticks()[0]
                    infimum = int(inf) if inf.is_integer() else inf
                    new_ticklabels = [
                        f"${infimum} + 10^{{{int(np.log10(t))}}}$" for t in ticks
                    ]
                    plt.xticks(ticks, new_ticklabels)
                    plt.xlim(x_adjusted[0], x_adjusted[-1])

            # Add error bands if requested
            if plot_var and data.xi_var is not None:
                error_bands = data.with_error_bands(n_obs)
                if "xi" in error_bands:
                    y_err = error_bands["xi"][mask]
                    plt.fill_between(x_adjusted, y_xi - y_err, y_xi + y_err, alpha=0.2)

            # Save the data and splines
            self._save_data_and_splines(cs_xi, data)

    def _get_parameter_values(
        self, copula: Any, n_params: int, log_scale: bool
    ) -> np.ndarray:
        """
        Get parameter values for evaluation.

        Args:
            copula: Copula object
            n_params: Number of parameter values
            log_scale: Whether to use logarithmic scale

        Returns:
            Array of parameter values
        """
        # Use copula's get_params method if available
        if hasattr(copula, "get_params"):
            return copula.get_params(n_params, log_scale=log_scale)

        # Otherwise, use our own implementation
        return self.get_params(n_params, log_scale=log_scale)

    def _save_data_and_splines(self, cs, data_points=None):
        """
        Save data and splines to files.

        Args:
            cs: CubicSpline object or data to save
            data_points: Optional data points to save
        """
        try:
            # Create directory if it doesn't exist
            self.functions_dir.mkdir(exist_ok=True, parents=True)

            # Determine file name base
            class_name = self.copul.__class__.__name__

            # Save cubic spline if provided
            if isinstance(cs, CubicSpline) and data_points is not None:
                file_path = self.functions_dir / f"{class_name}.pkl"
                with open(file_path, "wb") as f:
                    import pickle

                    pickle.dump(cs, f)

                # Save data points separately
                data_file_path = self.functions_dir / f"{class_name}Data.pkl"
                with open(data_file_path, "wb") as f:
                    import pickle

                    pickle.dump(data_points, f)
            else:
                # Handle case where only one argument is provided
                file_path = self.functions_dir / f"{class_name}Data.pkl"
                with open(file_path, "wb") as f:
                    import pickle

                    pickle.dump(cs, f)

        except Exception as e:
            log.warning(f"Failed to save data: {e}")

    @staticmethod
    def _mix_params(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Mix parameters for multiple plots.

        Args:
            params: Dictionary of parameters to mix
                Keys are parameter names, values can be:
                - Single value
                - List of values
                - String
                - Property

        Returns:
            List of parameter dictionaries for all combinations
        """
        if not params:
            return [{}]

        # Find keys that should be included in cross product
        cross_prod_keys = [
            key
            for key, value in params.items()
            if isinstance(value, (list, str, property))
        ]

        # If no keys have multiple values, return the original parameters
        if not cross_prod_keys:
            return [params]

        # Extract values for cross product
        values_to_cross = [
            params[key] if isinstance(params[key], list) else [params[key]]
            for key in cross_prod_keys
        ]

        # Generate all combinations
        cross_prod = list(itertools.product(*values_to_cross))

        # Create parameter dictionaries
        return [dict(zip(cross_prod_keys, combo)) for combo in cross_prod]

    def get_params(self, n_params: int, log_scale: bool = False) -> np.ndarray:
        """
        Generate parameter values for evaluation.

        Args:
            n_params: Number of parameter values
            log_scale: Whether to use logarithmic scale

        Returns:
            Array of parameter values
        """
        # Get interval for main parameter
        interval = self.copul.intervals[str(self.copul.params[0])]

        # Handle finite sets (discrete parameters)
        if isinstance(interval, sympy.FiniteSet):
            return np.array([float(val) for val in interval])

        # Get cut-off value
        cut_off = self.log_cut_off if log_scale else 10

        # Get bounds from interval
        inf = float(interval.inf)
        sup = float(interval.sup)

        # Handle logarithmic scale
        if log_scale:
            # For log scale, we need a positive base to add to the logspace
            # If inf is negative, we'll use a small positive value instead
            base = max(0.0001, inf)

            if isinstance(cut_off, tuple):
                return np.logspace(*cut_off, n_params) + base
            else:
                return np.logspace(-cut_off, cut_off, n_params) + base

        # Handle linear scale with tuple cut-off
        if isinstance(cut_off, tuple):
            left_border = float(max(inf, cut_off[0]))
            right_border = float(min(sup, cut_off[1]))
        else:
            # Handle linear scale with numeric cut-off
            left_border = float(max(-cut_off, inf))
            right_border = float(min(cut_off, sup))

        # Handle open intervals
        if hasattr(interval, "left_open") and interval.left_open:
            left_border += 0.01
        if hasattr(interval, "right_open") and interval.right_open:
            right_border -= 0.01

        # Create linearly spaced parameter values
        return np.linspace(left_border, right_border, n_params)


def plot_rank_correlations(
    copula: Any,
    n_obs: int = 10_000,
    n_params: int = 20,
    params: Optional[Dict[str, Any]] = None,
    plot_var: bool = False,
    ylim: Tuple[float, float] = (-1, 1),
    log_cut_off: Optional[Union[float, Tuple[float, float]]] = None,
    approximate=False,
) -> None:
    """
    Convenience function to plot rank correlations for a copula.

    Args:
        copula: Copula object to analyze
        n_obs: Number of observations per point
        n_params: Number of parameter values to evaluate
        params: Dictionary of parameter values to mix
        plot_var: Whether to plot variance bands
        ylim: Y-axis limits
        log_cut_off: Cut-off value(s) for logarithmic scale
        approximate: Whether to use approximate sampling via checkerboard copulas
    """
    plotter = RankCorrelationPlotter(copula, log_cut_off, approximate=approximate)
    plotter.plot_rank_correlations(n_obs, n_params, params, plot_var, ylim)


if __name__ == "__main__":
    # Example usage
    import copul

    families = [e.name for e in copul.family_list.Families]

    params_dict = {"Clayton": {"log_cut_off": (-1.5, 1.5)}}
    for family in families[:1]:
        copula = copul.family_list.Families.create(family)
        params = params_dict.get(family, {})
        copula.plot_rank_correlations(
            n_obs=100_000, n_params=50, approximate=True, **params
        )
