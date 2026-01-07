#!/usr/bin/env python3
"""
Minimal, extensible rank-correlation plotting for copulas.

How to add a new measure:
-------------------------
1) Define a function with signature: f(x, y, rank_x, rank_y) -> float
2) Decorate it with @measure("Nice Name")
   (that's all; it will show up automatically in compute/plot/export)

Built-ins: xi, rho, tau, footrule, gini_gamma, blomqvist_beta
"""

from __future__ import annotations

import logging
import pathlib
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import sympy as sp

# If you have it in your project:
from copul.chatterjee import xi_ncalculate

log = logging.getLogger(__name__)

# ------------------------ Measure Registry ------------------------

MeasureFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
MEASURES: Dict[str, MeasureFn] = {}


def measure(name: str) -> Callable[[MeasureFn], MeasureFn]:
    """Decorator to register a rank-correlation-like measure."""

    def _wrap(fn: MeasureFn) -> MeasureFn:
        MEASURES[name] = fn
        return fn

    return _wrap


# ------------------------ Built-in measures ------------------------


@measure("chatterjees_xi")
def m_xi(x, y, rx, ry) -> float:
    return float(xi_ncalculate(x, y))


@measure("spearmans_rho")
def m_rho(x, y, rx, ry) -> float:
    # Pearson correlation of ranks = Spearman's rho
    r, _ = st.pearsonr(rx, ry)
    return float(r)


@measure("kendalls_tau")
def m_tau(x, y, rx, ry) -> float:
    t, _ = st.kendalltau(x, y)
    return float(t)


@measure("spearmans_footrule")
def m_footrule(x, y, rx, ry) -> float:
    n = len(rx)
    d = np.abs(rx - ry).sum()
    return 1.0 + 3.0 / n - (3.0 / (n * n)) * d


@measure("gini_gamma")
def m_gini_gamma(x, y, rx, ry) -> float:
    n = len(rx)
    u, v = rx / n, ry / n
    integral_1 = np.mean(1 - np.maximum(u, v))
    integral_2 = np.mean(np.maximum(0.0, 1.0 - u - v))
    return 4.0 * (integral_1 + integral_2) - 2.0


@measure("blomqvist_beta")
def m_blomqvist_beta(x, y, rx, ry) -> float:
    n = len(x)
    med_x, med_y = np.median(x), np.median(y)
    agree = np.sum(((x <= med_x) & (y <= med_y)) | ((x > med_x) & (y > med_y)))
    return 2.0 * agree / n - 1.0


@measure("blests_nu")
def m_nu(x, y, rx, ry) -> float:
    """
    Blest's rank correlation ν via the empirical copula plug-in.

    Using C_n(u,v) = (1/n) Σ 1{R_j/n ≤ u, S_j/n ≤ v} with ranks R,S∈{1,…,n},
    we get
        ∫∫ (1-u) C_n(u,v) du dv
      = (1/n) Σ [ ∫_{u=R_j/n}^1 (1-u) du ] [ ∫_{v=S_j/n}^1 dv ]
      = (1/n) Σ [ (1 - R_j/n)^2 / 2 ] [ 1 - S_j/n ].

    Hence the estimator:
        ν̂ = 24 * ( (1/n) Σ ((1 - R_j/n)^2 / 2) * (1 - S_j/n) ) - 2.
    """
    n = len(rx)
    u = rx / n  # rx,ry are 1..n (from scipy.stats.rankdata)
    v = ry / n
    II = np.mean(((1.0 - u) ** 2) * (1.0 - v) / 2.0)
    return float(24.0 * II - 2.0)


# ------------------------ Data container ------------------------


@dataclass
class CorrelationData:
    params: np.ndarray
    values: Dict[str, np.ndarray]  # measure_name -> values (aligned with params)


# ------------------------ Runner ------------------------


class RankCorrelationPlotter:
    """
    Minimal runner:
      - param grid
      - sample once per param
      - compute registered measures
      - plot & export
    """

    def __init__(
        self,
        copula: Any,
        *,
        measures: Optional[Iterable[str]] = None,
        images_dir: pathlib.Path | str = "images",
        save_pickles: bool = True,
    ):
        self.copula = copula
        self.measures = list(measures) if measures else list(MEASURES)
        self.images_dir = pathlib.Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self.images_dir / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._splines_dir = self.images_dir / "splines"
        self._splines_dir.mkdir(parents=True, exist_ok=True)
        self.save_pickles = save_pickles

    def param_grid(
        self,
        n_params: int = 20,
        *,
        log_cut_off: Optional[Tuple[float, float]] = None,
        xlim: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Build a grid over the primary parameter interval in the copula.
        If log_cut_off is provided, use a log grid *shifted by the left boundary (inf)*,
        matching the behavior of the previous implementation.
        """
        # Prefer a family-provided helper if it exists
        if hasattr(self.copula, "get_params"):
            return self.copula.get_params(n_params, log_scale=bool(log_cut_off))

        # Fallback: use interval of the first parameter
        p = str(self.copula.params[0])
        interval = self.copula.intervals[p]

        # Support FiniteSet (discrete parameter sets)
        if isinstance(interval, sp.FiniteSet):
            return np.array([float(val) for val in interval])

        inf = float(interval.inf)
        sup = float(interval.sup)

        # Respect open bounds
        left = inf + (1e-2 if getattr(interval, "left_open", False) else 0.0)
        right = sup - (1e-2 if getattr(interval, "right_open", False) else 0.0)

        if log_cut_off is not None:
            # log grid shifted by inf: inf + 10^a ... inf + 10^b  (or symmetric if scalar)
            if isinstance(log_cut_off, tuple):
                lo, hi = log_cut_off
                return np.logspace(lo, hi, n_params) + inf
            else:
                s = float(log_cut_off)
                return np.logspace(-s, s, n_params) + inf

        # Linear grid (with optional numeric crop)
        if xlim is not None:
            left = max(left, xlim[0])
            right = min(right, xlim[1])

        return np.linspace(left, right, n_params)

    # ---- compute ----
    def compute(
        self,
        params: np.ndarray,
        n_obs: int = 1_000_000,
        *,
        approximate: bool = False,
    ) -> CorrelationData:
        """
        For each parameter:
          - sample (x,y)
          - compute ranks once
          - evaluate each registered measure
        """
        vals: Dict[str, List[float]] = {m: [] for m in self.measures}

        pname = str(self.copula.params[0])
        for theta in params:
            try:
                c = self.copula(**{pname: theta})
                data = c.rvs(n_obs, approximate=approximate)
                x, y = data[:, 0], data[:, 1]

                # ranks once
                rx = st.rankdata(x)
                ry = st.rankdata(y)

                # measures
                for m in self.measures:
                    fn = MEASURES[m]
                    vals[m].append(fn(x, y, rx, ry))
            except Exception as e:
                log.warning("Param %g failed: %s", theta, e)
                for m in self.measures:
                    vals[m].append(np.nan)

        return CorrelationData(
            params=params, values={k: np.array(v) for k, v in vals.items()}
        )

    def plot(
        self,
        data: "CorrelationData",
        *,
        title: Optional[str] = None,
        ylim: Tuple[float, float] = (-1, 1),
        log_x: bool = False,
        log_cut_off: Optional[Tuple[float, float]] = None,  # <— NEW
    ) -> Dict[str, CubicSpline]:
        """
        Scatter + CubicSpline for each measure.
        If log_x=True, we plot against x - inf, set log scale, and clamp the axis
        to the exact range implied by log_cut_off (if provided).
        """
        if title is None:
            from copul.family.copula_graphs import CopulaGraphs

            title = CopulaGraphs(self.copula, False).get_copula_title()

        # interval and left boundary (inf) of the first parameter
        p = str(self.copula.params[0])
        interval = self.copula.intervals[p]
        inf = float(getattr(interval, "inf", 0.0))

        splines: Dict[str, CubicSpline] = {}
        x = data.params

        for m, y in data.values.items():
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                continue

            x_masked = x[mask]
            y_masked = y[mask]

            # Fit spline in ORIGINAL x-domain (not shifted)
            cs = CubicSpline(x_masked, y_masked)
            xs_dense = np.linspace(x_masked.min(), x_masked.max(), 600)
            ys_dense = cs(xs_dense)

            if log_x:
                x_plot_pts = x_masked - inf
                xs_plot = xs_dense - inf
                plt.scatter(x_plot_pts, y_masked, s=16, label=m)
                plt.plot(xs_plot, ys_dense)
            else:
                plt.scatter(x_masked, y_masked, s=16, label=m)
                plt.plot(xs_dense, ys_dense)

            splines[m] = cs

        plt.title(title)
        plt.xlabel(f"${{{self.copula.params[0]}}}$")
        plt.ylabel("Correlation")
        plt.ylim(*ylim)

        if log_x:
            self._format_log_x_axis(
                inf, x, log_cut_off
            )  # <— pass both x and the cutoffs

        ax = plt.gca()
        # after drawing the curves and before saving/showing:
        if log_x:
            # ensure the data were plotted against (x - inf) on a log axis
            self._set_log_ticks(ax, inf, log_cut_off)

        plt.grid(True)
        plt.legend()
        out = self.images_dir / f"{title}_rank_correlations.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        return splines

    def _format_log_x_axis(
        self,
        inf: float,
        x_original: np.ndarray,
        log_cut_off: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Apply log x-scale to the *shifted* axis (x - inf).
        If log_cut_off=(a,b) is given, clamp to [10^a, 10^b] exactly.
        Format ticks as 'inf + 10^{k}' when inf != 0.
        """
        plt.xscale("log")

        # Set limits precisely from log_cut_off if provided
        if log_cut_off is not None:
            a, b = (
                log_cut_off
                if isinstance(log_cut_off, tuple)
                else (-log_cut_off, log_cut_off)
            )
            xmin, xmax = 10.0**a, 10.0**b
            plt.xlim(xmin, xmax)
        else:
            # fallback to data-driven limits in shifted domain
            x_plot = x_original - inf
            x_plot = x_plot[np.isfinite(x_plot) & (x_plot > 0)]
            if x_plot.size:
                plt.xlim(x_plot.min(), x_plot.max())

    @staticmethod
    def _set_log_ticks(
        ax, inf: float, log_cut_off: Optional[Tuple[float, float]] = None
    ):
        """
        On a shifted-log x-axis (x' = x - inf), put ticks at 10^k and label them
        as 'inf + 10^{k}'. This never creates ticks outside the visible range.
        """
        ax.set_xscale("log")

        # 1) Determine visible range on the shifted axis
        if log_cut_off is not None:
            a, b = (
                log_cut_off
                if isinstance(log_cut_off, tuple)
                else (-log_cut_off, log_cut_off)
            )
            xmin, xmax = 10.0**a, 10.0**b
            ax.set_xlim(xmin, xmax)
        else:
            xmin, xmax = ax.get_xlim()

        # 2) Choose integer exponents fully inside [xmin, xmax]
        k_lo = int(np.ceil(np.log10(max(xmin, np.finfo(float).tiny))))
        k_hi = int(np.floor(np.log10(xmax)))
        if k_lo > k_hi:  # degenerate range; just bail with defaults
            return

        ticks = 10.0 ** np.arange(k_lo, k_hi + 1)

        # 3) Set ticks and labels without changing limits
        if inf == 0.0:
            labels = [rf"$10^{{{k}}}$" for k in range(k_lo, k_hi + 1)]
        else:
            inf_str = f"{int(inf)}" if float(inf).is_integer() else f"{inf:.2f}"
            labels = [rf"${inf_str} + 10^{{{k}}}$" for k in range(k_lo, k_hi + 1)]

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    # ---- save ----
    def save(
        self, base_name: str, data: CorrelationData, splines: Dict[str, CubicSpline]
    ) -> None:
        if not self.save_pickles:
            return
        try:
            with open(self._data_dir / f"{base_name}_data.pkl", "wb") as f:
                pickle.dump(data, f)
            with open(self._splines_dir / f"{base_name}_splines.pkl", "wb") as f:
                pickle.dump(splines, f)
        except Exception as e:
            log.warning("Failed to save pickles: %s", e)


# ------------------------ Convenience API ------------------------


def run_plot(
    copula: Any,
    *,
    measures: Optional[Iterable[str]] = None,
    n_obs: int = 10_000,
    n_params: int = 20,
    log_cut_off: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Tuple[float, float] = (-1, 1),
    approximate: bool = False,
    images_dir: str | pathlib.Path = "images",
    save_pickles: bool = True,
):
    runner = RankCorrelationPlotter(
        copula, measures=measures, images_dir=images_dir, save_pickles=save_pickles
    )
    params = runner.param_grid(
        n_params=n_params,
        log_cut_off=log_cut_off,
        xlim=xlim,
    )
    data = runner.compute(params, n_obs=n_obs, approximate=approximate)
    splines = runner.plot(
        data,
        title=None,
        ylim=ylim,
        log_x=bool(log_cut_off),
        log_cut_off=log_cut_off,  # <— add this
    )
    # Title again (to name files consistently)
    from copul.family.copula_graphs import CopulaGraphs

    base = CopulaGraphs(copula, False).get_copula_name()
    runner.save(base, data, splines)


# ------------------------ Example ------------------------

if __name__ == "__main__":
    import copul
    from pathlib import Path

    # Families we want to run
    main_families = ["NELSEN1", "FRANK", "GUMBEL_HOUGAARD", "JOE"]  # , "GAUSSIAN"]
    # main_families = ["GAUSSIAN"]

    # Per-family settings (log grids, x-lims, fixed params, etc.)
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

    # Measures to show (registered via @measure); include ν ("nu")
    measures = [
        "chatterjees_xi",
        "blests_nu",
        "spearmans_rho",
        "kendalls_tau",
        "spearmans_footrule",
        "gini_gamma",
        "blomqvist_beta",
    ]

    images_dir = Path("images")
    images_dir.mkdir(parents=True, exist_ok=True)

    for family in main_families:
        print(f"Plotting rank correlations for {family} copula...")
        copula_class = copul.family_list.Families.create(family)

        # Pull per-family settings
        run_params = dict(params_dict.get(family, {}))  # shallow copy

        # Instantiate with any fixed constructor params, if given
        ctor_params = run_params.pop("params", None)
        copula_instance = copula_class(**ctor_params) if ctor_params else copula_class()

        # Hand remaining plotting/grid kwargs to the new API
        run_plot(
            copula=copula_instance,
            measures=measures,
            n_obs=2_000_000,
            n_params=200,
            approximate=False,
            images_dir=images_dir,
            save_pickles=True,
            **run_params,  # e.g. log_cut_off, xlim, ylim
        )
