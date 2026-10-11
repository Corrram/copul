r"""Paired bootstrap inference for fitted bivariate copula parameters.

Rows, not individual columns, are resampled. In maximum pseudo-likelihood
mode each bootstrap sample is ranked again, propagating estimation of the
unknown continuous margins. Parametric replicates are sampled from the
fitted copula and subjected to the same ranking/estimation procedure.

Percentile intervals use the alpha/2 and 1-alpha/2 empirical quantiles;
basic intervals reflect them around the original estimate. These are
first-order bootstrap procedures, not simultaneous confidence regions or
boundary corrections. See Efron & Tibshirani (1993), An Introduction to
the Bootstrap, ch. 13; Genest, Ghoudi & Rivest (1995), Biometrika 82,
543--552, for copula pseudo-likelihood estimation.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from copul._lazy import pd
from copul.stats._adapters import sample
from copul.stats._utils import _positive_integer, as_data, as_rng
from copul.stats.fitting import FitResult, fit


@dataclass
class BootstrapFitResult:
    """Bootstrap replicates, covariance, intervals and failed-refit details.

    Failed replicates remain NaN rows in ``replicates`` and have messages
    in ``failures``. Intervals computed after explicit ``errors='omit'``
    are conditional on successful refits and can be biased.
    """

    fit: FitResult
    parameter_names: tuple[str, ...]
    replicates: np.ndarray
    method: str
    ci_method: str
    level: float
    failures: dict[int, str]

    @property
    def successful(self):
        """Boolean mask of finite, converged bootstrap refits."""
        return np.all(np.isfinite(self.replicates), axis=1)

    @property
    def n_success(self):
        return int(self.successful.sum())

    @property
    def cov(self):
        """Sample covariance of successful parameter estimates (ddof=1)."""
        return np.atleast_2d(np.cov(self.replicates[self.successful], rowvar=False, ddof=1))

    @property
    def se(self):
        """Bootstrap standard errors keyed by free parameter name."""
        return dict(zip(self.parameter_names, np.sqrt(np.diag(self.cov))))

    @property
    def bias(self):
        """Mean bootstrap estimate minus the original estimate."""
        values = self.replicates[self.successful].mean(axis=0)
        return {p: float(v - self.fit.params[p]) for p, v in zip(self.parameter_names, values)}

    def conf_int(self, level=None, method=None):
        """Pointwise percentile/basic intervals; no silent domain clipping."""
        level = self.level if level is None else level
        method = self.ci_method if method is None else method
        _validate_interval(level, method)
        lo, hi = np.quantile(
            self.replicates[self.successful], [(1 - level) / 2, (1 + level) / 2], axis=0
        )
        if method == "basic":
            theta = np.array([self.fit.params[p] for p in self.parameter_names])
            lo, hi = 2 * theta - hi, 2 * theta - lo
        return {p: (float(a), float(b)) for p, a, b in zip(self.parameter_names, lo, hi)}

    def to_frame(self):
        """Parameter estimates, bootstrap errors, bias and interval endpoints."""
        intervals, se, bias = self.conf_int(), self.se, self.bias
        return pd.DataFrame(
            [
                {
                    "parameter": p,
                    "estimate": self.fit.params[p],
                    "se": se[p],
                    "bias": bias[p],
                    "ci_low": intervals[p][0],
                    "ci_high": intervals[p][1],
                }
                for p in self.parameter_names
            ]
        ).set_index("parameter")


def _validate_interval(level, method):
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be between 0 and 1.")
    if method not in ("percentile", "basic"):
        raise ValueError("ci_method must be 'percentile' or 'basic'.")


def bootstrap_fit(
    family_or_fit,
    data,
    *,
    n_boot=200,
    method="nonparametric",
    ci_method="percentile",
    level=0.95,
    pseudo_obs=None,
    fit_method=None,
    random_state=None,
    errors="raise",
    **fit_options,
):
    """Bootstrap the free parameters of a fixed bivariate copula family.

    Parameters
    ----------
    family_or_fit : class, instance, str or FitResult
        A bivariate family accepted by ``fit``, or an existing fit to these
        same data. Partially fixed parameters remain fixed in every refit.
        This function does not perform family or vine-structure selection.
    data : array_like of shape (n, 2)
        Original data, not separately resampled margins.
    n_boot : int
        At least two bootstrap replicates (use hundreds for final inference).
    method : {"nonparametric", "parametric"}
        Resample rows, or simulate from the fitted copula.
    ci_method : {"percentile", "basic"}
        Pointwise interval construction. Basic intervals can leave the
        parameter domain and are deliberately not clipped.
    level : float
        Confidence level between zero and one.
    pseudo_obs : bool or None
        Re-rank every resample if True; with False use known uniforms
        directly. None uses the supplied fit's convention, or True for a
        new fit. Do not set False for estimated ranks if margin uncertainty
        is intended to be included.
    fit_method : str or None
        Estimator used in every replicate; defaults to the supplied fit's
        method or maximum likelihood for a new fit.
    random_state : int, Generator or None
        Local random state, preserving the global NumPy state.
    errors : {"raise", "omit"}
        Default: raise on the first failed/nonconverged refit. Explicit
        omission retains diagnostics and warns; at least two successes are
        required. No retrying until a desired number of successes is reached.
    **fit_options
        Options for the underlying fitter (for example ``options``).

    Returns
    -------
    BootstrapFitResult
        Original fit, all replicates, covariance, bias and confidence intervals.
    """
    n_boot = _positive_integer(n_boot, "n_boot")
    if n_boot < 2:
        raise ValueError("n_boot must be at least two.")
    _validate_interval(level, ci_method)
    if method not in ("nonparametric", "parametric"):
        raise ValueError("method must be 'nonparametric' or 'parametric'.")
    if errors not in ("raise", "omit"):
        raise ValueError("errors must be 'raise' or 'omit'.")
    if "_light" in fit_options:
        raise ValueError("_light is managed internally by bootstrap_fit.")
    X = as_data(data, min_dim=2, max_dim=2)
    if isinstance(family_or_fit, FitResult):
        result = family_or_fit
        if result.n != len(X):
            raise ValueError("The supplied fit and data must have the same sample size.")
        original_pobs = result.extra.get("pseudo_obs", True)
        if pseudo_obs is not None and bool(pseudo_obs) != original_pobs:
            raise ValueError("pseudo_obs must match the supplied fit; refit to change conventions.")
        pseudo_obs = original_pobs
        if fit_method is not None and fit_method != result.method:
            raise ValueError("fit_method must match the supplied fit.")
        fit_method = result.method
        base = result.extra.get("family_spec")
        if base is None:
            base = type(result.copula)(**result.fixed)
        base = deepcopy(base)
    else:
        pseudo_obs = True if pseudo_obs is None else bool(pseudo_obs)
        fit_method = fit_method or "mle"
        result = fit(family_or_fit, X, method=fit_method, pseudo_obs=pseudo_obs, **fit_options)
        if not isinstance(result, FitResult):
            raise TypeError("bootstrap_fit supports fixed bivariate families, not vine fits.")
        base = deepcopy(result.extra["family_spec"])
    if not result.converged or not result.params:
        raise ValueError("Bootstrap requires a converged fit with free parameters.")
    if not pseudo_obs and np.any((X <= 0) | (X >= 1)):
        raise ValueError("With pseudo_obs=False, data must be in the open unit square.")
    names = tuple(result.params)
    rng = as_rng(random_state)
    replicates = np.full((n_boot, len(names)), np.nan)
    failures = {}
    for b in range(n_boot):
        try:
            Xb = (
                X[rng.integers(0, len(X), len(X))]
                if method == "nonparametric"
                else sample(result.copula, len(X), random_state=rng)
            )
            options = dict(fit_options)
            if fit_method == "mle":
                options.setdefault("start", result.params)
            fitted = fit(base, Xb, method=fit_method, pseudo_obs=pseudo_obs, _light=True, **options)
            values = np.array([fitted.params[p] for p in names])
            if not fitted.converged or not np.all(np.isfinite(values)):
                raise RuntimeError(f"Refit did not converge: {fitted.message}")
            replicates[b] = values
        except Exception as exc:
            if errors == "raise":
                raise RuntimeError(f"Bootstrap replicate {b} failed: {exc}") from exc
            failures[b] = f"{type(exc).__name__}: {exc}"
    if failures:
        warnings.warn(
            f"{len(failures)} of {n_boot} bootstrap refits failed; intervals use only successful refits.",
            stacklevel=2,
        )
    if n_boot - len(failures) < 2:
        raise RuntimeError("At least two successful bootstrap refits are required.")
    return BootstrapFitResult(result, names, replicates, method, ci_method, level, failures)
