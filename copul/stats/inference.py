r"""
Confidence intervals for dependence measures and tests of independence.

* :func:`estimate` -- tidy table of sample estimates with standard errors and
  asymptotic or bootstrap confidence intervals;
* :func:`asymptotic_variance` -- plug-in estimates of the asymptotic
  variance :math:`\sigma^2` in :math:`\sqrt n(\hat\kappa_n - \kappa)\to
  N(0,\sigma^2)` for Kendall's :math:`\tau`, Spearman's :math:`\rho`,
  Chatterjee's :math:`\xi` and Blomqvist's :math:`\beta`;
* :func:`independence_test` -- rank tests of :math:`H_0: C = \Pi` based on
  :math:`\tau`, :math:`\rho`, :math:`\xi`, Hoeffding's :math:`D` and the
  Cramér--von Mises functional of the empirical copula.

References
----------
* Borkowf, C. B. (2002). Computing the nonnull asymptotic variance and the
  asymptotic relative efficiency of Spearman's rank correlation.
  *Comput. Statist. Data Anal.* 39, 271--286.
* Chatterjee, S. (2021). A new coefficient of correlation. *JASA* 116,
  2009--2022.
* Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the
  Bootstrap*. Chapman & Hall.
* Genest, C. and Rémillard, B. (2004). Tests of independence and randomness
  based on the empirical copula process. *Test* 13, 335--369.
* Hoeffding, W. (1948). A class of statistics with asymptotically normal
  distribution. *Ann. Math. Statist.* 19, 293--325.
* Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*
  30, 81--93.
* Lin, Z. and Han, F. (2023). On boosting the power of Chatterjee's rank
  correlation. *Biometrika* 110, 283--299.
* Schmid, F. and Schmidt, R. (2007). Nonparametric inference on multivariate
  versions of Blomqvist's beta and related measures of tail dependence.
  *Metrika* 66, 323--354.
* Segers, J. (2012). Asymptotics of empirical copula processes under
  non-restrictive smoothness assumptions. *Bernoulli* 18, 764--782.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from copul._lazy import pd
from copul.chatterjee import xi_null_variance, xi_nvarcalculate
from copul.measures.registry import _iter_keys, get_measure
from copul.stats._utils import RandomLike, as_data, as_rng, chunk_size
from copul.stats.estimators import (
    cramer_von_mises_independence,
    sample_bkr,
    sample_measure,
    sample_rho,
    sample_tau,
    sample_xi,
)

__all__ = [
    "ASYMPTOTIC_MEASURES",
    "IndependenceTestResult",
    "asymptotic_variance",
    "bootstrap",
    "estimate",
    "independence_test",
]


def _xy(data: Any) -> tuple[np.ndarray, np.ndarray]:
    raw = getattr(data, "data", None) if hasattr(data, "U") else None
    arr = as_data(raw if raw is not None else data, min_dim=2, max_dim=2)
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------------------
# asymptotic variances
# ---------------------------------------------------------------------------


def _tau_h1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r""":math:`\hat h_1(i) = \frac1{n-1}\sum_{j\ne i}\operatorname{sgn}(x_i-x_j)
    \operatorname{sgn}(y_i-y_j)` (chunked, ties give sign 0)."""
    n = x.size
    out = np.empty(n)
    step = chunk_size(n)
    for s in range(0, n, step):
        sx = np.sign(x[s : s + step, None] - x[None, :])
        sy = np.sign(y[s : s + step, None] - y[None, :])
        out[s : s + step] = np.einsum("ij,ij->i", sx, sy)
    return out / (n - 1.0)


def _var_tau(x, y) -> float:
    r"""Hoeffding's U-statistic variance :math:`4\,\mathrm{Var}(h_1)` of
    :math:`\sqrt n\,\tau_n`, estimated by the plug-in of the projection
    :math:`h_1(x,y) = E[\operatorname{sgn}(x-X)\operatorname{sgn}(y-Y)]`."""
    h = _tau_h1(x, y)
    return float(4.0 * np.var(h, ddof=1))


def _var_rho(x, y) -> float:
    r"""Plug-in asymptotic variance of :math:`\sqrt n\,\rho_n`
    (Borkowf, 2002; Schmid & Schmidt, 2007):
    :math:`144\,\mathrm{Var}(g)`, with
    :math:`g_i = \hat U_i\hat V_i + \frac1n\sum_j 1\{\hat U_i\le\hat U_j\}\hat V_j
    + \frac1n\sum_j 1\{\hat V_i\le\hat V_j\}\hat U_j`, computed in
    :math:`O(n\log n)`."""
    n = x.size
    u = stats.rankdata(x, method="max") / n
    v = stats.rankdata(y, method="max") / n

    def tail_mean(a, b):
        # (1/n) sum_j 1{a_i <= a_j} b_j
        order = np.argsort(a, kind="stable")
        a_s, b_s = a[order], b[order]
        suffix = np.concatenate([np.cumsum(b_s[::-1])[::-1], [0.0]])
        idx = np.searchsorted(a_s, a, side="left")
        return suffix[idx] / n

    g = u * v + tail_mean(u, v) + tail_mean(v, u)
    return float(144.0 * np.var(g, ddof=1))


def _var_beta(x, y) -> float:
    r"""Asymptotic variance of :math:`\sqrt n\,\beta_n` (Schmid & Schmidt,
    2007): :math:`16\,\mathrm{Var}\,\mathbb G_C(\tfrac12,\tfrac12)` with

    .. math::

       \mathrm{Var}\,\mathbb G_C(\tfrac12,\tfrac12) = c(1-c) + \frac{a^2+b^2}4
       - (a+b)c + 2ab\bigl(c - \tfrac14\bigr),

    :math:`c = C(\tfrac12,\tfrac12)`, :math:`a = \partial_1 C`,
    :math:`b = \partial_2 C` at :math:`(\tfrac12,\tfrac12)`; the partial
    derivatives are estimated by central differences of :math:`C_n` with
    bandwidth :math:`n^{-1/2}` (Segers, 2012).  Equals 1 under independence.
    """
    n = x.size
    u = stats.rankdata(x) / (n + 1.0)
    v = stats.rankdata(y) / (n + 1.0)

    def cn(a, b):
        return np.count_nonzero((u <= a) & (v <= b)) / n

    h = min(0.25, n**-0.5)
    c = cn(0.5, 0.5)
    a = (cn(0.5 + h, 0.5) - cn(0.5 - h, 0.5)) / (2 * h)
    b = (cn(0.5, 0.5 + h) - cn(0.5, 0.5 - h)) / (2 * h)
    var_g = c * (1 - c) + (a * a + b * b) / 4.0 - (a + b) * c + 2 * a * b * (c - 0.25)
    return float(16.0 * max(var_g, 0.0))


#: measures with an asymptotic (plug-in) variance
ASYMPTOTIC_MEASURES: tuple[str, ...] = ("tau", "rho", "xi", "xi_2", "beta")


def asymptotic_variance(data: Any, key: str, null: bool = False) -> float:
    r"""Estimated asymptotic variance :math:`\sigma^2` of
    :math:`\sqrt n(\hat\kappa_n - \kappa)`.

    Parameters
    ----------
    data : array_like of shape (n, 2) or EmpiricalCopula
    key : {"tau", "rho", "xi", "xi_2", "beta"}
    null : bool
        Return the variance under independence instead:
        :math:`\tau`: :math:`\frac{2(2n+5)}{9(n-1)}` (exact, times :math:`n`;
        Kendall, 1938), :math:`\rho`: :math:`\frac{n}{n-1}`,
        :math:`\xi`: :math:`2/5` (Chatterjee, 2021), :math:`\beta`: 1.

    Notes
    -----
    General (non-null) variances: Kendall's :math:`\tau` -- Hoeffding's
    (1948) U-statistic variance with plug-in projection; Spearman's
    :math:`\rho` -- Borkowf (2002); Chatterjee's :math:`\xi` --
    :func:`copul.chatterjee.xi_nvarcalculate` (Lin & Han, 2023);
    Blomqvist's :math:`\beta` -- Schmid & Schmidt (2007).
    """
    x, y = _xy(data)
    n = x.size
    k = get_measure(key).key
    if k not in ASYMPTOTIC_MEASURES:
        raise ValueError(
            f"No asymptotic variance for {k!r}; available: {ASYMPTOTIC_MEASURES}. "
            "Use ci='bootstrap'."
        )
    if null:
        return {
            "tau": 2.0 * (2 * n + 5) / (9.0 * (n - 1)),
            "rho": n / (n - 1.0),
            "xi": float(xi_null_variance(y)),
            "xi_2": float(xi_null_variance(x)),
            "beta": 1.0,
        }[k]
    if k == "tau":
        return _var_tau(x, y)
    if k == "rho":
        return _var_rho(x, y)
    if k == "xi":
        return float(xi_nvarcalculate(x, y))
    if k == "xi_2":
        return float(xi_nvarcalculate(y, x))
    return _var_beta(x, y)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def bootstrap(
    data: Any,
    measures: str | Iterable[str] | None = None,
    n_boot: int = 500,
    random_state: RandomLike = None,
    **options: Any,
) -> dict[str, np.ndarray]:
    r"""Nonparametric bootstrap replicates of sample measures.

    Pairs are resampled with replacement and the estimators recomputed on
    the resampled data (mid-ranks for the resulting ties).  The bootstrap of
    the empirical copula process is consistent (Fermanian, Radulović &
    Wegkamp, 2004, *Bernoulli* 10, 847--860).

    Returns
    -------
    dict
        ``{key: ndarray of shape (n_boot,)}``.
    """
    x, y = _xy(data)
    keys = _iter_keys(measures)
    rng = as_rng(random_state)
    n = x.size
    reps = {k: np.empty(int(n_boot)) for k in keys}
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        for k in keys:
            opts = dict(options)
            if k in ("xi", "xi_2", "mutual_information"):
                opts.setdefault("random_state", rng)
            try:
                reps[k][b] = sample_measure(xb, yb, k, **opts)
            except Exception:
                reps[k][b] = np.nan
    return reps


# ---------------------------------------------------------------------------
# estimate()
# ---------------------------------------------------------------------------


def estimate(
    data: Any,
    measures: str | Iterable[str] | None = None,
    ci: str | None = None,
    level: float = 0.95,
    n_boot: int = 500,
    random_state: RandomLike = None,
    **options: Any,
):
    r"""Estimate dependence measures, optionally with confidence intervals.

    Parameters
    ----------
    data : array_like of shape (n, 2), DataFrame or EmpiricalCopula
        Bivariate sample (raw data or pseudo-observations).
    measures : str or iterable of str, optional
        Measure keys or aliases (default: ``xi, rho, tau, footrule, gamma,
        beta, nu``); every key of :data:`copul.stats.SAMPLE_ESTIMATORS`.
    ci : {None, "asymptotic", "bootstrap"}
        Confidence intervals.  ``"asymptotic"`` uses the normal approximation
        :math:`\hat\kappa_n \pm z_{(1+\text{level})/2}\,\hat\sigma/\sqrt n`
        (see :func:`asymptotic_variance`) for ``tau, rho, xi, xi_2, beta``
        and falls back to the bootstrap for the other measures (reported in
        the ``ci_method`` column); ``"bootstrap"`` uses percentile intervals
        of ``n_boot`` nonparametric bootstrap replicates (Efron & Tibshirani,
        1993) and the bootstrap standard deviation as standard error.
        Intervals are clipped to the range of the measure.
    level : float
        Confidence level.
    n_boot : int
        Number of bootstrap replicates.
    random_state : int, Generator or None
        Seed (bootstrap, random tie breaking).
    **options
        Estimator options, e.g. ``k=50`` / ``method="cfg"`` for the tail
        coefficients or ``p=3`` for ``"lp"``.

    Returns
    -------
    pandas.DataFrame
        Indexed by measure key with columns ``estimate, se, ci_low,
        ci_high, ci_method, n``.

    Notes
    -----
    The normal intervals for Kendall's tau, Spearman's rho and Blomqvist's
    beta have close to nominal coverage already for :math:`n\approx 200`.
    Chatterjee's :math:`\xi_n` is biased downwards for dependent data and
    its variance estimator underestimates in small samples, so its
    intervals undercover in moderate samples (about 80% nominal-95%
    coverage at :math:`n = 400` for a Clayton copula with
    :math:`\tau = 0.5`).

    Examples
    --------
    >>> import copul as cp
    >>> from copul.stats import estimate
    >>> X = cp.Gaussian(rho=0.5).rvs(1000, random_state=0)
    >>> estimate(X, ["tau", "rho"], ci="asymptotic")  # doctest: +SKIP
             estimate        se    ci_low   ci_high   ci_method     n
    tau      0.3375  0.018...  0.30...  0.37...  asymptotic  1000
    rho      0.4865  0.024...  ...
    """
    x, y = _xy(data)
    n = x.size
    keys = _iter_keys(measures)
    if ci not in (None, "asymptotic", "bootstrap"):
        raise ValueError("ci must be None, 'asymptotic' or 'bootstrap'")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    rng = as_rng(random_state)
    rows = {}
    for k in keys:
        opts = dict(options)
        if k in ("xi", "xi_2", "mutual_information"):
            opts.setdefault("random_state", rng)
        rows[k] = {
            "estimate": sample_measure(x, y, k, **opts),
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_method": "none" if ci is None else ci,
            "n": n,
        }
    if ci is not None:
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        boot_keys = []
        for k in keys:
            if ci == "asymptotic" and k in ASYMPTOTIC_MEASURES:
                se = np.sqrt(asymptotic_variance((x, y), k) / n)
                est = rows[k]["estimate"]
                rows[k].update(se=se, ci_low=est - z * se, ci_high=est + z * se)
            else:
                boot_keys.append(k)
                rows[k]["ci_method"] = "bootstrap"
        if boot_keys:
            reps = bootstrap((x, y), boot_keys, n_boot=n_boot, random_state=rng, **options)
            a = (1.0 - level) / 2.0
            for k in boot_keys:
                r = reps[k][np.isfinite(reps[k])]
                if r.size:
                    rows[k].update(
                        se=float(np.std(r, ddof=1)) if r.size > 1 else np.nan,
                        ci_low=float(np.quantile(r, a)),
                        ci_high=float(np.quantile(r, 1.0 - a)),
                    )
        for k in keys:
            lo, hi = get_measure(k).range
            rows[k]["ci_low"] = float(np.clip(rows[k]["ci_low"], lo, hi))
            rows[k]["ci_high"] = float(np.clip(rows[k]["ci_high"], lo, hi))
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "measure"
    return df


# ---------------------------------------------------------------------------
# independence tests
# ---------------------------------------------------------------------------


@dataclass
class IndependenceTestResult:
    """Result of :func:`independence_test`.

    Attributes
    ----------
    statistic : float
        Value of the test statistic (the sample measure for ``tau``,
        ``rho``, ``xi``; Hoeffding's :math:`D_n`; :math:`I_n` for ``cvm``).
    pvalue : float
    method : str
        Name of the test.
    null_distribution : str
        ``"asymptotic"`` or ``"permutation"``.
    alternative : str
    n : int
    extra : dict
        Additional information (e.g. the z-score, number of permutations).
    """

    statistic: float
    pvalue: float
    method: str
    null_distribution: str
    alternative: str
    n: int
    extra: dict = field(default_factory=dict)

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether :math:`H_0` is rejected at level ``alpha``."""
        return bool(self.pvalue < alpha)

    def __repr__(self) -> str:
        return (
            f"IndependenceTestResult(method={self.method!r}, statistic={self.statistic:.6g}, "
            f"pvalue={self.pvalue:.4g}, null={self.null_distribution!r}, n={self.n})"
        )


_TEST_METHODS = ("tau", "rho", "xi", "hoeffding", "cvm")


def _perm_pvalue(stat_fn, x, y, observed, n_perm, rng, greater=True, two_sided=False):
    cnt = 0
    for _ in range(int(n_perm)):
        s = stat_fn(x, y[rng.permutation(y.size)])
        if two_sided:
            cnt += abs(s) >= abs(observed) - 1e-12
        elif greater:
            cnt += s >= observed - 1e-12
        else:
            cnt += s <= observed + 1e-12
    return (cnt + 1.0) / (n_perm + 1.0)


def independence_test(
    data: Any,
    method: str = "tau",
    null_distribution: str | None = None,
    alternative: str | None = None,
    n_perm: int = 999,
    random_state: RandomLike = None,
) -> IndependenceTestResult:
    r"""Rank test of independence :math:`H_0: C = \Pi`.

    Parameters
    ----------
    data : array_like of shape (n, 2) or EmpiricalCopula
    method : {"tau", "rho", "xi", "hoeffding", "cvm"}
        * ``"tau"``: :math:`z = \tau_n/\sqrt{2(2n+5)/(9n(n-1))}`
          (Kendall, 1938);
        * ``"rho"``: :math:`z = \rho_n\sqrt{n-1}`;
        * ``"xi"``: one-sided test with :math:`\sqrt n\,\xi_n \to N(0, 2/5)`
          (Chatterjee, 2021) -- consistent against all alternatives;
        * ``"hoeffding"``: Hoeffding's (1948) :math:`D_n`
          (:func:`~copul.stats.estimators.sample_bkr`), consistent against
          all continuous alternatives;
        * ``"cvm"``: the Cramér--von Mises statistic
          :math:`I_n = n\iint(C_n - \Pi)^2` (Deheuvels, 1981; Genest &
          Rémillard, 2004).
    null_distribution : {"asymptotic", "permutation"}, optional
        Default ``"asymptotic"`` for ``tau, rho, xi`` and ``"permutation"``
        (exact up to Monte Carlo error, :math:`p = (1 + \#\{T^*\ge T\})/(B+1)`)
        for ``hoeffding`` and ``cvm``.
    alternative : {"two-sided", "greater", "less"}, optional
        For ``tau`` and ``rho`` (default ``"two-sided"``); the other tests
        are one-sided (large values are significant).
    n_perm : int
        Number of permutations.
    random_state : int, Generator or None

    Returns
    -------
    IndependenceTestResult
    """
    x, y = _xy(data)
    n = x.size
    method = str(method).lower()
    if method in ("d", "hoeffdings_d", "bkr"):
        method = "hoeffding"
    if method not in _TEST_METHODS:
        raise ValueError(f"method must be one of {_TEST_METHODS}, got {method!r}")
    if null_distribution is None:
        null_distribution = "asymptotic" if method in ("tau", "rho", "xi") else "permutation"
    if null_distribution not in ("asymptotic", "permutation"):
        raise ValueError("null_distribution must be 'asymptotic' or 'permutation'")
    if method in ("hoeffding", "cvm") and null_distribution == "asymptotic":
        raise ValueError(f"method={method!r} supports null_distribution='permutation' only.")
    if method in ("tau", "rho"):
        alternative = alternative or "two-sided"
    else:
        if alternative not in (None, "greater"):
            raise ValueError(f"method={method!r} is one-sided (alternative='greater').")
        alternative = "greater"
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    rng = as_rng(random_state)

    xi_seed = int(rng.integers(2**32))
    stat_fns = {
        "tau": sample_tau,
        "rho": sample_rho,
        "xi": lambda a, b: sample_xi(a, b, random_state=xi_seed),
        "hoeffding": sample_bkr,
        "cvm": cramer_von_mises_independence,
    }
    fn = stat_fns[method]
    stat = float(fn(x, y))
    extra: dict[str, Any] = {}
    if null_distribution == "asymptotic":
        var0 = asymptotic_variance((x, y), method, null=True)
        zval = np.sqrt(n) * stat / np.sqrt(var0)
        extra["z"] = float(zval)
        if alternative == "two-sided":
            p = 2.0 * stats.norm.sf(abs(zval))
        elif alternative == "greater":
            p = stats.norm.sf(zval)
        else:
            p = stats.norm.cdf(zval)
    else:
        extra["n_perm"] = int(n_perm)
        p = _perm_pvalue(
            fn,
            x,
            y,
            stat,
            n_perm,
            rng,
            greater=alternative != "less",
            two_sided=alternative == "two-sided",
        )
    return IndependenceTestResult(
        statistic=stat,
        pvalue=float(min(1.0, p)),
        method=method,
        null_distribution=null_distribution,
        alternative=alternative,
        n=n,
        extra=extra,
    )
