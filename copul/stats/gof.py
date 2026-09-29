r"""
Goodness-of-fit tests for parametric copula families.

Blanket test of Genest, Rémillard & Beaudoin (2009) based on the empirical
copula process :math:`\mathbb C_n = \sqrt n\,(C_n - C_{\hat\theta_n})`:

.. math::

   S_n = \sum_{i=1}^n \bigl\{C_n(\hat U_i) - C_{\hat\theta_n}(\hat U_i)\bigr\}^2
   \quad(\text{Cramér--von Mises}),\qquad
   T_n = \sqrt n\,\max_i \bigl|C_n(\hat U_i) - C_{\hat\theta_n}(\hat U_i)\bigr|
   \quad(\text{Kolmogorov--Smirnov}),

with p-values from the parametric bootstrap (Genest, Rémillard & Beaudoin,
2009, Appendix A; validity: Genest & Rémillard, 2008): for
:math:`k = 1,\dots,N` draw a sample of size :math:`n` from
:math:`C_{\hat\theta_n}`, compute its pseudo-observations, re-estimate
:math:`\theta_k^*` with the same method and compute :math:`S^*_{n,k}`; then
:math:`p = \bigl(\#\{k : S^*_{n,k}\ge S_n\} + \tfrac12\bigr)/(N + 1)`.

References
----------
* Genest, C. and Rémillard, B. (2008). Validity of the parametric bootstrap
  for goodness-of-fit testing in semiparametric models. *Ann. Inst. Henri
  Poincaré Probab. Stat.* 44, 1096--1127.
* Genest, C., Rémillard, B. and Beaudoin, D. (2009). Goodness-of-fit tests
  for copulas: a review and a power study. *Insurance Math. Econom.* 44,
  199--213.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from copul.stats._adapters import ParametricCDF
from copul.stats._adapters import cdf as _cdf
from copul.stats._adapters import sample as _sample
from copul.stats._utils import RandomLike, as_rng, dominance_counts
from copul.stats.fitting import FitResult, _params_of, _uv, fit, resolve_family
from copul.stats.pseudo_obs import pseudo_obs

log = logging.getLogger(__name__)

__all__ = ["GofResult", "gof_statistic", "gof_test"]


def gof_statistic(copula, U: np.ndarray, statistic: str = "cvm", model_cdf=None) -> float:
    r"""Cramér--von Mises :math:`S_n` or Kolmogorov--Smirnov :math:`T_n`
    distance between the empirical copula of the pseudo-observations ``U``
    and ``copula``, evaluated at the pseudo-observations (``model_cdf``:
    precomputed values :math:`C_\theta(\hat U_i)`, internal)."""
    U = np.asarray(U, dtype=float)
    n = U.shape[0]
    cn = dominance_counts(U[:, 0], U[:, 1], ties="weak") / n
    ct = _cdf(copula, U[:, 0], U[:, 1]) if model_cdf is None else np.asarray(model_cdf)
    d = cn - ct
    if statistic == "cvm":
        return float(np.sum(d * d))
    if statistic == "ks":
        return float(np.sqrt(n) * np.max(np.abs(d)))
    raise ValueError("statistic must be 'cvm' or 'ks'")


def _typicals(base, names) -> list[float]:
    by_name = {p.name: p for p in _params_of(base)}
    return [by_name[n].typical() if n in by_name else 0.0 for n in names]


@dataclass
class GofResult:
    """Result of :func:`gof_test`.

    Attributes
    ----------
    statistic : float
        Observed :math:`S_n` (``cvm``) or :math:`T_n` (``ks``).
    pvalue : float
    statistic_name : str
    fit : FitResult
        Fit on the observed data.
    n_boot : int
        Number of successful bootstrap replicates.
    boot_statistics : numpy.ndarray
    refit : bool
        Whether the parameter was re-estimated on each bootstrap sample.
    seconds : float
    """

    statistic: float
    pvalue: float
    statistic_name: str
    fit: FitResult
    n_boot: int
    boot_statistics: np.ndarray = field(repr=False)
    refit: bool = True
    seconds: float = 0.0

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether the family is rejected at level ``alpha``."""
        return bool(self.pvalue < alpha)

    def __repr__(self) -> str:
        return (
            f"GofResult({self.fit.family}, {self.statistic_name}={self.statistic:.5g}, "
            f"pvalue={self.pvalue:.4g}, n_boot={self.n_boot}, refit={self.refit}, "
            f"method={self.fit.method!r})"
        )


def gof_test(
    family_or_fit: Any,
    data: Any,
    statistic: str = "cvm",
    n_boot: int = 100,
    method: str = "parametric_bootstrap",
    fit_method: str | None = None,
    refit: bool = True,
    sampler: str = "auto",
    random_state: RandomLike = None,
    **fit_kwargs: Any,
) -> GofResult:
    r"""Parametric-bootstrap goodness-of-fit test of Genest, Rémillard &
    Beaudoin (2009).

    Parameters
    ----------
    family_or_fit : family (class, instance, name) or FitResult
        Hypothesized family :math:`H_0: C\in\{C_\theta\}`; a
        :class:`~copul.stats.FitResult` reuses its estimate and method.
    data : array_like of shape (n, 2)
        Raw data (rank-transformed internally).
    statistic : {"cvm", "ks"}
        :math:`S_n` (default, recommended by GRB 2009) or :math:`T_n`.
    n_boot : int
        Number :math:`N` of bootstrap replicates.
    method : {"parametric_bootstrap"}
        Only the parametric bootstrap is implemented.
    fit_method : str, optional
        Estimation method (default: the method of the given ``FitResult``,
        else ``"mle"``); ``"itau"`` is much faster for one-parameter
        families.
    refit : bool
        Re-estimate the parameter on each bootstrap sample (required for the
        validity of the test; ``refit=False`` keeps :math:`\hat\theta_n` and
        is only a fast, conservative approximation).
    sampler : {"auto", "conditional", "rvs"}
        How bootstrap samples are drawn (see
        :func:`copul.stats._adapters.sample`; ``"auto"`` uses the seeded
        conditional distribution method whenever :math:`\partial_1 C` is
        available in closed form).
    random_state : int, Generator or None
        Seed.
    **fit_kwargs
        Passed to :func:`copul.stats.fit`.

    Returns
    -------
    GofResult
    """
    if method not in ("parametric_bootstrap", "pb"):
        raise ValueError("only method='parametric_bootstrap' is implemented")
    statistic = statistic.lower()
    if statistic not in ("cvm", "ks"):
        raise ValueError("statistic must be 'cvm' or 'ks'")
    t0 = time.perf_counter()
    rng = as_rng(random_state)
    U = _uv(data, True)
    n = U.shape[0]
    if isinstance(family_or_fit, FitResult):
        res = family_or_fit
        # refit the family with the fixed parameters of the fit kept fixed
        fam = type(res.copula)()
        if res.fixed:
            fam = fam(**res.fixed)
        fit_method = fit_method or res.method
    else:
        fit_method = fit_method or "mle"
        res = fit(family_or_fit, U, method=fit_method, pseudo_obs=False, **fit_kwargs)
        fam = family_or_fit
    s_obs = gof_statistic(res.copula, U, statistic)
    boot = np.full(int(n_boot), np.nan)
    start = res.params
    # compile the family cdf once for the refitted parameters
    base = resolve_family(fam)[0]
    names = list(res.params)
    theta_hat = np.array([res.params[k] for k in names])
    pcdf = None
    if refit:
        pcdf = ParametricCDF(
            base,
            names,
            lambda t: base(**{k: float(x) for k, x in zip(names, t)}),
            [theta_hat, theta_hat * 0.8 + 0.2 * np.array(_typicals(base, names))],
        )
        if not pcdf.ok:
            pcdf = None
    for k in range(int(n_boot)):
        try:
            Xs = _sample(res.copula, n, random_state=rng, method=sampler)
            Us = pseudo_obs(Xs)
            if refit:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kw = dict(fit_kwargs)
                    if fit_method == "mle":
                        kw.setdefault("start", start)
                    rk = fit(fam, Us, method=fit_method, pseudo_obs=False, _light=True, **kw)
                cop = rk.copula
                mc = (
                    pcdf([rk.params[k] for k in names], Us[:, 0], Us[:, 1])
                    if pcdf is not None
                    else None
                )
            else:
                cop, mc = res.copula, None
            boot[k] = gof_statistic(cop, Us, statistic, model_cdf=mc)
        except Exception as e:
            log.debug("bootstrap replicate %d failed: %s", k, e)
            if k == 0:
                raise RuntimeError(
                    f"parametric bootstrap failed for {res.family}: {type(e).__name__}: {e}"
                ) from e
    ok = boot[np.isfinite(boot)]
    if ok.size < boot.size:
        warnings.warn(f"{boot.size - ok.size} of {boot.size} bootstrap replicates failed.")
    p = (np.count_nonzero(ok >= s_obs) + 0.5) / (ok.size + 1.0)
    return GofResult(
        statistic=s_obs,
        pvalue=float(p),
        statistic_name=statistic,
        fit=res,
        n_boot=int(ok.size),
        boot_statistics=ok,
        refit=refit,
        seconds=time.perf_counter() - t0,
    )
