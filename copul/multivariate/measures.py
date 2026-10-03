r"""
Multivariate dependence measures of :math:`d`-copulas and of data.

**Spearman's rho** (Schmid & Schmidt, 2007a; Nelsen, 1996; Joe, 1990). With
:math:`h(d) = (d+1)/(2^d-(d+1))`,

.. math::

   \rho_1(C) = h(d)\Bigl(2^d\int_{[0,1]^d}C(u)\,du - 1\Bigr),\qquad
   \rho_2(C) = h(d)\Bigl(2^d\int_{[0,1]^d}\Pi(u)\,dC(u) - 1\Bigr),

and :math:`\rho_3` is the average of the bivariate Spearman's rho of all
:math:`\binom d2` pairs (Kendall, 1970; Nelsen, 1996).  Since
:math:`\int C\,du = E\prod_i(1-U_i)` and :math:`\int\Pi\,dC = E\prod_iU_i
= \int\bar C\,du`, :math:`\rho_1(C)=\rho_2(\hat C)`; for :math:`d=3`,
:math:`\rho_3 = (\rho_1+\rho_2)/2` (Nelsen, 1996), so for radially symmetric
copulas :math:`\rho_1=\rho_2=\rho_3` in three dimensions.

**Kendall's tau** (Nelsen, 1996; Joe, 1990):

.. math::

   \tau_d(C) = \frac{2^d\int_{[0,1]^d}C(u)\,dC(u) - 1}{2^{d-1}-1},
   \qquad \int C\,dC = E\,C(U) = P(U'\le U),

with an independent copy :math:`U'` of :math:`U`; for :math:`d=3` it is the
average of the three pairwise Kendall's taus (Nelsen, 1996).

**Blomqvist's beta** (Úbeda-Flores, 2005; Schmid & Schmidt, 2007b):

.. math::

   \beta_d(C) = \frac{2^{d-1}\bigl(C(\tfrac12,\dots,\tfrac12)
   + \bar C(\tfrac12,\dots,\tfrac12)\bigr) - 1}{2^{d-1}-1}.

All three reduce to the classical bivariate measures for :math:`d=2`, vanish
at :math:`\Pi_d` and equal one at :math:`M_d`.

Computation (``method=``) for copulas:

* ``"auto"`` (default): exact values where available -- closed forms of the
  class (e.g. :math:`\Pi_d`, :math:`M_d`, Archimedean :math:`\tau_d` from the
  Kendall distribution), the bivariate measures of copul for :math:`d=2`, the
  pairwise identities for :math:`d=3` -- otherwise ``"qmc"`` if the cdf is
  cheap, else ``"mc"``;
* ``"qmc"``: randomized (scrambled Sobol) quasi-Monte Carlo integration of
  :math:`C` or :math:`\bar C` over :math:`[0,1]^d` (:math:`\rho_1,\rho_2`);
* ``"mc"``: Monte Carlo averages over a sample of the copula, e.g.
  :math:`E\prod_i(1-U_i)`, :math:`E\,C(U)` (or :math:`P(U'\le U)` from
  paired samples if the cdf is expensive);
* ``"exact"``: only closed forms (raises if none is known).

``random_state`` (default ``0``) makes the stochastic methods reproducible;
``return_se=True`` also returns the (Monte Carlo) standard error.

Sample versions take an ``(n, d)`` data array and use the empirical copula
of the pseudo-observations :math:`\hat U_{ij}=R_{ij}/n`.

References
----------
* Joe, H. (1990). Multivariate concordance. *J. Multivariate Anal.* 35,
  12--30.
* Kendall, M. G. (1970). *Rank Correlation Methods*, 4th ed. Griffin.
* Nelsen, R. B. (1996). Nonparametric measures of multivariate association.
  In *Distributions with Fixed Marginals and Related Topics*, IMS Lecture
  Notes 28, 223--232.
* Nelsen, R. B. (2002). Concordance and copulas: a survey. In *Distributions
  with Given Marginals and Statistical Modelling*, 169--177. Kluwer.
* Schmid, F. and Schmidt, R. (2007a). Multivariate extensions of Spearman's
  rho and related statistics. *Statist. Probab. Lett.* 77, 407--416.
* Schmid, F. and Schmidt, R. (2007b). Nonparametric inference on
  multivariate versions of Blomqvist's beta and related measures of tail
  dependence. *Metrika* 66, 323--354.
* Genest, C., Nešlehová, J. and Ben Ghorbal, N. (2011). Estimators based on
  Kendall's tau in multivariate copula models. *Aust. N. Z. J. Stat.* 53,
  157--177.
* Úbeda-Flores, M. (2005). Multivariate versions of Blomqvist's beta and
  Spearman's footrule. *Ann. Inst. Statist. Math.* 57, 781--788.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

from copul.multivariate.base import CopulaND, as_rng

__all__ = [
    "blomqvists_beta_nd",
    "kendalls_tau_nd",
    "sample_blomqvists_beta_nd",
    "sample_kendalls_tau_nd",
    "sample_spearmans_rho_nd",
    "spearmans_rho_h",
    "spearmans_rho_nd",
]

_DEFAULT_MC = 200_000
_DEFAULT_QMC_LOG2 = 15


def spearmans_rho_h(d: int) -> float:
    r"""Normalizing constant :math:`h(d)=(d+1)/(2^d-(d+1))` of :math:`\rho_1,\rho_2`."""
    d = int(d)
    return (d + 1.0) / (2.0**d - (d + 1.0))


def _is_data(obj: Any) -> bool:
    if isinstance(obj, CopulaND):
        return False
    if type(obj).__name__ == "EmpiricalCopula":
        return True
    if isinstance(obj, (np.ndarray, list, tuple)):
        return True
    from copul._lazy import is_pandas_instance

    return is_pandas_instance(obj, "DataFrame")


def _as_data(obj: Any) -> np.ndarray:
    if type(obj).__name__ == "EmpiricalCopula":
        return np.asarray(obj.data, dtype=float)
    from copul.stats._utils import as_data

    return as_data(obj, min_dim=2)


def _copula(C: Any) -> CopulaND:
    from copul.multivariate.basic import as_copula_nd

    return as_copula_nd(C)


def _ret(val: float, se: float, return_se: bool):
    return (float(val), float(se)) if return_se else float(val)


def _sobol(d: int, log2n: int, random_state) -> np.ndarray:
    seed = int(as_rng(random_state).integers(0, 2**31 - 1))
    return stats.qmc.Sobol(d, scramble=True, seed=seed).random_base2(log2n)


def _pairwise_mean(C: CopulaND, name: str, **kw) -> float:
    if C.exchangeable:
        return float(getattr(C.margin(0, 1), name)(**kw))
    vals = [float(getattr(C.margin(i, j), name)(**kw)) for i, j in combinations(range(C.dim), 2)]
    return float(np.mean(vals))


def _check_method(method: str) -> str:
    method = str(method).lower()
    if method not in ("auto", "exact", "mc", "qmc"):
        raise ValueError("method must be 'auto', 'exact', 'mc' or 'qmc'.")
    return method


# ---------------------------------------------------------------------------
# Spearman's rho
# ---------------------------------------------------------------------------


def spearmans_rho_nd(
    C: Any,
    kind: int = 1,
    method: str = "auto",
    n_samples: int | None = None,
    random_state: Any = 0,
    return_se: bool = False,
):
    r"""Multivariate Spearman's rho :math:`\rho_1`, :math:`\rho_2` or :math:`\rho_3`.

    See the module docstring for the definitions (Schmid & Schmidt, 2007a).

    Parameters
    ----------
    C : CopulaND, copul copula or data
        A :math:`d`-copula (anything accepted by
        :func:`~copul.multivariate.as_copula_nd`) or an ``(n, d)`` data array /
        DataFrame / ``EmpiricalCopula`` (then the sample version
        :func:`sample_spearmans_rho_nd` is returned).
    kind : {1, 2, 3}
        Which version.
    method : {"auto", "exact", "qmc", "mc"}
        Computation method (see module docstring).
    n_samples : int, optional
        Monte Carlo sample size (default 200 000) or number of quasi-Monte
        Carlo points (rounded to a power of two, default :math:`2^{15}`).
    random_state : int, Generator or None
        Seed of the stochastic methods (default 0).
    return_se : bool
        Also return the standard error (zero for exact values; for ``"qmc"``
        estimated from 8 independent scramblings).

    Returns
    -------
    float or (float, float)
    """
    if _is_data(C):
        val = sample_spearmans_rho_nd(C, kind=kind)
        return _ret(val, np.nan, return_se)
    kind = int(kind)
    if kind not in (1, 2, 3):
        raise ValueError("kind must be 1, 2 or 3.")
    method = _check_method(method)
    C = _copula(C)
    d = C.dim
    if kind == 3:  # average of the bivariate (closed-form or quadrature) values
        exact = C._exact_measure("rho3")
        val = exact if exact is not None else _pairwise_mean(C, "spearmans_rho")
        return _ret(val, 0.0, return_se)
    if method in ("auto", "exact"):
        exact = C._exact_measure(f"rho{kind}")
        if exact is not None:
            return _ret(exact, 0.0, return_se)
        if d == 2 or (d == 3 and C.radially_symmetric):
            return _ret(_pairwise_mean(C, "spearmans_rho"), 0.0, return_se)
        if method == "exact":
            raise ValueError(f"no closed form of rho_{kind} is known for {C!r}.")
        method = "qmc" if C._cheap_cdf else "mc"
    h = spearmans_rho_h(d)
    if method == "qmc":
        log2n = _DEFAULT_QMC_LOG2 if n_samples is None else max(4, int(np.log2(n_samples)))
        rng = as_rng(random_state)
        reps = 8 if return_se else 1
        means = []
        for _ in range(reps):
            P = _sobol(d, log2n - (3 if return_se else 0), rng)
            f = C._cdf_clean(P) if kind == 1 else np.asarray(C._survival(P), dtype=float)
            means.append(float(np.mean(f)))
        m = float(np.mean(means))
        se = float(np.std(means, ddof=1) / np.sqrt(reps)) if reps > 1 else np.nan
        return _ret(h * (2.0**d * m - 1.0), h * 2.0**d * se, return_se)
    n = _DEFAULT_MC if n_samples is None else int(n_samples)
    X = C.rvs(n, random_state=random_state)
    vals = np.prod(1.0 - X, axis=1) if kind == 1 else np.prod(X, axis=1)
    m, se = float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n))
    return _ret(h * (2.0**d * m - 1.0), h * 2.0**d * se, return_se)


def sample_spearmans_rho_nd(X: Any, kind: int = 1) -> float:
    r"""Sample versions of :math:`\rho_1,\rho_2,\rho_3` (Schmid & Schmidt, 2007a).

    .. math::

       \hat\rho_1 = h(d)\Bigl(\frac{2^d}{n}\sum_{i=1}^n\prod_{j=1}^d(1-\hat U_{ij})-1\Bigr),
       \qquad
       \hat\rho_2 = h(d)\Bigl(\frac{2^d}{n}\sum_{i=1}^n\prod_{j=1}^d\hat U_{ij}-1\Bigr),

    with :math:`\hat U_{ij}=R_{ij}/n` (the integrals of the empirical copula);
    :math:`\hat\rho_3` is the average of the classical pairwise Spearman
    rank correlations.

    Parameters
    ----------
    X : array_like of shape (n, d), DataFrame or EmpiricalCopula
    kind : {1, 2, 3}
    """
    from copul.stats.pseudo_obs import pseudo_obs

    X = _as_data(X)
    n, d = X.shape
    kind = int(kind)
    if kind == 3:
        R = stats.spearmanr(X).statistic
        if np.ndim(R) == 0:
            return float(R)
        return float(np.asarray(R)[np.triu_indices(d, 1)].mean())
    if kind not in (1, 2):
        raise ValueError("kind must be 1, 2 or 3.")
    U = pseudo_obs(X, scale="n")
    vals = np.prod(1.0 - U, axis=1) if kind == 1 else np.prod(U, axis=1)
    return float(spearmans_rho_h(d) * (2.0**d * vals.mean() - 1.0))


# ---------------------------------------------------------------------------
# Kendall's tau
# ---------------------------------------------------------------------------


def kendalls_tau_nd(
    C: Any,
    method: str = "auto",
    n_samples: int | None = None,
    random_state: Any = 0,
    return_se: bool = False,
):
    r"""Multivariate Kendall's tau :math:`\tau_d` (Nelsen, 1996).

    :math:`\tau_d=(2^d E\,C(U)-1)/(2^{d-1}-1)`; see the module docstring.

    Parameters
    ----------
    C : CopulaND, copul copula or data
        Copula, or data (then :func:`sample_kendalls_tau_nd` is returned).
    method : {"auto", "exact", "mc"}
        ``"auto"`` uses closed forms (the class's, :math:`d=2`, and the
        pairwise average for :math:`d=3`), otherwise ``"mc"``: the average of
        :math:`C(U_k)` over a sample (or of :math:`1\{U'_k\le U_k\}` over
        paired samples if the cdf is expensive).
    n_samples : int, optional
        Monte Carlo sample size (default 200 000).
    random_state : int, Generator or None
    return_se : bool
        Also return the standard error.
    """
    if _is_data(C):
        return _ret(sample_kendalls_tau_nd(C), np.nan, return_se)
    method = _check_method(method)
    if method == "qmc":
        raise ValueError("Kendall's tau is an integral w.r.t. dC; use method='mc'.")
    C = _copula(C)
    d = C.dim
    if method in ("auto", "exact"):
        exact = C._exact_measure("tau")
        if exact is not None:
            return _ret(exact, 0.0, return_se)
        if d in (2, 3):
            return _ret(_pairwise_mean(C, "kendalls_tau"), 0.0, return_se)
        if method == "exact":
            raise ValueError(f"no closed form of tau_d is known for {C!r}.")
    a = 2.0**d / (2.0 ** (d - 1) - 1.0)
    rng = as_rng(random_state)
    n = _DEFAULT_MC if n_samples is None else int(n_samples)
    if C._cheap_cdf:
        X = C.rvs(n, random_state=rng)
        vals = C._cdf_clean(X)
    else:
        X = C.rvs(n, random_state=rng)
        Y = C.rvs(n, random_state=rng)
        vals = np.all(Y <= X, axis=1).astype(float)
    m, se = float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n))
    return _ret(a * m - 1.0 / (2.0 ** (d - 1) - 1.0), a * se, return_se)


def sample_kendalls_tau_nd(X: Any) -> float:
    r"""Sample version of :math:`\tau_d` (Nelsen, 1996; Genest, Nešlehová &
    Ben Ghorbal, 2011).

    .. math::

       \hat\tau_d = \frac1{2^{d-1}-1}\Bigl(\frac{2^d}{n(n-1)}\sum_{i\ne k}
       \mathbf 1\{X_k < X_i\} - 1\Bigr),

    with componentwise strict inequality (a U-statistic; :math:`O(n^2d)` time,
    chunked).  For :math:`d=2` and no ties it equals Kendall's
    :math:`\tau_a`.
    """
    from copul.stats._utils import chunk_size

    X = _as_data(X)
    n, d = X.shape
    count = 0
    step = chunk_size(n * d)
    for s in range(0, n, step):
        blk = X[s : s + step]
        less = np.ones((blk.shape[0], n), dtype=bool)
        for j in range(d):
            less &= X[None, :, j] < blk[:, j, None]
        count += int(np.count_nonzero(less))
    return float((2.0**d * count / (n * (n - 1.0)) - 1.0) / (2.0 ** (d - 1) - 1.0))


# ---------------------------------------------------------------------------
# Blomqvist's beta
# ---------------------------------------------------------------------------


def blomqvists_beta_nd(C: Any) -> float:
    r"""Multivariate Blomqvist's beta :math:`\beta_d` (Úbeda-Flores, 2005;
    Schmid & Schmidt, 2007b).

    .. math::

       \beta_d = \frac{2^{d-1}\bigl(C(\tfrac12\mathbf 1)+\bar C(\tfrac12\mathbf 1)\bigr)-1}
       {2^{d-1}-1},

    evaluated exactly from the cdf (:math:`\bar C` by inclusion--exclusion).
    For data the sample version :func:`sample_blomqvists_beta_nd` is returned.
    """
    if _is_data(C):
        return sample_blomqvists_beta_nd(C)
    C = _copula(C)
    exact = C._exact_measure("beta")
    if exact is not None:
        return float(exact)
    d = C.dim
    half = np.full((1, d), 0.5)
    s = float(C._cdf_clean(half)[0]) + float(np.asarray(C._survival(half)).reshape(-1)[0])
    return float((2.0 ** (d - 1) * s - 1.0) / (2.0 ** (d - 1) - 1.0))


def sample_blomqvists_beta_nd(X: Any) -> float:
    r"""Sample version of :math:`\beta_d` (Schmid & Schmidt, 2007b).

    :math:`\hat\beta_d=(2^{d-1}(C_n(\tfrac12)+\bar C_n(\tfrac12))-1)/(2^{d-1}-1)`
    with the empirical copula of :math:`\hat U_{ij}=R_{ij}/n`:
    :math:`C_n(\tfrac12)` and :math:`\bar C_n(\tfrac12)` are the proportions of
    observations with all :math:`\hat U_{ij}\le\tfrac12`, respectively all
    :math:`\hat U_{ij}>\tfrac12`.
    """
    from copul.stats.pseudo_obs import pseudo_obs

    X = _as_data(X)
    n, d = X.shape
    U = pseudo_obs(X, scale="n")
    lower = float(np.mean(np.all(U <= 0.5, axis=1)))
    upper = float(np.mean(np.all(U > 0.5, axis=1)))
    return float((2.0 ** (d - 1) * (lower + upper) - 1.0) / (2.0 ** (d - 1) - 1.0))
