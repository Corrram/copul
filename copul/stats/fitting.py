r"""
Parametric estimation and model selection for bivariate copula families.

* :func:`fit` -- maximum pseudo-likelihood estimation (``method="mle"``,
  Genest, Ghoudi & Rivest, 1995) and moment-type inversion of rank
  correlations (``"itau"``, ``"irho"``, ``"ixi"``, ``"ibeta"``; Genest &
  Rivest, 1993; Kojadinovic & Yan, 2010) for every family of :mod:`copul`
  with fully numeric parameters (one or several), returning a
  :class:`FitResult`;
* :func:`select` -- fit several families and rank them by AIC/BIC.

Data are transformed to pseudo-observations :math:`\hat U_i = R_i/(n+1)`
unless ``pseudo_obs=False`` (data already uniform on :math:`(0,1)^2`).

The pseudo-log-likelihood :math:`\ell(\theta) = \sum_i\log c_\theta(\hat U_i,
\hat V_i)` is maximized with :func:`scipy.optimize.minimize` over the
parameter box of the family, mapped to :math:`\mathbb R^p` (logit for bounded,
log for half-bounded intervals).  Standard errors come from the observed
information (numerical Hessian of :math:`-\ell` at the maximizer); note that
for pseudo-observations these ignore the rank transform and are therefore
somewhat optimistic (Genest, Ghoudi & Rivest, 1995, give the sandwich form).

References
----------
* Akaike, H. (1974). A new look at the statistical model identification.
  *IEEE Trans. Automat. Control* 19, 716--723.
* Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric
  estimation procedure of dependence parameters in multivariate families of
  distributions. *Biometrika* 82, 543--552.
* Genest, C. and Rivest, L.-P. (1993). Statistical inference procedures for
  bivariate Archimedean copulas. *JASA* 88, 1034--1043.
* Kojadinovic, I. and Yan, J. (2010). Comparison of three semiparametric
  methods for estimating dependence parameters in copula models.
  *Insurance Math. Econom.* 47, 52--63.
* Schwarz, G. (1978). Estimating the dimension of a model. *Ann. Statist.*
  6, 461--464.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, stats

from copul._lazy import pd
from copul.measures.backend import free_parameters
from copul.measures.engine import compute
from copul.stats._adapters import ParametricLogDensity
from copul.stats._utils import as_data
from copul.stats.estimators import sample_measure
from copul.stats.pseudo_obs import pseudo_obs as _pseudo_obs

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FAMILIES",
    "FitResult",
    "SingularFamilyError",
    "fit",
    "loglik",
    "resolve_family",
    "select",
]

#: families compared by :func:`select` by default (enum names of
#: :class:`copul.family_list.Families`)
DEFAULT_FAMILIES: tuple[str, ...] = (
    "CLAYTON",
    "FRANK",
    "GUMBEL_HOUGAARD",
    "JOE",
    "GAUSSIAN",
    "T",
    "PLACKETT",
    "GALAMBOS",
)

_MOMENT_METHODS = {"itau": "tau", "irho": "rho", "ixi": "xi", "ibeta": "beta"}


class SingularFamilyError(ValueError):
    """The family has a singular component, so it has no density and the
    likelihood is not available (use a moment method such as ``itau``)."""


# ---------------------------------------------------------------------------
# families and parameters
# ---------------------------------------------------------------------------


def _pretty_name(obj) -> str:
    name = obj.__name__ if isinstance(obj, type) else type(obj).__name__
    return name[3:] if name.startswith("Biv") and len(name) > 3 else name


def resolve_family(family: Any):
    """Return a (partially specified) copula instance and a display name.

    ``family`` may be a copula class (``cp.Clayton``), an instance with some
    parameters fixed (``cp.StudentT(nu=4)`` fits only ``rho``), an enum name
    of :class:`copul.family_list.Families` (``"GUMBEL_HOUGAARD"``, ``"T"``)
    or a class name (``"Clayton"``, case-insensitive).
    """
    if isinstance(family, FitResult):
        return family.copula, family.family
    if isinstance(family, type):
        return family(), _pretty_name(family)
    if not isinstance(family, str):
        return family, _pretty_name(family)
    from copul.family_list import Families

    key = family.strip()
    upper = key.upper().replace("-", "_").replace(" ", "_")
    if upper in Families.__members__:
        return Families[upper].cls(), key
    norm = key.lower().replace("_", "").replace("-", "").replace(" ", "")
    for member in Families:
        try:
            cls = member.cls
        except Exception:  # pragma: no cover - broken optional imports
            continue
        names = {
            cls.__name__.lower(),
            _pretty_name(cls).lower(),
            member.name.lower().replace("_", ""),
        }
        if norm in names:
            return cls(), key
    import copul

    obj = getattr(copul, key, None)
    if isinstance(obj, type):
        return obj(), key
    raise ValueError(f"Unknown copula family {family!r}.")


_ZCAP = 12.0
_XCAP = math.exp(_ZCAP)


@dataclass(frozen=True)
class _Param:
    name: str
    lo: float
    hi: float

    def to_z(self, x: float) -> float:
        lo, hi = self.lo, self.hi
        if math.isfinite(lo) and math.isfinite(hi):
            p = (x - lo) / (hi - lo)
            p = min(max(p, 1e-12), 1 - 1e-12)
            return math.log(p / (1 - p))
        if math.isfinite(lo):
            return math.log(max(x - lo, 1e-300))
        if math.isfinite(hi):
            return math.log(max(hi - x, 1e-300))
        return float(x)

    def from_z(self, z: float) -> float:
        lo, hi = self.lo, self.hi
        if math.isfinite(lo) and math.isfinite(hi):
            return lo + (hi - lo) * float(1.0 / (1.0 + np.exp(-z)))
        # unbounded sides are capped at a distance of e^12 (~1.6e5): larger
        # values are numerically degenerate and reported as "at the boundary"
        if math.isfinite(lo):
            return lo + float(np.exp(min(z, _ZCAP)))
        if math.isfinite(hi):
            return hi - float(np.exp(min(z, _ZCAP)))
        return float(np.clip(z, -_XCAP, _XCAP))

    def typical(self) -> float:
        lo, hi = self.lo, self.hi
        if math.isfinite(lo) and math.isfinite(hi):
            return 0.5 * (lo + hi)
        if math.isfinite(lo):
            return lo + 1.0
        if math.isfinite(hi):
            return hi - 1.0
        return 1.0

    def inside(self, x: float) -> bool:
        return self.lo < x < self.hi

    def grid(self, m: int = 25) -> np.ndarray:
        """Search grid over the (open) interval."""
        lo, hi = self.lo, self.hi
        if math.isfinite(lo) and math.isfinite(hi):
            return lo + (hi - lo) * np.linspace(0.02, 0.98, m)
        if math.isfinite(lo):
            return lo + np.geomspace(1e-2, 1e2, m)
        if math.isfinite(hi):
            return hi - np.geomspace(1e-2, 1e2, m)
        return np.sinh(np.linspace(-4.5, 4.5, m))

    def at_boundary(self, x: float) -> bool:
        """Whether ``x`` is (numerically) at a finite bound or huge on an
        unbounded side."""
        lo, hi = self.lo, self.hi
        if math.isfinite(lo) and x - lo < 1e-6 * max(1.0, abs(lo)):
            return True
        if math.isfinite(hi) and hi - x < 1e-6 * max(1.0, abs(hi)):
            return True
        if math.isfinite(lo):
            return x - lo > 0.999 * _XCAP
        if math.isfinite(hi):
            return hi - x > 0.999 * _XCAP
        return abs(x) > 0.999 * _XCAP


def _params_of(base) -> list[_Param]:
    out = []
    ivs = getattr(base, "intervals", None) or {}
    for name in free_parameters(base):
        iv = ivs.get(name)
        if iv is None:
            lo, hi = -math.inf, math.inf
        else:
            lo, hi = float(iv.inf), float(iv.sup)
        out.append(_Param(name, lo, hi))
    return out


def _instance(base, names: Sequence[str], theta: Sequence[float]):
    return base(**{n: float(t) for n, t in zip(names, theta)})


def _is_singular(copula) -> bool:
    try:
        ac = copula.is_absolutely_continuous
    except Exception:
        return False
    return ac is False or (isinstance(ac, (bool, np.bool_)) and not bool(ac))


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Result of :func:`fit`.

    Attributes
    ----------
    family : str
        Display name of the family.
    method : str
        ``"mle"``, ``"itau"``, ``"irho"``, ``"ixi"`` or ``"ibeta"``.
    params : dict
        Estimated parameters ``{name: value}``.
    copula : copula object
        The fitted family member.
    loglik : float
        Pseudo-log-likelihood at the estimate (``nan`` if the family has no
        density).
    n : int
        Sample size.
    k : int
        Number of estimated parameters.
    se : dict
        Standard errors (observed information for ``mle``, delta method for
        moment methods).
    cov : numpy.ndarray or None
        Estimated covariance matrix of the parameter estimates.
    converged : bool
    message : str
    nfev : int
        Number of objective evaluations.
    fixed : dict
        Parameters that were held fixed.
    extra : dict
        Method-specific information (e.g. the sample tau for ``itau``).
    """

    family: str
    method: str
    params: dict[str, float]
    copula: Any
    loglik: float
    n: int
    k: int
    se: dict[str, float] = field(default_factory=dict)
    cov: np.ndarray | None = None
    converged: bool = True
    message: str = ""
    nfev: int = 0
    fixed: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def aic(self) -> float:
        r""":math:`\mathrm{AIC} = 2k - 2\ell` (Akaike, 1974)."""
        return 2.0 * self.k - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        r""":math:`\mathrm{BIC} = k\log n - 2\ell` (Schwarz, 1978)."""
        return self.k * math.log(self.n) - 2.0 * self.loglik

    def conf_int(self, level: float = 0.95) -> dict[str, tuple[float, float]]:
        """Wald confidence intervals ``estimate +- z * se``."""
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        return {
            p: (v - z * self.se.get(p, np.nan), v + z * self.se.get(p, np.nan))
            for p, v in self.params.items()
        }

    def to_frame(self, level: float = 0.95):
        """Parameter table (estimate, se, Wald CI) as a DataFrame."""
        ci = self.conf_int(level)
        rows = {
            p: {
                "estimate": v,
                "se": self.se.get(p, np.nan),
                "ci_low": ci[p][0],
                "ci_high": ci[p][1],
            }
            for p, v in self.params.items()
        }
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "parameter"
        return df

    def summary(self, level: float = 0.95) -> str:
        """Human readable summary."""
        lines = [
            f"Copula fit: {self.family}  (method={self.method}, n={self.n})",
            f"  log-likelihood = {self.loglik:.4f}   AIC = {self.aic:.4f}   BIC = {self.bic:.4f}",
            f"  converged = {self.converged}  ({self.message})"
            if self.message
            else f"  converged = {self.converged}",
        ]
        if self.fixed:
            lines.append("  fixed: " + ", ".join(f"{k}={v:g}" for k, v in self.fixed.items()))
        ci = self.conf_int(level)
        lines.append(
            f"  {'param':>10s} {'estimate':>12s} {'se':>10s}   {int(level * 100)}% Wald CI"
        )
        for p, v in self.params.items():
            se = self.se.get(p, np.nan)
            lines.append(f"  {p:>10s} {v:12.6g} {se:10.4g}   [{ci[p][0]:.5g}, {ci[p][1]:.5g}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        ps = ", ".join(
            f"{k}={v:.5g}" + (f"±{self.se[k]:.2g}" if np.isfinite(self.se.get(k, np.nan)) else "")
            for k, v in self.params.items()
        )
        return (
            f"FitResult({self.family}: {ps}; method={self.method!r}, loglik={self.loglik:.4f}, "
            f"aic={self.aic:.4f}, n={self.n})"
        )

    def rvs(self, n: int, random_state: Any = None) -> np.ndarray:
        """Sample from the fitted copula."""
        return self.copula.rvs(n, random_state=random_state)


# ---------------------------------------------------------------------------
# likelihood
# ---------------------------------------------------------------------------


def _uv(data: Any, pseudo_obs: bool) -> np.ndarray:
    if hasattr(data, "U") and hasattr(data, "data"):
        return np.asarray(data.U, dtype=float) if pseudo_obs else as_data(data.data, 2, 2)
    arr = as_data(data, min_dim=2, max_dim=2)
    if pseudo_obs:
        return _pseudo_obs(arr)
    if np.any(arr <= 0) or np.any(arr >= 1):
        raise ValueError("with pseudo_obs=False the data must lie in the open unit square.")
    return arr


def loglik(copula, data: Any, pseudo_obs: bool = True) -> float:
    r"""(Pseudo-)log-likelihood :math:`\sum_i \log c(\hat U_i, \hat V_i)` of a
    fully specified copula."""
    from copul.stats._adapters import logpdf

    U = _uv(data, pseudo_obs)
    return float(np.sum(logpdf(copula, U[:, 0], U[:, 1])))


def _hessian(f, x: np.ndarray, params: Sequence[_Param]) -> tuple[np.ndarray, bool]:
    """Central-difference Hessian (steps kept inside the parameter box) and
    whether it is numerically stable: the diagonal is recomputed with a
    quadrupled step and must agree within 25% (a noisy log-likelihood, e.g.
    from cancellation in a closed-form density, fails this check)."""
    p = x.size
    h = np.empty(p)
    for i, prm in enumerate(params):
        hi = 1e-4 * max(1.0, abs(x[i]))
        room = min(x[i] - prm.lo, prm.hi - x[i])
        if math.isfinite(room):
            hi = min(hi, 0.2 * room)
        h[i] = max(hi, 1e-10)
    f0 = f(x)
    H = np.empty((p, p))
    stable = True
    for i in range(p):
        ei = np.zeros(p)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / h[i] ** 2
        h4 = 4.0 * ei
        d4 = (f(x + h4) - 2.0 * f0 + f(x - h4)) / (16.0 * h[i] ** 2)
        if not abs(d4 - H[i, i]) <= 0.25 * max(abs(H[i, i]), abs(d4)):
            stable = False
        for j in range(i + 1, p):
            ej = np.zeros(p)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4.0 * h[i] * h[j])
    return H, stable


# ---------------------------------------------------------------------------
# moment methods
# ---------------------------------------------------------------------------


def _measure_value(copula, key: str) -> float:
    return float(compute(copula, key))


def _signed_bracket(base, pname: str, key: str, target: float, sign: float):
    """Bracket of the parameter on which ``key`` crosses ``target`` and
    Kendall's tau has the sign ``sign`` (for measures such as xi that do
    not distinguish positive from negative dependence)."""
    from copul.measures.curves import default_parameter_values

    grid = default_parameter_values(base, pname, 41)
    vals, taus = np.full(grid.size, np.nan), np.full(grid.size, np.nan)
    for i, t in enumerate(grid):
        try:
            c = _instance(base, [pname], [t])
            vals[i] = _measure_value(c, key) - target
            taus[i] = _measure_value(c, "tau")
        except Exception:
            pass
    for i in range(grid.size - 1):
        if (
            vals[i] * vals[i + 1] <= 0
            and np.sign(taus[i]) * sign >= 0
            and np.sign(taus[i + 1]) * sign >= 0
        ):
            return float(grid[i]), float(grid[i + 1])
    return None


def _invert_measure(
    base, pname: str, prm: _Param, key: str, target: float, sign: float | None = None
):
    """``(theta, clipped)``: parameter whose measure equals ``target`` (or the
    closest attainable one).  ``sign`` (the sign of the sample tau) selects
    the branch for sign-blind measures (``xi``)."""
    from copul.measures.curves import from_measure

    if sign is not None and key in ("xi", "xi_2"):
        try:
            br = _signed_bracket(base, pname, key, target, sign)
            if br is not None:
                c = from_measure(base, key, target, param=pname, bracket=br)
                return float(getattr(c, pname)), False
        except Exception as e:
            log.debug("signed inversion failed (%s)", e)
    try:
        c = from_measure(base, key, target, param=pname)
        return float(getattr(c, pname)), False
    except Exception as e:
        log.debug("from_measure failed (%s); scanning the parameter range", e)
    from copul.measures.curves import default_parameter_values

    grid = default_parameter_values(base, pname, 41)
    vals = np.full(grid.size, np.nan)
    for i, t in enumerate(grid):
        try:
            vals[i] = _measure_value(_instance(base, [pname], [t]), key)
        except Exception:
            pass
    if not np.any(np.isfinite(vals)):
        raise ValueError(f"cannot evaluate {key} for {type(base).__name__}")
    i = int(np.nanargmin(np.abs(vals - target)))
    return float(grid[i]), True


def _fit_moment(base, fam_name, U, method, fixed, n, light=False) -> FitResult:
    key = _MOMENT_METHODS[method]
    params = _params_of(base)
    if len(params) != 1:
        raise ValueError(
            f"method={method!r} needs exactly one free parameter, {fam_name} has "
            f"{[p.name for p in params]}; fix the others, e.g. {fam_name}({params[-1].name}=...)."
        )
    prm = params[0]
    target = sample_measure(U[:, 0], U[:, 1], key, random_state=0)
    sign = None
    if key in ("xi", "xi_2"):
        sign = float(np.sign(sample_measure(U[:, 0], U[:, 1], "tau")))
    theta, clipped = _invert_measure(base, prm.name, prm, key, target, sign=sign)
    copula = _instance(base, [prm.name], [theta])
    # delta method: se(theta) = se(kappa_n) / |d kappa / d theta|
    se = np.nan
    try:
        from copul.stats.inference import ASYMPTOTIC_MEASURES, asymptotic_variance

        if key in ASYMPTOTIC_MEASURES and not clipped and not light:
            se_k = math.sqrt(asymptotic_variance(U, key) / n)
            h = 1e-4 * max(1.0, abs(theta))
            room = min(theta - prm.lo, prm.hi - theta)
            if math.isfinite(room):
                h = min(h, 0.5 * room)
            d = (
                _measure_value(_instance(base, [prm.name], [theta + h]), key)
                - _measure_value(_instance(base, [prm.name], [theta - h]), key)
            ) / (2 * h)
            se = se_k / abs(d) if d != 0 else np.nan
    except Exception as e:  # pragma: no cover - diagnostic
        log.debug("delta-method se failed: %s", e)
    ll = np.nan
    if not light and not _is_singular(copula):
        try:
            ll = loglik(copula, U, pseudo_obs=False)
        except Exception:
            ll = np.nan
    msg = f"sample {key} = {target:.6g}"
    if clipped:
        msg += " outside the family's range; parameter set to the closest attainable value"
        warnings.warn(f"{fam_name}: {msg}.")
    return FitResult(
        family=fam_name,
        method=method,
        params={prm.name: theta},
        copula=copula,
        loglik=float(ll),
        n=n,
        k=1,
        se={prm.name: float(se)},
        cov=np.array([[se * se]]),
        converged=not clipped,
        message=msg,
        nfev=0,
        fixed=fixed,
        extra={key: target, "clipped": clipped},
    )


# ---------------------------------------------------------------------------
# maximum likelihood
# ---------------------------------------------------------------------------


def _user_start(params: list[_Param], start) -> np.ndarray:
    names = [p.name for p in params]
    if isinstance(start, Mapping):
        x0 = np.array([float(start.get(p.name, p.typical())) for p in params])
    else:
        x0 = np.atleast_1d(np.asarray(start, dtype=float))
        if x0.size != len(params):
            raise ValueError(f"start must have {len(params)} entries ({names}).")
    for i, p in enumerate(params):
        if not p.inside(x0[i]):
            raise ValueError(f"start value {names[i]}={x0[i]} outside ({p.lo}, {p.hi}).")
    return x0


def _grid_start(params: list[_Param], nll_theta, trusted=None) -> np.ndarray:
    """Coordinate-wise grid search of the log-likelihood (one sweep over
    each parameter, the others held at their current values).  Grid points
    where ``trusted`` (fast density available) is false are skipped unless
    none is trusted."""
    x = np.array([p.typical() for p in params])
    for i, p in enumerate(params):
        best, best_val = x[i], nll_theta(x)
        cand = list(p.grid())
        if trusted is not None:
            ok = [t for t in cand if trusted(np.concatenate([x[:i], [t], x[i + 1 :]]))]
            cand = ok or cand
        for t in cand:
            y = x.copy()
            y[i] = t
            val = nll_theta(y)
            if val < best_val:
                best, best_val = t, val
        x[i] = best
    return x


# compiled parametric log-densities, keyed by family class, fixed parameters
# and free parameter names (bounded LRU-style dictionary)
_DENSITY_CACHE: dict[tuple, ParametricLogDensity] = {}
_DENSITY_CACHE_SIZE = 64


def _density(base, names: list[str], fixed: dict[str, float], params) -> ParametricLogDensity:
    key = (type(base), tuple(sorted(fixed.items())), tuple(names))
    dens = _DENSITY_CACHE.get(key)
    if dens is None:
        dens = ParametricLogDensity(
            base,
            names,
            lambda t: _instance(base, names, t),
            center=[p.grid()[12] for p in params],
            grids=[p.grid() for p in params],
            bounds=[(p.lo, p.hi) for p in params],
        )
        if len(_DENSITY_CACHE) >= _DENSITY_CACHE_SIZE:
            _DENSITY_CACHE.pop(next(iter(_DENSITY_CACHE)))
        _DENSITY_CACHE[key] = dens
    return dens


def _density_cdf_mismatch(logdens, copula, a: float = 0.05, b: float = 0.95) -> float:
    r"""Consistency of density and cdf on :math:`[a,b]^2`: the difference of
    :math:`\iint_{[a,b]^2} c` (60-point tensor Gauss--Legendre rule; the
    density is smooth away from the corners) and the rectangle mass
    :math:`C(b,b) - C(a,b) - C(b,a) + C(a,a)`."""
    from copul.stats._adapters import cdf as model_cdf

    x, w = np.polynomial.legendre.leggauss(60)
    x = a + (b - a) * 0.5 * (x + 1.0)
    w = (b - a) * 0.5 * w
    uu, vv = np.meshgrid(x, x, indexing="ij")
    with np.errstate(all="ignore"):
        c = np.exp(np.asarray(logdens(uu.ravel(), vv.ravel()), dtype=float))
    if not np.all(np.isfinite(c)):
        return float("nan")
    integral = float(np.sum(np.outer(w, w).ravel() * c))
    cc = model_cdf(copula, np.array([b, a, b, a]), np.array([b, b, a, a]))
    return integral - float(cc[0] - cc[1] - cc[2] + cc[3])


def _fit_mle(base, fam_name, U, fixed, n, start, optimizer, options, light=False) -> FitResult:
    params = _params_of(base)
    if not params:
        raise ValueError(f"{fam_name} has no free parameters to estimate.")
    names = [p.name for p in params]
    x_ref = (
        _user_start(params, start) if start is not None else np.array([p.typical() for p in params])
    )
    if _is_singular(base) or _is_singular(_instance(base, names, x_ref)):
        raise SingularFamilyError(
            f"{fam_name} has a singular component (no density); maximum likelihood is "
            "not available. Use method='itau' (or 'irho', 'ixi', 'ibeta')."
        )
    dens = _density(base, names, fixed, params)
    u, v = U[:, 0], U[:, 1]
    nfev = [0]
    use_fast = [True]

    def nll_theta(theta: np.ndarray) -> float:
        nfev[0] += 1
        for p, t in zip(params, theta):
            if not p.inside(t):
                return np.inf
        try:
            lp = dens(theta, u, v, fast=use_fast[0])
        except Exception:
            return np.inf
        s = float(np.sum(lp))
        return -s if np.isfinite(s) else np.inf

    t0 = time.perf_counter()
    x0 = x_ref if start is not None else _grid_start(params, nll_theta, dens.trusted)

    big = [None]

    def nll_z(z: np.ndarray) -> float:
        theta = np.array([p.from_z(zi) for p, zi in zip(params, z)])
        val = nll_theta(theta)
        if not np.isfinite(val):
            # finite penalty keeps simplex/quasi-Newton methods well defined
            if big[0] is None:
                return 1e300
            return big[0] + 1e6
        if big[0] is None or val > big[0]:
            big[0] = val
        return val

    if not np.isfinite(nll_theta(x0)):
        raise ValueError(
            f"{fam_name}: log-likelihood is not finite at the start value "
            f"{dict(zip(names, map(float, x0)))}; pass start=..."
        )
    opts = dict(options or {})
    if optimizer is None:
        optimizer = "Nelder-Mead" if len(params) > 1 else "Powell"
    if optimizer == "Nelder-Mead":
        opts.setdefault("xatol", 1e-7)
        opts.setdefault("fatol", 1e-9)
        opts.setdefault("maxiter", 2000 * len(params))

    def optimize_from(x_start):
        big[0] = None
        z0 = np.array([p.to_z(t) for p, t in zip(params, x_start)])
        res = optimize.minimize(nll_z, z0, method=optimizer, options=opts)
        success, msg = bool(res.success), str(res.message)
        zbest, fbest = np.atleast_1d(res.x), float(res.fun)
        # polish with a quasi-Newton search from the optimum
        try:
            res2 = optimize.minimize(nll_z, zbest, method="BFGS", options={"gtol": 1e-7})
            if np.isfinite(res2.fun) and res2.fun < fbest - 1e-10:
                zbest = np.atleast_1d(res2.x)
                success = success or bool(res2.success) or "precision" in str(res2.message)
                msg += " (BFGS polish improved the optimum)"
        except Exception:  # pragma: no cover - defensive
            pass
        return np.array([p.from_z(zi) for p, zi in zip(params, zbest)]), success, msg

    theta, success, msg = optimize_from(x0)
    ll = -nll_theta(theta)
    copula = _instance(base, names, theta)
    if not light and dens.has_fast and dens.trusted(theta):
        # the log-expanded symbolic density can lose accuracy (cancellation)
        # far from the validation points: cross-check with the plain symbolic
        # density and re-optimize without the expansion if they disagree
        ll_plain = float(np.sum(dens(theta, u, v, fast=False)))
        if not np.isclose(ll, ll_plain, rtol=1e-6, atol=1e-6):
            use_fast[0] = False
            theta, success, msg = optimize_from(x0)
            msg += " (re-optimized without the log-expanded density)"
            copula = _instance(base, names, theta)
            ll = -nll_theta(theta)
    se = dict.fromkeys(names, np.nan)
    cov = None
    try:
        H, stable = (
            _hessian(nll_theta, theta, params)
            if not light
            else (np.full((len(params),) * 2, np.nan), True)
        )
        if not stable:
            warnings.warn(
                f"{fam_name}: the log-likelihood is numerically noisy near the estimate; "
                "standard errors are not available."
            )
            msg += "; numerically noisy log-likelihood (no standard errors)"
            H = np.full_like(H, np.nan)
        if np.all(np.isfinite(H)):
            cov = np.linalg.inv(H)
            d = np.diag(cov)
            if np.all(d > 0):
                se = {nm: float(math.sqrt(di)) for nm, di in zip(names, d)}
            else:
                warnings.warn(f"{fam_name}: observed information is not positive definite.")
    except np.linalg.LinAlgError:
        warnings.warn(f"{fam_name}: singular observed information matrix.")
    converged = bool(np.isfinite(ll)) and success
    at_bound = [nm for nm, p, t in zip(names, params, theta) if p.at_boundary(t)]
    if at_bound:
        msg += f"; estimate at (or diverging to) the boundary for {at_bound}"
    if not light:
        try:
            gap = _density_cdf_mismatch(lambda a, b: dens(theta, a, b, fast=use_fast[0]), copula)
        except Exception as e:  # pragma: no cover - diagnostic
            log.debug("density/cdf check failed: %s", e)
            gap = 0.0
        if not abs(gap) <= 0.01:
            converged = False
            msg += (
                f"; density and cdf are inconsistent at the estimate (mass of "
                f"[0.05, 0.95]^2 differs by {gap:.3g}; numerically unreliable density)"
            )
            warnings.warn(f"{fam_name}: density and cdf inconsistent at the estimate.")
    return FitResult(
        family=fam_name,
        method="mle",
        params=dict(zip(names, map(float, theta))),
        copula=copula,
        loglik=float(ll),
        n=n,
        k=len(params),
        se=se,
        cov=cov,
        converged=converged,
        message=msg,
        nfev=int(nfev[0]),
        fixed=fixed,
        extra={
            "density": dens.source,
            "optimizer": optimizer,
            "start": dict(zip(names, map(float, x0))),
            "seconds": time.perf_counter() - t0,
        },
    )


def _fixed_params(base) -> dict[str, float]:
    out = {}
    free = set(free_parameters(base))
    for p in list(getattr(type(base), "params", None) or []):
        name = str(p)
        if name in free:
            continue
        try:
            out[name] = float(getattr(base, name))
        except Exception:
            pass
    return out


def fit(
    family: Any,
    data: Any,
    method: str = "mle",
    start: Mapping[str, float] | Sequence[float] | None = None,
    pseudo_obs: bool = True,
    optimizer: str | None = None,
    options: dict | None = None,
    _light: bool = False,
) -> FitResult:
    r"""Fit a parametric copula family to bivariate data.

    Parameters
    ----------
    family : class, instance or str
        The family, e.g. ``cp.Clayton``, ``"GumbelHougaard"``, ``"T"`` or a
        partially specified instance such as ``cp.StudentT(nu=4)`` (fixed
        parameters are kept).
    data : array_like of shape (n, 2), DataFrame or EmpiricalCopula
        Raw data (transformed to pseudo-observations :math:`R_i/(n+1)`) or,
        with ``pseudo_obs=False``, observations in :math:`(0,1)^2`.
    method : {"mle", "itau", "irho", "ixi", "ibeta"}
        * ``"mle"``: maximum pseudo-likelihood
          :math:`\hat\theta = \arg\max_\theta\sum_i\log c_\theta(\hat U_i,
          \hat V_i)` (Genest, Ghoudi & Rivest, 1995), for any number of
          parameters;
        * ``"itau"`` / ``"irho"`` / ``"ixi"`` / ``"ibeta"``: solve
          :math:`\kappa(C_\theta) = \hat\kappa_n` for one free parameter
          (Genest & Rivest, 1993; Kojadinovic & Yan, 2010) using
          :func:`copul.measures.from_measure`; standard errors by the delta
          method :math:`\mathrm{se}(\hat\theta) = \mathrm{se}(\hat\kappa_n)/
          |\partial_\theta\kappa|`.  Sample values outside the family's range
          are mapped to the closest attainable parameter (with a warning).
    start : dict or sequence, optional
        Start values for ``mle`` (default: a coordinate-wise grid search of
        the log-likelihood over 25 values per parameter).  After the
        optimization the log-likelihood is cross-checked with the plain
        (non log-expanded) density and the total mass of the fitted density
        is checked by quadrature; failures set ``converged=False``.
    pseudo_obs : bool
        Rank-transform the data (default ``True``).
    optimizer : str, optional
        :func:`scipy.optimize.minimize` method (default Powell for one,
        Nelder--Mead for several parameters, each followed by a BFGS polish)
        on the unconstrained reparametrization.
    options : dict, optional
        Optimizer options.
    _light : bool
        Internal: skip standard errors (and the log-likelihood of moment
        estimates), used for bootstrap refits.

    Returns
    -------
    FitResult

    Raises
    ------
    SingularFamilyError
        For ``method="mle"`` if the family has a singular component.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.stats import fit
    >>> X = cp.Clayton(theta=2).rvs(1000, random_state=0)
    >>> res = fit(cp.Clayton, X)                 # doctest: +SKIP
    >>> res.params                               # doctest: +SKIP
    {'theta': 1.98...}
    >>> fit("Clayton", X, method="itau").params  # doctest: +SKIP
    {'theta': 2.0...}
    """
    base, fam_name = resolve_family(family)
    U = _uv(data, pseudo_obs)
    n = U.shape[0]
    method = str(method).lower()
    if method in ("ml", "mpl", "pmle"):
        method = "mle"
    fixed = _fixed_params(base)
    if method in _MOMENT_METHODS:
        return _fit_moment(base, fam_name, U, method, fixed, n, light=_light)
    if method != "mle":
        raise ValueError(
            f"method must be 'mle' or one of {sorted(_MOMENT_METHODS)}, got {method!r}"
        )
    return _fit_mle(base, fam_name, U, fixed, n, start, optimizer, options, light=_light)


# ---------------------------------------------------------------------------
# model selection
# ---------------------------------------------------------------------------


#: extremal/boundary constructions from exact-region research, excluded from
#: ``select(families="all")`` (they are not statistical models, and their cdf
#: is partly obtained by symbolic integration, which is very slow)
_NON_MODEL_FAMILIES = frozenset(
    {"DIAGONAL_BAND", "XI_NU_BOUNDARY", "XI_PSI_BOUNDARY", "XI_RHO_BOUNDARY"}
)


def _all_family_names() -> list[str]:
    from copul.family_list import Families

    out, seen = [], set()
    for name in Families.list_all():
        try:
            cls = Families[name].cls
        except Exception:  # pragma: no cover
            continue
        if cls in seen or name in _NON_MODEL_FAMILIES:
            continue
        seen.add(cls)
        out.append(name)
    return out


def select(
    data: Any,
    families: Iterable[Any] | str | None = None,
    criterion: str = "aic",
    method: str = "mle",
    pseudo_obs: bool = True,
    **fit_kwargs: Any,
):
    """Fit several families and rank them by an information criterion.

    Parameters
    ----------
    data : array_like of shape (n, 2)
    families : iterable, "all" or None
        Families (anything accepted by :func:`fit`); default
        :data:`DEFAULT_FAMILIES`; ``"all"`` fits every parametric family of
        :class:`copul.family_list.Families` except the boundary copulas of
        exact-region research (families with a singular component are listed
        with an error and ranked last).  Families without a symbolic density
        (extreme-value, elliptical) are fitted through per-parameter numeric
        backends, which can take tens of seconds each for three-parameter
        families.
    criterion : {"aic", "bic", "loglik"}
    method : str
        Estimation method passed to :func:`fit` (``"mle"`` by default; with a
        moment method the log-likelihood is evaluated at the moment estimate).

    Returns
    -------
    pandas.DataFrame
        One row per family, sorted by the criterion (best first), with
        columns ``family, params, loglik, aic, bic, k, converged, error,
        fit`` (the :class:`FitResult` objects).
    """
    criterion = criterion.lower()
    if criterion not in ("aic", "bic", "loglik"):
        raise ValueError("criterion must be 'aic', 'bic' or 'loglik'")
    if families is None:
        fams: list[Any] = list(DEFAULT_FAMILIES)
    elif isinstance(families, str):
        fams = _all_family_names() if families.lower() == "all" else [families]
    else:
        fams = list(families)
    U = _uv(data, pseudo_obs)
    rows = []
    for fam in fams:
        label = fam if isinstance(fam, str) else _pretty_name(fam)
        row = {
            "family": label,
            "params": None,
            "loglik": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "k": np.nan,
            "converged": False,
            "error": "",
            "fit": None,
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit(fam, U, method=method, pseudo_obs=False, **fit_kwargs)
            row.update(
                family=res.family,
                params=res.params,
                loglik=res.loglik,
                aic=res.aic,
                bic=res.bic,
                k=res.k,
                converged=res.converged,
                fit=res,
            )
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            log.info("select: %s failed: %s", label, e)
        rows.append(row)
    df = pd.DataFrame(rows)
    if criterion == "loglik":
        df = df.sort_values("loglik", ascending=False, na_position="last")
    else:
        df = df.sort_values(criterion, ascending=True, na_position="last")
    return df.reset_index(drop=True)
