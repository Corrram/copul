r"""Maximum likelihood for finite mixtures of bivariate copula densities.

For weights pi_j and component parameters theta_j, the objective is
sum_i log(sum_j pi_j c_j(U_i; theta_j)). EM uses responsibilities
r_ij = pi_j c_j(U_i) / sum_l pi_l c_l(U_i), weight updates
pi_j = mean_i r_ij, and weighted component maximum likelihood.

This is a generalized EM algorithm: a numerical component update is only
accepted if it increases its weighted objective, and every complete update
is checked against the observed likelihood. See Dempster, Laird & Rubin
(1977), JRSS B 39, 1--22, doi:10.1111/j.2517-6161.1977.tb01600.x.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from copul.family.constructions._base import component_is_ac, ensure_numeric_copula
from copul.family.constructions.mixture import MixtureCopula
from copul.measures.backend import numeric_backend
from copul.stats._utils import _positive_integer, as_rng
from copul.stats.fitting import _fit_mle, _fixed_params, _params_of, _user_start, _uv


def _log_components(components, U):
    return np.column_stack([numeric_backend(c).logpdf(U[:, 0], U[:, 1]) for c in components])


def _expectation(components, weights, U):
    with np.errstate(divide="ignore"):
        joint = _log_components(components, U) + np.log(weights)
    marginal = logsumexp(joint, axis=1)
    if not np.all(np.isfinite(marginal)):
        raise ValueError("The mixture must have a finite positive density at every observation.")
    return float(marginal.sum()), np.exp(joint - marginal[:, None])


@dataclass
class MixtureFitResult:
    """Fitted mixture and diagnostics for each starting point.

    ``component_params`` contains only estimated parameters; supplied
    numeric parameters stay fixed. ``k`` is the nominal count of component
    parameters plus m-1 estimated weights. Mixtures can be nonidentifiable
    or have parameters on the boundary; AIC/BIC are descriptive scores,
    not a justification for ordinary likelihood-ratio or Wald inference.
    """

    copula: MixtureCopula
    component_params: tuple[dict[str, float], ...]
    loglik: float
    n: int
    k: int
    converged: bool
    message: str
    n_iter: int
    loglik_path: np.ndarray
    starts: tuple[dict, ...]

    @property
    def weights(self):
        """Fitted component probabilities, in the supplied family order."""
        return self.copula.weights.copy()

    @property
    def aic(self):
        """Nominal AIC = 2k - 2 loglik (see class caveat)."""
        return 2 * self.k - 2 * self.loglik

    @property
    def bic(self):
        """Nominal BIC = k log(n) - 2 loglik (see class caveat)."""
        return self.k * np.log(self.n) - 2 * self.loglik

    def predict_proba(self, data):
        """Posterior component probabilities for copula-scale (n, 2) data.

        New raw data need their original marginal transformations; this
        method intentionally does not rank new observations independently.
        """
        U = np.asarray(data, float)
        if (
            U.ndim != 2
            or U.shape[1] != 2
            or not np.all(np.isfinite(U))
            or np.any((U <= 0) | (U >= 1))
        ):
            raise ValueError("data must be a finite (n, 2) array in the open unit square.")
        return _expectation(self.copula.copulas, self.weights, U)[1]

    def rvs(self, n=1, random_state=None):
        """Sample from the fitted mixture."""
        return self.copula.rvs(n, random_state=random_state)

    def summary(self):
        """Likelihood, weights and optimizer status."""
        return (
            f"Copula mixture: n={self.n}, components={len(self.weights)}, k={self.k}\n"
            f"log-likelihood={self.loglik:.6g}, AIC={self.aic:.6g}, BIC={self.bic:.6g}\n"
            f"weights={self.weights}, converged={self.converged}: {self.message}"
        )


def fit_mixture(
    data,
    families,
    *,
    pseudo_obs=True,
    weights=None,
    fit_weights=True,
    start=None,
    n_init=3,
    max_iter=200,
    tol=1e-7,
    random_state=None,
    options=None,
):
    """Fit a finite mixture by multiple-start generalized EM.

    Parameters
    ----------
    data : array_like of shape (n, 2)
        Raw observations, or uniforms with ``pseudo_obs=False``. Ranking
        occurs once before EM, never separately within components.
    families : sequence
        Bivariate classes, names, partially/fully specified copulas, or
        ``PairCopulaFamily`` objects for reflected families. Repeated
        families are allowed; their component labels are interchangeable.
    pseudo_obs : bool
        Rank-transform each column before fitting (default True).
    weights : sequence, optional
        Strictly positive starting weights, normalized to sum to one.
    fit_weights : bool
        Estimate weights (default), or keep the supplied weights fixed.
    start : sequence of dict or None, optional
        One initial parameter dictionary per component. Dictionaries
        specify exactly the free parameters; None uses a full-data fit.
    n_init : int
        Number of starts. After the first, free parameters are perturbed
        on an unconstrained scale and estimated weights are randomized.
    max_iter : int
        Maximum EM iterations per start.
    tol : float
        Stop when likelihood gain <= tol * (1 + abs(previous likelihood)).
    random_state : int, Generator or None
        Reproducible random starts; global NumPy state is not changed.
    options : dict, optional
        Passed to component likelihood optimizers.

    Returns
    -------
    MixtureFitResult
        The highest-likelihood run, even if it reached the iteration limit;
        inspect ``converged`` and ``starts``. Likelihoods use log-sum-exp.
        Components with vanishing weight are retained, not silently removed.

    Notes
    -----
    Only absolutely continuous components are supported. EM finds local
    optima; multiple starts do not guarantee the global maximum. Degenerate
    or duplicate components invalidate ordinary regular-model inference.
    """
    from copul.vine.fitting import PairCopulaFamily

    n_init = _positive_integer(n_init, "n_init")
    max_iter = _positive_integer(max_iter, "max_iter")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")
    U = _uv(data, pseudo_obs)
    specs = tuple(f if isinstance(f, PairCopulaFamily) else PairCopulaFamily(f) for f in families)
    if not specs:
        raise ValueError("families must contain at least one component.")
    bases, labels = zip(*(spec.base() for spec in specs))
    params = [_params_of(base) for base in bases]
    m, n = len(specs), len(U)
    w0 = np.ones(m) if weights is None else np.asarray(weights, float)
    if (
        w0.shape != (m,)
        or not np.all(np.isfinite(w0))
        or np.any(w0 <= 0)
        or not np.isfinite(w0.sum())
    ):
        raise ValueError("weights must be finite and strictly positive, one per component.")
    w0 = w0 / w0.sum()
    if start is None:
        start = [None] * m
    if len(start) != m:
        raise ValueError("start must have one entry per component.")
    transformed = [spec.transform_data(U) for spec in specs]
    initial = []
    for base, label, ps, supplied, X in zip(bases, labels, params, start, transformed):
        if ps:
            if supplied is None:
                fitted = _fit_mle(
                    base, label, X, _fixed_params(base), n, None, None, options, light=True
                )
                if not fitted.converged:
                    raise ValueError(f"Initialization failed for {label}: {fitted.message}")
                values = fitted.params
            else:
                if not isinstance(supplied, Mapping) or set(supplied) != {p.name for p in ps}:
                    raise ValueError(f"start for {label} must specify exactly its free parameters.")
                theta = _user_start(ps, supplied)
                if not all(p.inside(v) for p, v in zip(ps, theta)):
                    raise ValueError(f"start for {label} must be inside the parameter domain.")
                values = dict(zip((p.name for p in ps), theta))
            candidate = base(**values)
        else:
            if supplied:
                raise ValueError(
                    f"{label} has no free parameters; its start must be None or empty."
                )
            candidate, values = base, {}
        ensure_numeric_copula(candidate)
        if not component_is_ac(candidate):
            raise ValueError(
                f"Mixture fitting requires an absolutely continuous component: {label}."
            )
        initial.append(values)
    rng = as_rng(random_state)
    runs, solutions = [], []
    for run in range(n_init):
        values = deepcopy(initial)
        if run:
            for j, ps in enumerate(params):
                for p in ps:
                    values[j][p.name] = p.from_z(p.to_z(values[j][p.name]) + rng.normal(0, 1.5))
        components = [
            spec.wrap(base(**v) if v else deepcopy(base))
            for spec, base, v in zip(specs, bases, values)
        ]
        w = rng.dirichlet(np.ones(m)) if run and fit_weights else w0.copy()
        path, converged, message = [], False, "Maximum EM iterations reached."
        try:
            ll, responsibility = _expectation(components, w, U)
            path.append(ll)
            for _ in range(max_iter):
                next_w = responsibility.mean(axis=0) if fit_weights else w
                next_components, next_values = list(components), deepcopy(values)
                for j, ps in enumerate(params):
                    r = responsibility[:, j]
                    if not ps or not np.any(r > 0):
                        continue
                    fit = _fit_mle(
                        bases[j],
                        labels[j],
                        transformed[j],
                        _fixed_params(bases[j]),
                        n,
                        values[j],
                        None,
                        options,
                        light=True,
                        weights=r,
                    )
                    if not fit.converged:
                        raise RuntimeError(f"Component {j} optimizer failed: {fit.message}")
                    candidate = specs[j].wrap(fit.copula)
                    mask = r > 0
                    old_q = np.dot(r[mask], numeric_backend(components[j]).logpdf(*U[mask].T))
                    new_q = np.dot(r[mask], numeric_backend(candidate).logpdf(*U[mask].T))
                    if np.isfinite(new_q) and new_q >= old_q:
                        next_components[j], next_values[j] = candidate, fit.params
                next_ll, next_r = _expectation(next_components, next_w, U)
                gain = next_ll - ll
                if gain < 0:
                    message = "Numerical likelihood decrease; retained previous iterate."
                    if abs(gain) <= 1e-10 * (1 + abs(ll)):
                        converged, message = True, "EM stationary within floating point precision."
                    break
                components, values, w = next_components, next_values, next_w
                path.append(next_ll)
                ll, responsibility = next_ll, next_r
                if gain <= tol * (1 + abs(path[-2])):
                    converged, message = True, "EM likelihood tolerance reached."
                    break
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            message = str(exc)
        runs.append(
            {
                "start": run,
                "loglik": path[-1] if path else -np.inf,
                "converged": converged,
                "message": message,
                "n_iter": max(0, len(path) - 1),
            }
        )
        solutions.append((components, values, w, np.asarray(path)))
    best = max(range(n_init), key=lambda i: runs[i]["loglik"])
    if not np.isfinite(runs[best]["loglik"]):
        raise RuntimeError("All mixture starts failed: " + "; ".join(r["message"] for r in runs))
    components, values, w, path = solutions[best]
    status = runs[best]
    if np.any(w < 1e-6):
        status["message"] += " A component has negligible weight; inference is nonregular."
    model = MixtureCopula(components, w)
    return MixtureFitResult(
        model,
        tuple(values),
        status["loglik"],
        n,
        sum(len(ps) for ps in params) + (m - 1 if fit_weights else 0),
        status["converged"],
        status["message"],
        status["n_iter"],
        path,
        tuple(runs),
    )
