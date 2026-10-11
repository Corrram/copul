r"""Sequential and joint likelihood fitting of simplified regular vines.

Tree selection follows the maximum spanning tree heuristic using absolute
empirical Kendall tau (Dißmann et al., 2013, doi:10.1016/j.csda.2012.08.010).
Conditional observations are propagated through fitted h-functions, never
ranked again. Sequential estimation is distinct from joint maximum
likelihood; optional joint refinement recomputes all conditional arguments
at every objective evaluation (Hobæk Haff, 2013, doi:10.3150/12-BEJ413).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np
from scipy import optimize, stats

from copul.family.constructions.rotation import TransformedCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.stats.fitting import FitResult, _params_of, fit, resolve_family
from copul.stats.pseudo_obs import pseudo_obs as to_pseudo_obs
from copul.vine.model import CVineCopula, DVineCopula, VineCopula, interior, pair_backend
from copul.vine.structure import VineEdge, VineStructure, integer, permutation

DEFAULT_PAIR_FAMILIES = ("Independence", "Gaussian", "Clayton", "Frank", "GumbelHougaard", "T")


@dataclass(frozen=True)
class PairCopulaFamily:
    """Candidate family and optional reflection for vine fitting.

    Parameters
    ----------
    family : class, instance or str
        A family accepted by ``copul.fit``. Specified parameters stay fixed;
        a fully specified copula is a candidate with zero fitted parameters.
        ``"Independence"`` is also accepted.
    rotation : {0, 90, 180, 270}
        Standard vine convention: reflect the first variable at 90, both
        at 180, the second at 270. For asymmetric pairs these are reflections,
        not geometric scatter-plot rotations. The parameter count is unchanged.
    """

    family: object
    rotation: int = 0

    def __post_init__(self):
        integer(self.rotation, "rotation")
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be 0, 90, 180 or 270.")

    def base(self):
        if isinstance(self.family, str) and self.family.lower() in ("independence", "indep"):
            return BivIndependenceCopula(), "Independence"
        return resolve_family(self.family)

    def transform_data(self, U):
        U = U.copy()
        if self.rotation in (90, 180):
            U[:, 0] = 1 - U[:, 0]
        if self.rotation in (180, 270):
            U[:, 1] = 1 - U[:, 1]
        return U

    def wrap(self, copula):
        if not self.rotation:
            return copula
        return TransformedCopula(
            copula, flip_u=self.rotation in (90, 180), flip_v=self.rotation in (180, 270)
        )


@dataclass
class VineFitResult:
    """Fitted vine and likelihood diagnostics.

    ``edge_fits`` are in tree order and hold final conditional log-likelihoods.
    ``k`` counts estimated pair parameters; AIC/BIC do not penalize the
    discrete search over structures/families. No naive pair standard errors
    are reported as joint uncertainty. Bootstrap the entire selection and
    fitting procedure if such uncertainty is required.
    """

    copula: VineCopula
    loglik: float
    n: int
    k: int
    method: str
    converged: bool
    message: str
    edge_fits: tuple[tuple[FitResult, ...], ...]
    sequential_loglik: float
    selection_failures: tuple[str, ...] = ()
    optimizer_result: object = field(default=None, repr=False)

    @property
    def aic(self):
        """AIC = 2k - 2 loglik, counting only estimated pair parameters."""
        return 2 * self.k - 2 * self.loglik

    @property
    def bic(self):
        """BIC = k log(n) - 2 loglik."""
        return self.k * np.log(self.n) - 2 * self.loglik

    @property
    def structure(self):
        """Selected or supplied regular-vine structure."""
        return self.copula.structure

    def rvs(self, n=1, random_state=None):
        """Sample from the fitted copula."""
        return self.copula.rvs(n, random_state)

    def summary(self):
        """Human-readable likelihood, parameter count and convergence status."""
        return (
            f"Vine fit: d={self.copula.dim}, n={self.n}, method={self.method}, k={self.k}\n"
            f"log-likelihood={self.loglik:.6g}, AIC={self.aic:.6g}, BIC={self.bic:.6g}\n"
            f"converged={self.converged}: {self.message}"
        )


def _prepare(data, pseudo_obs, ties):
    U = np.asarray(data, float)
    if U.ndim != 2 or U.shape[0] < 2 or U.shape[1] < 2:
        raise ValueError("data must be an (n, d) array with n >= 2 and d >= 2.")
    if not np.all(np.isfinite(U)):
        raise ValueError("data must be finite; missing values are not silently removed.")
    if np.any(np.ptp(U, axis=0) == 0):
        raise ValueError("Constant columns cannot identify a continuous copula.")
    if pseudo_obs:
        U = to_pseudo_obs(U, ties=ties)
    if np.any((U <= 0) | (U >= 1)):
        raise ValueError("Copula-scale data must lie in the open unit cube.")
    return U


def _tau(x, y):
    tau = stats.kendalltau(x, y).statistic
    return abs(float(tau)) if np.isfinite(tau) else 0.0


def _select_pair(U, candidates, criterion, failures, label, options):
    best = None
    for spec in candidates:
        try:
            base, name = spec.base()
            if _params_of(base):
                res = fit(
                    base, spec.transform_data(U), pseudo_obs=False, options=options, _light=True
                )
                cop = spec.wrap(res.copula)
            else:
                cop = spec.wrap(base)
                res = FitResult(name, "fixed", {}, cop, 0.0, len(U), 0)
            be = pair_backend(cop)
            ll = float(be.logpdf(U[:, 0], U[:, 1]).sum())
            if not np.isfinite(ll) or not res.converged:
                raise ValueError(f"non-finite likelihood or failed optimization: {res.message}")
            res = replace(res, copula=cop, loglik=ll, family=f"{name}[{spec.rotation}]")
            score = getattr(res, criterion) if criterion != "loglik" else -ll
            if best is None or score < best[0]:
                best = (score, res, spec)
        except (ValueError, TypeError, FloatingPointError, NotImplementedError) as exc:
            failures.append(f"{label}, {spec.family!r}, rotation={spec.rotation}: {exc}")
    if best is None:
        raise ValueError(
            f"No candidate pair could be fitted at {label}. "
            + "; ".join(failures[-len(candidates) :])
        )
    return best[1], best[2]


def _mst(nodes, previous_links, cache):
    candidates = []
    for i, j in combinations(range(len(nodes)), 2):
        if previous_links is not None and not (previous_links[i] & previous_links[j]):
            continue
        D = nodes[i] & nodes[j]
        aa, bb = nodes[i] - D, nodes[j] - D
        if len(aa) != 1 or len(bb) != 1:
            continue
        a, b = next(iter(aa)), next(iter(bb))
        score = _tau(cache[a, D], cache[b, D])
        candidates.append((-score, i, j, VineEdge(a, b, tuple(D))))
    candidates.sort(key=lambda x: x[:3])
    parents = list(range(len(nodes)))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    edges, links = [], []
    for _, i, j, edge in candidates:
        ri, rj = root(i), root(j)
        if ri != rj:
            parents[ri] = rj
            edges.append(edge)
            links.append(frozenset((i, j)))
            if len(edges) == len(nodes) - 1:
                break
    if len(edges) != len(nodes) - 1:
        raise ValueError("No spanning tree satisfies the proximity condition.")
    return edges, links


def _path_order(U):
    # Greedy path, initialized by strongest marginal pair; not a TSP optimum.
    d = U.shape[1]
    weights = {(i, j): _tau(U[:, i], U[:, j]) for i, j in combinations(range(d), 2)}
    start = max(weights, key=weights.get)
    path, left = list(start), set(range(d)) - set(start)
    while left:
        choices = [
            (weights[tuple(sorted((end, j)))], -j, side, j)
            for side, end in enumerate((path[0], path[-1]))
            for j in sorted(left)
        ]
        _, _, side, j = max(choices)
        path.insert(0 if side == 0 else len(path), j)
        left.remove(j)
    return tuple(path)


def _make_model(kind, structure, pairs):
    if kind == "cvine":
        return CVineCopula(structure.dim, pairs, structure.order)
    if kind == "dvine":
        return DVineCopula(structure.dim, pairs, structure.order)
    return VineCopula(structure, pairs)


def _joint(model, U, specs, edge_fits, options):
    layout, start = [], []
    for tree_specs, tree_fits in zip(specs, edge_fits):
        row = []
        for spec, res in zip(tree_specs, tree_fits):
            base, _ = spec.base()
            params = _params_of(base)
            row.append((spec, base, params))
            start.extend(p.to_z(res.params[p.name]) for p in params)
        layout.append(row)
    if not start:
        return model, None

    def construct(z):
        pairs, offset = [], 0
        for row in layout:
            pt = []
            for spec, base, params in row:
                values = {p.name: p.from_z(z[offset + j]) for j, p in enumerate(params)}
                offset += len(params)
                pt.append(spec.wrap(base(**values) if values else base))
            pairs.append(pt)
        return VineCopula(model.structure, pairs)

    def objective(z):
        try:
            value = -construct(z).loglik(U)
            return value if np.isfinite(value) else 1e100
        except (ValueError, FloatingPointError, OverflowError):
            return 1e100

    opts = {"maxiter": 200, "ftol": 1e-9, **(options or {})}
    res = optimize.minimize(objective, np.array(start), method="L-BFGS-B", options=opts)
    if np.isfinite(res.fun) and res.fun <= -model.loglik(U):
        updated = construct(res.x)
        kind = (
            "cvine"
            if isinstance(model, CVineCopula)
            else "dvine"
            if isinstance(model, DVineCopula)
            else "rvine"
        )
        model = _make_model(kind, updated.structure, updated.pair_copulas)
        offset = 0
        for t, row in enumerate(layout):
            for j, (_, _, params) in enumerate(row):
                values = {p.name: p.from_z(res.x[offset + k]) for k, p in enumerate(params)}
                offset += len(params)
                edge_fits[t][j] = replace(edge_fits[t][j], params=values)
    return model, res


def fit_vine(
    data,
    *,
    vine_type="rvine",
    structure=None,
    order=None,
    families=None,
    criterion="aic",
    pseudo_obs=True,
    ties="average",
    truncation=None,
    joint=False,
    pair_options=None,
    joint_options=None,
):
    r"""Fit a simplified vine to continuous multivariate observations.

    Parameters
    ----------
    data : array_like of shape (n, d)
        Raw observations, or uniforms when ``pseudo_obs=False``.
    vine_type : {"rvine", "cvine", "dvine"}
        R-vines use sequential maximum spanning trees of absolute Kendall
        tau, restricted by proximity. C-vines greedily select each root by
        its sum of conditional absolute taus. D-vines use a greedy marginal
        path heuristic. These searches do not claim global optimality.
    structure : VineStructure, optional
        Fix a general regular-vine structure; use ``vine_type='rvine'``.
    order : sequence of int, optional
        Fix C-vine roots or D-vine path. For R-vines supply ``structure``.
    families : iterable, optional
        Candidates accepted by :class:`PairCopulaFamily`, including classes,
        names and partially/fully specified copulas. Default: independence,
        Gaussian, Clayton, Frank, Gumbel and Student-t. Add reflected
        candidates explicitly to cover asymmetric negative dependence.
    criterion : {"aic", "bic", "loglik"}
        Pair-family selection criterion. Parameter counts exclude fixed
        parameters and the discrete structure/family search.
    pseudo_obs : bool
        Rank-transform *once*, before fitting any tree, using R/(n+1).
    ties : str
        Deterministic tie method for ``copul.pseudo_obs`` (default average).
        Tied data are treated as continuous; this is not a discrete likelihood.
    truncation : int, optional
        Number of non-independent trees, from 0 to d-1. Later trees use
        independence and contribute neither parameters nor log-density.
    joint : bool
        Refine all free parameters jointly, holding selected structure and
        families fixed. Keep the sequential solution if refinement is worse.
    pair_options, joint_options : dict, optional
        Options for the pair optimizer and joint L-BFGS-B optimizer.

    Returns
    -------
    VineFitResult
        Model, selected trees, pair fits, failures, convergence and AIC/BIC.
        ``VineCopula.fit`` instead returns its ``copula`` directly.
    """
    if vine_type not in ("rvine", "cvine", "dvine"):
        raise ValueError("vine_type must be 'rvine', 'cvine' or 'dvine'.")
    if criterion not in ("aic", "bic", "loglik"):
        raise ValueError("criterion must be 'aic', 'bic' or 'loglik'.")
    if ties == "random":
        raise ValueError("Use a deterministic ties method, or rank the data externally.")
    U = _prepare(data, pseudo_obs, ties)
    n, d = U.shape
    truncation = d - 1 if truncation is None else integer(truncation, "truncation")
    if truncation > d - 1:
        raise ValueError("truncation must be between 0 and d-1.")
    if structure is not None:
        if vine_type != "rvine" or order is not None:
            raise ValueError("Use structure with vine_type='rvine' and without order.")
        if not isinstance(structure, VineStructure) or structure.dim != d:
            raise ValueError("structure must be a VineStructure matching the data dimension.")
        structure = VineStructure(structure.trees, structure.order)
    if order is not None:
        if vine_type == "rvine":
            raise ValueError("For an R-vine, specify an explicit structure instead of order.")
        order = permutation(order, d)
    if vine_type == "dvine":
        order = _path_order(U) if order is None else order
        structure = VineStructure.dvine(order)
    elif vine_type == "cvine" and order is not None:
        structure = VineStructure.cvine(order)
    families = DEFAULT_PAIR_FAMILIES if families is None else families
    if isinstance(families, (str, PairCopulaFamily, type)):
        families = [families]
    candidates = tuple(
        f if isinstance(f, PairCopulaFamily) else PairCopulaFamily(f) for f in families
    )
    if not candidates:
        raise ValueError("families must contain at least one candidate.")
    # Fail invalid specifications early, even when truncation=0.
    for spec in candidates:
        spec.base()
    cache = {(i, frozenset()): U[:, i] for i in range(d)}
    nodes, links = [frozenset((i,)) for i in range(d)], None
    roots, trees, pairs, edge_fits, specs, failures = [], [], [], [], [], []
    for t in range(d - 1):
        if structure is not None:
            edges = structure.trees[t]
        elif vine_type == "cvine":
            D = frozenset(roots)
            remaining = sorted(set(range(d)) - D)
            scores = {
                a: sum(_tau(cache[a, D], cache[b, D]) for b in remaining if b != a)
                for a in remaining
            }
            root = max(remaining, key=scores.get)
            edges = [VineEdge(root, b, tuple(D)) for b in remaining if b != root]
            roots.append(root)
        else:
            edges, links = _mst(nodes, links, cache)
        pt, ft, st = [], [], []
        for j, e in enumerate(edges):
            D = frozenset(e.conditioning)
            x, y = cache[e.a, D], cache[e.b, D]
            choices = candidates if t < truncation else (PairCopulaFamily("Independence"),)
            res, spec = _select_pair(
                np.column_stack((x, y)),
                choices,
                criterion,
                failures,
                f"tree {t + 1}, edge {j + 1}",
                pair_options,
            )
            be = pair_backend(res.copula)
            cache[e.a, D | {e.b}] = interior(be.h2(x, y))
            cache[e.b, D | {e.a}] = interior(be.h1(x, y))
            pt.append(res.copula)
            ft.append(res)
            st.append(spec)
        trees.append(edges)
        pairs.append(pt)
        edge_fits.append(ft)
        specs.append(st)
        nodes = [e.complete for e in edges]
    if structure is None:
        if vine_type == "cvine":
            roots.extend(sorted(set(range(d)) - set(roots)))
            # Fitted edges retain their original per-tree order, which can
            # differ from the subsequently selected root order.
            structure = VineStructure(trees, order=roots)
        else:
            structure = VineStructure(trees)
    if vine_type == "cvine":
        # Align edge lists with the canonical constructor's root order.
        canonical = VineStructure.cvine(structure.order)
        for t, tree in enumerate(canonical.trees):
            indices = [structure.trees[t].index(e) for e in tree]
            pairs[t] = [pairs[t][i] for i in indices]
            edge_fits[t] = [edge_fits[t][i] for i in indices]
            specs[t] = [specs[t][i] for i in indices]
        structure = canonical
    model = _make_model(vine_type, structure, pairs)
    sequential_ll = model.loglik(U)
    optimizer_result = None
    if joint:
        model, optimizer_result = _joint(model, U, specs, edge_fits, joint_options)
    # Recompute every edge contribution at the final model parameters.
    final_cache = model._initial(U)
    for t, tree in enumerate(model.structure.trees):
        for j, e in enumerate(tree):
            D = frozenset(e.conditioning)
            x = model._conditional(e.a, D, final_cache)
            y = model._conditional(e.b, D, final_cache)
            ll = float(model._backends[t][j].logpdf(x, y).sum())
            edge_fits[t][j] = replace(edge_fits[t][j], copula=model.pair_copulas[t][j], loglik=ll)
    converged = optimizer_result is None or bool(optimizer_result.success)
    message = "Sequential pair maximum likelihood."
    if optimizer_result is not None:
        message = str(optimizer_result.message)
    result = VineFitResult(
        model,
        model.loglik(U),
        n,
        sum(r.k for tree in edge_fits for r in tree),
        "joint_mle" if joint else "sequential",
        converged,
        message,
        tuple(tuple(tree) for tree in edge_fits),
        sequential_ll,
        tuple(failures),
        optimizer_result,
    )
    model.fit_result_ = result
    return result
