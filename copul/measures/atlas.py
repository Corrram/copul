"""Numerical population coefficients from the pairwise rank-correlation atlas.

Definitions follow the coefficient table of ``pairwise-exact-regions.tex``.
No pairwise regions are implemented here. CDF and conditional-distribution
formulas apply to singular copulas; divergences use the density of the
absolutely continuous part, with singular mass accounted for separately.
"""

import math

import numpy as np
from scipy.linalg import svdvals
from scipy.special import ndtri

from copul.measures.numeric import _bkr_h, _call, _i2, _lp_cdf, _mi_pdf, _xi_h
from copul.measures.quadrature import integrate_1d, integrate_1d_batch

KEYS = {
    "eta",
    "normal_score",
    "savage_score",
    "omega",
    "hoeffding_d",
    "tau_star",
    "chi",
    "maximal_correlation",
    "integrated_r2",
    "linfoot",
    "hellinger",
    "silvey",
}


def _hellinger(affinity):
    """Rationalize the defining formula to avoid cancellation at B=0 and B=1."""
    b = float(np.clip(affinity, 0, 1))
    x = b**4
    # 1-b**4 in factored form retains accuracy near independence.
    return math.sqrt(4 * (1 - b) * (1 + b) * (1 + b * b) / (2 - x + math.sqrt(4 - 3 * x)))


def _transformed(value, error, transform):
    result = transform(value)
    delta = max(abs(transform(value - error) - result), abs(transform(value + error) - result))
    return result, delta


def _linfoot(information):
    return math.sqrt(-math.expm1(-2 * max(0.0, information)))


def _density(funcs):
    """Require a density of the AC part, not a numerical derivative across jumps."""
    if not hasattr(funcs, "source") and funcs.pdf is None:
        raise NotImplementedError("Density divergences require an explicit pdf callable.")
    pdf = funcs.get("pdf")
    if getattr(funcs, "source", {}).get("pdf") == "finite_differences":
        raise NotImplementedError(
            "Density divergences require an explicit density of the absolutely continuous "
            "part; a finite-difference density cannot reliably distinguish singular mass."
        )
    return pdf


def _chi(h, rtol, atol, breaks=None):
    ub, vb = (None, None) if breaks is None else breaks
    inner_error = 0.0

    def integrand(u):
        nonlocal inner_error
        u = np.asarray(u)
        mean, err = integrate_1d_batch(
            lambda v, row: _call(h, u[row], v) - v,
            np.zeros(u.size),
            np.ones(u.size),
            rtol=rtol / 4,
            atol=atol / 48,
            breaks=vb,
            jump_search=True,
        )
        inner_error = max(inner_error, float(np.max(err)))
        return 12 * mean**2

    value, err = integrate_1d(integrand, rtol=rtol, atol=atol / 2, breaks=ub)
    return value, err + 12 * (inner_error + inner_error**2)


def _partition_correlation(masses):
    """Norm of the centered conditional-expectation matrix of finite categories."""
    p = np.asarray(masses, float)
    rows, cols = p.sum(1), p.sum(0)
    p = p[np.ix_(rows > 0, cols > 0)]
    rows, cols = rows[rows > 0], cols[cols > 0]
    normalized = p / np.sqrt(rows[:, None] * cols[None, :])
    centered = normalized - np.sqrt(rows[:, None] * cols[None, :])
    return float(np.clip(svdvals(centered)[0], 0, 1))


def _validate_grid(max_grid):
    if isinstance(max_grid, (bool, np.bool_)) or not isinstance(max_grid, (int, np.integer)):
        raise ValueError("max_grid must be a power of two of at least 16")
    if max_grid < 16 or max_grid & (max_grid - 1):
        raise ValueError("max_grid must be a power of two of at least 16")


def _maximal(cdf, max_grid=256):
    """Dyadic Galerkin lower approximations of the centered Markov operator norm.

    Always use the requested resolution: successive equal coarse answers do
    not exclude unresolved fine-scale dependence. The returned refinement
    difference is not an upper bound on the approximation error.
    """
    _validate_grid(max_grid)
    t = np.arange(max_grid + 1) / max_grid
    grid = np.empty((t.size, t.size))
    grid[1:-1, 1:-1] = _call(cdf, t[1:-1, None], t[None, 1:-1])
    grid[0, :] = grid[:, 0] = 0
    grid[-1, :] = grid[:, -1] = t
    p = np.diff(np.diff(grid, axis=0), axis=1)
    if np.min(p) < -1e-9:
        raise ValueError("CDF evaluation produced negative cell masses; use a more accurate CDF")
    p = np.maximum(p, 0)
    p /= p.sum()
    value = _partition_correlation(p)
    coarse = p.reshape(max_grid // 2, 2, max_grid // 2, 2).sum(axis=(1, 3))
    return value, abs(value - _partition_correlation(coarse))


def evaluate(key, funcs, rtol, atol):
    """Evaluate an additional coefficient from the shared numerical backend."""
    if key == "eta":
        value, err = _i2(
            lambda u, v: (2 - u - v) * (_call(funcs.get("cdf"), u, v) - u * v), rtol, atol, 12
        )
        return 12 * value, 12 * err
    if key in ("normal_score", "savage_score"):
        cdf = funcs.get("cdf")

        def integrand(u, v):
            diff = _call(cdf, u, v) - u * v
            if key == "savage_score":
                return diff / ((1 - u) * (1 - v))
            return diff * (2 * np.pi) * np.exp((ndtri(u) ** 2 + ndtri(v) ** 2) / 2)

        return _i2(integrand, rtol, atol, 1)
    if key == "omega":
        x, ex = _xi_h(funcs.get("h1"), rtol, atol)
        y, ey = _xi_h(funcs.get("h2"), rtol, atol)
        return _transformed((x + y) / 2, (ex + ey) / 2, lambda z: math.sqrt(max(0.0, z)))
    if key in ("hoeffding_d", "tau_star"):
        d, ed = _bkr_h(funcs.get("cdf"), funcs.get("h1"), funcs.get("h2"), rtol, atol)
        if key == "hoeffding_d":
            return d, ed
        phi, ep = _lp_cdf(funcs.get("cdf"), 2, rtol, atol)
        return 0.4 * d + 4 * phi / 15, 0.4 * ed + 4 * ep / 15
    if key == "chi":
        return _chi(funcs.get("h1"), rtol, atol, funcs.breaks)
    if key == "integrated_r2":
        h = funcs.get("h1")
        return _i2(lambda u, v: (_call(h, u, v) - v) ** 2 / (v * (1 - v)), rtol, atol, 1)
    if key == "maximal_correlation":
        return _maximal(funcs.get("cdf"))
    if key == "linfoot":
        # A known singular component implies infinite mutual information,
        # however small its mass; do not infer it by a quadrature tolerance.
        if getattr(funcs, "_ac", True) is False:
            return 1.0, 0.0
        value, err = _mi_pdf(_density(funcs), rtol, atol)
        return _transformed(value, err, _linfoot)
    if key in ("hellinger", "silvey"):
        pdf = _density(funcs)
        if key == "hellinger":
            b, eb = _i2(lambda u, v: np.sqrt(np.maximum(_call(pdf, u, v), 0)), rtol, atol, 1)
            return _transformed(b, eb, _hellinger)
        # (int |c-1| + 1-int c)/2 = int (1-c)_+; includes singular mass.
        return _i2(lambda u, v: np.maximum(1 - _call(pdf, u, v), 0), rtol, atol, 1)
    raise KeyError(key)


def _checkerboard(copula, key, rtol, atol, transpose=False):
    """Share the checkerboard module's exact formulas with the numeric route."""
    from copul.checkerboard import _rank_measures as rank
    from copul.measures.registry import get_measure

    p, s = rank._inputs(copula.matr, copula._kernel_signs(), transpose)
    formula = getattr(rank, get_measure(key).method_name, None)
    if formula is None:
        return None
    try:
        return formula(p, s), 0.0
    except NotImplementedError:
        if key != "normal_score":
            raise
    # Off-diagonal singular normal-score products have no closed formula.
    m, n = p.shape
    a, b = rank.score_means(m, True), rank.score_means(n, True)
    value = float(np.sum(p * a[:, None] * b[None, :] * (s == 0)))
    error = 0.0
    for i, j in np.argwhere((s != 0) & (p > 0)):
        val, err = integrate_1d(
            lambda t, i=i, j=j: ndtri((i + t) / m) * ndtri((j + (t if s[i, j] > 0 else 1 - t)) / n),
            rtol=rtol,
            atol=atol,
        )
        value += p[i, j] * val
        error += p[i, j] * err
    return value, error


def evaluate_copula(copula, key, options, rtol, atol):
    """Return a special route (value, error, source), or None for shared quadrature.

    For a*M+b*W+(1-a-b)*Pi, the centered conditional CDF is a*(hM-v)+
    b*(hW-v). Squaring gives xi=a²+b²-ab; integration with weight
    1/[v(1-v)] gives A=a²+b²+2ab*(1-2log(2)). Its regression is
    (a-b)*(u-1/2), and its even centered Markov eigenfunctions have
    eigenvalue a+b. Its AC density is the constant 1-a-b.
    Gaussian Hermite modes have eigenvalues rho**k, so R=|rho|;
    its KL information is -log(1-rho²)/2 and its affinity satisfies
    B**4=16*(1-rho²)/(4-rho²)**2, giving L=H=|rho|.
    """
    if key not in KEYS:
        return None
    from copul.checkerboard._biv_mixin import BivCheckerboardMixin
    from copul.checkerboard.shuffle_min import ShuffleOfMin
    from copul.family.elliptical.gaussian import Gaussian
    from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
    from copul.family.frechet.frechet import Frechet
    from copul.family.frechet.lower_frechet import LowerFrechet
    from copul.family.frechet.upper_frechet import UpperFrechet
    from copul.measures.backend import numeric_backend
    from copul.measures.registry import get_measure

    if key == "maximal_correlation":
        _validate_grid(options.get("max_grid", 256))
    spec = get_measure(key)
    for cls, value in (
        (BivIndependenceCopula, spec.at_Pi),
        (UpperFrechet, spec.at_M),
        (LowerFrechet, spec.at_W),
    ):
        if isinstance(copula, cls):
            return float(value), 0.0, "endpoint"
    if isinstance(copula, Gaussian) and key in (
        "normal_score",
        "maximal_correlation",
        "linfoot",
        "hellinger",
    ):
        rho = float(copula.rho)
        return rho if key == "normal_score" else abs(rho), 0.0, "gaussian"
    if isinstance(copula, Frechet):
        a, b = float(copula.alpha), float(copula.beta)
        values = {
            "eta": a - b,
            "normal_score": a - b,
            "savage_score": a + b * (1 - np.pi**2 / 6),
            "omega": math.sqrt(max(0.0, a * a + b * b - a * b)),
            "chi": (a - b) ** 2,
            "integrated_r2": a * a + b * b + 2 * a * b * (1 - 2 * np.log(2)),
            "maximal_correlation": a + b,
            "linfoot": float(a + b > 0),
            "hellinger": _hellinger(math.sqrt(max(0.0, 1 - a - b))),
            "silvey": a + b,
        }
        if key in values:
            return float(values[key]), 0.0, "frechet_mixture"
    if isinstance(copula, ShuffleOfMin) and key in (
        "omega",
        "chi",
        "integrated_r2",
        "maximal_correlation",
        "linfoot",
        "hellinger",
        "silvey",
    ):
        return 1.0, 0.0, "shuffle"
    if isinstance(copula, ShuffleOfMin):
        result = _checkerboard(copula.as_checkerboard(), key, rtol, atol)
        if result is not None:
            return *map(float, result), "shuffle"
    if isinstance(copula, BivCheckerboardMixin):
        result = _checkerboard(copula, key, rtol, atol, options.get("condition_on_y", False))
        if result is not None:
            return *map(float, result), "checkerboard"
    be = numeric_backend(copula)
    if key == "maximal_correlation":
        val, err = _maximal(be.get("cdf"), options.get("max_grid", 256))
        return val, err, "partition_lower_approximation (refinement difference, not an error bound)"
    if options.get("condition_on_y") and key in ("chi", "integrated_r2"):
        from copul.measures.numeric import NumericCopula
        from copul.measures.numeric import evaluate as numeric_evaluate

        h2 = be.get("h2")
        transposed = NumericCopula(
            h1=lambda u, v: h2(v, u), breaks=None if be.breaks is None else be.breaks[::-1]
        )
        val, err = numeric_evaluate(key, transposed, rtol=rtol, atol=atol)
        return val, err, "transposed_conditional"
    return None
