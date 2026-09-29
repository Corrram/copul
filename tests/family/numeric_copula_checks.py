"""Shared numerical checks for vectorised bivariate copula objects."""

import numpy as np

GRID = np.linspace(0.0, 1.0, 26)
INTERIOR = np.array([0.03, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.97])


def _diff(f, x, h=2e-6):
    """Fourth-order central difference quotient."""
    return (8 * (f(x + h) - f(x - h)) - (f(x + 2 * h) - f(x - 2 * h))) / (12 * h)


def check_axioms(C, tol=1e-10):
    """Groundedness, uniform margins, 2-increasingness and Fréchet bounds on a grid."""
    U, V = np.meshgrid(GRID, GRID, indexing="ij")
    Z = np.asarray(C.cdf_vectorized(U, V), dtype=float)
    assert np.all(np.isfinite(Z))
    np.testing.assert_allclose(Z[0, :], 0.0, atol=tol)
    np.testing.assert_allclose(Z[:, 0], 0.0, atol=tol)
    np.testing.assert_allclose(Z[-1, :], GRID, atol=tol)
    np.testing.assert_allclose(Z[:, -1], GRID, atol=tol)
    mass = np.diff(np.diff(Z, axis=0), axis=1)
    assert mass.min() >= -1e-10, f"negative rectangle mass {mass.min()}"
    assert np.all(np.maximum(U + V - 1, 0) - tol <= Z)
    assert np.all(np.minimum(U, V) + tol >= Z)


def check_conditionals(C, rtol=1e-4, atol=1e-5):
    """h-functions against central differences of the cdf; ranges in [0, 1]."""
    U, V = np.meshgrid(INTERIOR, INTERIOR, indexing="ij")
    u, v = U.ravel(), V.ravel()
    c = C.cdf_vectorized
    fd1 = _diff(lambda x: c(x, v), u)
    fd2 = _diff(lambda y: c(u, y), v)
    h1 = np.asarray(C.cond_distr_1(u, v))
    h2 = np.asarray(C.cond_distr_2(u, v))
    for val in (h1, h2):
        assert np.all((val >= 0) & (val <= 1))
    np.testing.assert_allclose(h1, fd1, rtol=rtol, atol=atol)
    np.testing.assert_allclose(h2, fd2, rtol=rtol, atol=atol)


def check_density(C, rtol=1e-4, atol=1e-4, total=True):
    """Density against differences of h1, non-negativity and total mass 1."""
    U, V = np.meshgrid(INTERIOR, INTERIOR, indexing="ij")
    u, v = U.ravel(), V.ravel()
    fd = _diff(lambda y: np.asarray(C.cond_distr_1(u, y)), v)
    c = np.asarray(C.pdf(u, v))
    assert np.all(c >= 0)
    np.testing.assert_allclose(c, fd, rtol=rtol, atol=atol)
    if total:
        # the conditional densities v -> c(u, v) integrate to one (so does c);
        # breakpoints at the diagonals, where tail-dependent densities peak
        from copul.measures.quadrature import integrate_1d

        for u0 in (0.1, 0.35, 0.5, 0.8):
            val, _ = integrate_1d(
                lambda y, u0=u0: C.pdf_vectorized(np.full_like(y, u0), y),
                atol=1e-11,
                rtol=1e-11,
                breaks=sorted({u0, 1 - u0}),
            )
            assert abs(val - 1.0) < 1e-6, (u0, val)


def check_sampling(C, n=20_000, seed=7, tol=None):
    """Empirical cdf of ``rvs`` against the cdf (seeded, deterministic)."""
    x = np.asarray(C.rvs(n, random_state=seed))
    assert x.shape == (n, 2)
    assert np.all((x >= 0) & (x <= 1))
    tol = tol if tol is not None else 5.0 * 0.5 / np.sqrt(n)
    pts = [(0.1, 0.1), (0.2, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.9), (0.3, 1.0), (1.0, 0.6)]
    for a, b in pts:
        emp = np.mean((x[:, 0] <= a) & (x[:, 1] <= b))
        assert abs(emp - C.cdf(a, b)) < tol, (a, b, emp, C.cdf(a, b))
    if not C.is_absolutely_continuous:
        return
    # probability integral transform of the conditional law is uniform
    from scipy import stats

    w = np.asarray(C.cond_distr_1(x[:, 0], x[:, 1]))
    assert stats.kstest(w, "uniform").pvalue > 1e-3


def closed_vs_numeric(C, keys, tol=1e-7):
    """For every key evaluated by a closed form, compare with ``method='numeric'``.

    Returns the list of keys that were closed forms.
    """
    closed = []
    for key in keys:
        res = C.measure(key, full_output=True)
        if res.method != "closed":
            continue
        closed.append(key)
        num = C.measure(key, method="numeric", full_output=True)
        t = tol if key not in ("lambda_l", "lambda_u") else 1e-5
        assert abs(res.value - num.value) < t + 10 * (num.error or 0), (
            key,
            res.value,
            num.value,
            num.error,
        )
    return closed
