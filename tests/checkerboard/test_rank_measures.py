"""Independent integrals and dispatch checks for the exact checkerboard atlas."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtri

import copul as cp
from copul.checkerboard import _rank_measures as rank
from copul.measures import ATLAS_MEASURES, compute, get_measure

P = np.array([[1 / 4, 1 / 8, 1 / 8], [1 / 12, 5 / 24, 5 / 24]])
SIGNS = np.array([[0, 1, -1], [-1, 0, 1]])


@pytest.mark.parametrize("key", ATLAS_MEASURES)
@pytest.mark.parametrize(
    "factory, endpoint",
    [(cp.BivCheckPi, "at_Pi"), (cp.BivCheckMin, "at_M"), (cp.BivCheckW, "at_W")],
)
def test_closed_endpoints(key, factory, endpoint):
    result = compute(factory([[1]]), key, method="closed", full_output=True)
    assert result.method == "closed"
    assert result.error == 0
    assert result.value == pytest.approx(getattr(get_measure(key), endpoint), abs=2e-14)


@pytest.mark.parametrize("key", ATLAS_MEASURES)
def test_pi_atlas_closed_dispatch(key, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Closed checkerboard route must not fall back to the measure engine")

    monkeypatch.setattr("copul.measures.atlas.evaluate_copula", unexpected)
    c = cp.BivCheckPi(P)
    result = compute(c, key, full_output=True)
    assert result.method == "closed"
    assert result.error == 0
    assert result.value == pytest.approx(getattr(c, get_measure(key).method_name)(method="closed"))


@pytest.mark.parametrize("key", [k for k in ATLAS_MEASURES if k != "normal_score"])
def test_mixed_closed_dispatch(key):
    result = compute(cp.BivCheckMixed(P, SIGNS), key, method="closed", full_output=True)
    assert result.method == "closed"
    assert np.isfinite(result.value)


def _nodes(n=6):
    x, w = np.polynomial.legendre.leggauss(n)
    return (1 + x) / 2, w / 2


def _cdf_moments(c, signs):
    """Independent degree-exact quadrature of the definition, split at every kink."""
    m, n = c.matr.shape
    t, w = _nodes()
    a = t[:, None]
    phi, d = 0.0, 0.0
    for i, j in np.ndindex(c.matr.shape):
        sign = signs[i, j]
        split = a if sign > 0 else 1 - a
        intervals = [(0, 1)] if sign == 0 else [(0, split), (split, 1)]
        cell = 0.0
        for low, high in intervals:
            b = low + (high - low) * t[None, :]
            u, v = (i + a) / m, (j + b) / n
            cell += w @ ((c.cdf(u, v) - u * v) ** 2 * (high - low)) @ w
        phi += cell / (m * n)
        if sign:
            u, v = (i + t) / m, (j + (t if sign > 0 else 1 - t)) / n
            cell = w @ ((c.cdf(u, v) - u * v) ** 2)
        d += c.matr[i, j] * cell
    return 90 * phi, 30 * d


@pytest.mark.parametrize(
    "signs", [np.zeros_like(SIGNS), np.ones_like(SIGNS), -np.ones_like(SIGNS), SIGNS]
)
def test_hoeffding_integrals_and_tau_star(signs):
    c = cp.BivCheckMixed(P, signs)
    phi, d = _cdf_moments(c, signs)
    assert c.hoeffdings_d(method="closed") == pytest.approx(phi, abs=2e-14)
    assert c.hoeffding_d(method="closed") == pytest.approx(d, abs=2e-14)
    assert c.blum_kiefer_rosenblatt(method="closed") == pytest.approx(d, abs=2e-14)
    assert c.bergsma_dassios(method="closed") == pytest.approx(0.4 * d + 4 * phi / 15, abs=2e-14)


def _sigma_reference(c, signs):
    m, n = c.matr.shape
    total = 0.0
    for i, j in np.ndindex(c.matr.shape):

        def outer(a):
            u = (i + a) / m
            sign = signs[i, j]
            split = a if sign > 0 else 1 - a
            intervals = [(0, 1)] if sign == 0 else [(0, split), (split, 1)]
            result = 0.0
            for lo, hi in intervals:

                def difference(b):
                    v = (j + b) / n
                    return c.cdf(u, v) - u * v

                fl, fh = difference(lo), difference(hi)
                points = [lo - fl * (hi - lo) / (fh - fl)] if fl * fh < 0 else []
                result += quad(lambda b: abs(difference(b)), lo, hi, points=points, epsabs=1e-12)[0]
            return result

        total += quad(outer, 0, 1, epsabs=1e-11, limit=100)[0]
    return 12 * total / (m * n)


@pytest.mark.parametrize("seed", range(4))
def test_sigma_crossing_zero_curves(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 1, (3, 4))
    for _ in range(100):
        p /= 3 * p.sum(1)[:, None]
        p /= 4 * p.sum(0)[None, :]
    signs = rng.integers(-1, 2, p.shape)
    c = cp.BivCheckMixed(p, signs)
    expected = _sigma_reference(c, signs)
    assert c.schweizer_wolff_sigma(method="closed") == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("sign", [0, 1, -1])
def test_sigma_near_independence_and_zero_slopes(sign):
    p = np.array([[1.0 + 1e-9, 1.0 - 1e-9], [1.0 - 1e-9, 1.0 + 1e-9]]) / 4
    signs = np.full(p.shape, sign)
    c = cp.BivCheckMixed(p, signs)
    assert c.schweizer_wolff_sigma(method="closed") == pytest.approx(
        _sigma_reference(c, signs), abs=1e-12
    )


@pytest.mark.parametrize("signs", [np.zeros_like(SIGNS), SIGNS, -SIGNS])
def test_integrated_r2_against_conditional_integral(signs):
    c = cp.BivCheckMixed(P, signs)
    m, n = P.shape

    def outer(v):
        j = min(int(n * v), n - 1)
        b = n * v - j
        points = list(np.arange(1, m) / m)
        for i in range(m):
            if signs[i, j]:
                points.append((i + (b if signs[i, j] > 0 else 1 - b)) / m)
        return quad(
            lambda u: (c.cond_distr_1(u, v) - v) ** 2 / (v * (1 - v)),
            0,
            1,
            points=points,
            epsabs=1e-11,
        )[0]

    expected = quad(outer, 0, 1, points=np.arange(1, n) / n, epsabs=1e-10)[0]
    assert c.integrated_r2(method="closed") == pytest.approx(expected, abs=1e-10)
    ct = cp.BivCheckMixed(P.T, signs.T)
    assert c.integrated_r2(condition_on_y=True, method="closed") == pytest.approx(
        ct.integrated_r2()
    )
    assert c.correlation_ratio(condition_on_y=True, method="closed") == pytest.approx(
        ct.correlation_ratio()
    )


@pytest.mark.parametrize("n", [1, 2, 17, 128])
def test_integrated_r2_logarithmic_moments_on_fine_grids(n):
    # An invertible shuffle has A=1, irrespective of the number of cells.
    p = np.eye(n)[::-1] / n
    assert rank.integrated_r2(p, 1) == pytest.approx(1, abs=1e-12)
    assert rank.integrated_r2(np.ones((2, n)) / (2 * n)) == pytest.approx(0, abs=1e-14)


def test_uniform_distance_interior_singular_extremum():
    # 1x1 has zero deviations at every corner; its extremum is on a diagonal.
    assert cp.BivCheckMin([[1]]).uniform_distance(method="closed") == 1
    assert cp.BivCheckW([[1]]).uniform_distance(method="closed") == 1
    c = cp.BivCheckMixed(P, SIGNS)
    t = np.linspace(0, 1, 1201)
    dense = 4 * np.max(np.abs(c.cdf(t[:, None], t[None, :]) - t[:, None] * t[None, :]))
    assert c.uniform_distance(method="closed") == pytest.approx(dense, abs=2e-6)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("a,b", [(1, 1), (1, 2), (2, 1), (3, 5), (64, 63), (128, 128)])
def test_savage_log_product_primitives(a, b, sign):
    expected = quad(
        lambda t: np.log(a - t) * np.log(b - (t if sign > 0 else 1 - t)), 0, 1, epsabs=1e-11
    )[0]
    formula = rank._log_product_min if sign > 0 else rank._log_product_w
    assert formula(a, b) == pytest.approx(expected, abs=3e-11)


def test_mixed_savage_against_score_expectation():
    c = cp.BivCheckMixed(P, SIGNS)
    total = 0.0
    for i, j in np.ndindex(P.shape):
        score_u = lambda t: -np.log1p(-(i + t) / 2)
        score_v = lambda t: -np.log1p(-(j + t) / 3)
        sign = SIGNS[i, j]
        if sign:
            product = quad(lambda t: score_u(t) * score_v(t if sign > 0 else 1 - t), 0, 1)[0]
        else:
            product = quad(score_u, 0, 1)[0] * quad(score_v, 0, 1)[0]
        total += P[i, j] * product
    assert c.savage_score_correlation(method="closed") == pytest.approx(total - 1, abs=1e-11)


def test_normal_score_closed_and_fallback():
    for cls, p, expected in [
        (cp.BivCheckMin, np.eye(3) / 3, 1),
        (cp.BivCheckW, np.eye(3)[::-1] / 3, -1),
    ]:
        assert cls(p).normal_score_correlation(method="closed") == pytest.approx(expected)
    c = cp.BivCheckMixed(P, SIGNS)
    with pytest.raises(NotImplementedError, match="off-diagonal singular"):
        c.normal_score_correlation(method="closed")
    result = compute(c, "normal_score", full_output=True)
    assert result.method == "numeric"
    expected = 0.0
    for i, j in np.ndindex(P.shape):
        sign = SIGNS[i, j]
        if sign:
            product = quad(
                lambda t: ndtri((i + t) / 2) * ndtri((j + (t if sign > 0 else 1 - t)) / 3), 0, 1
            )[0]
        else:
            product = (
                quad(lambda t: ndtri((i + t) / 2), 0, 1)[0]
                * quad(lambda t: ndtri((j + t) / 3), 0, 1)[0]
            )
        expected += P[i, j] * product
    assert result.value == pytest.approx(expected, abs=1e-8)


BERNSTEIN_FGM = {
    "eta": lambda t: t / 3,
    "omega": lambda t: abs(t) / np.sqrt(15),
    "hoeffdings_d": lambda t: t * t / 10,
    "hoeffding_d": lambda t: t * t / 30,
    "tau_star": lambda t: t * t / 25,
    "chi": lambda t: t * t / 9,
    "integrated_r2": lambda t: t * t / 18,
    "savage_score": lambda t: t / 4,
    "maximal_correlation": lambda t: abs(t) / 3,
}


@pytest.mark.parametrize("t", [0.0, 1e-10, 0.6])
def test_omega_stable_near_independence(t):
    p = np.array([[1 + t, 1 - t], [1 - t, 1 + t]]) / 4
    assert cp.BivCheckPi(p).siburg_stoimenov(method="closed") == pytest.approx(
        t / np.sqrt(2), abs=1e-15
    )
    assert cp.BivBernstein(p).siburg_stoimenov(method="closed") == pytest.approx(
        t / np.sqrt(15), abs=1e-15
    )


@pytest.mark.parametrize("key", ATLAS_MEASURES)
@pytest.mark.parametrize("pi", [[1, 2, 3], [2, 3, 1]])
def test_shuffle_matches_checkerboard(key, pi):
    c = cp.ShuffleOfMin(pi)
    if key == "normal_score" and pi != sorted(pi):
        with pytest.raises(NotImplementedError, match="off-diagonal singular"):
            compute(c, key, method="closed")
        assert compute(c, key) == pytest.approx(compute(c.as_checkerboard(), key), abs=1e-10)
    else:
        actual = compute(c, key, method="closed", full_output=True)
        assert actual.method == "closed"
        assert actual.value == pytest.approx(compute(c.as_checkerboard(), key), abs=1e-12)


@pytest.mark.parametrize("key", ATLAS_MEASURES)
def test_block_diagonal_inherits_closed_formulas(key):
    c = cp.BivBlockDiagMixed([1, 2])
    result = compute(c, key, method="closed", full_output=True)
    assert result.method == "closed"
    assert np.isfinite(result.value)


@pytest.mark.parametrize("key", BERNSTEIN_FGM)
@pytest.mark.parametrize("t", [-0.8, 0, 0.6])
def test_bernstein_closed_form_fgm(key, t):
    c = cp.BivBernstein([[1 + t, 1 - t], [1 - t, 1 + t]])
    result = compute(c, key, method="closed", full_output=True)
    assert result.method == "closed"
    assert result.value == pytest.approx(BERNSTEIN_FGM[key](t), abs=3e-8)


@pytest.mark.parametrize("key", BERNSTEIN_FGM)
def test_bernstein_rectangular_against_independent_numeric(key):
    c = cp.BivBernstein(P)
    closed = compute(c, key, method="closed")
    numeric = compute(c, key, method="numeric", max_grid=128)
    assert closed == pytest.approx(numeric, abs=3e-5 if key == "maximal_correlation" else 1e-8)
    if key in ("chi", "integrated_r2"):
        assert compute(c, key, method="closed", condition_on_y=True) == pytest.approx(
            compute(c.transpose(), key)
        )
