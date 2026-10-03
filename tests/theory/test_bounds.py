"""Tests for copul.theory.bounds (best-possible pointwise bounds on sets of copulas).

Validation of the published closed forms (Nelsen 2006, Thm 3.2.3; Nelsen,
Quesada-Molina, Rodríguez-Lallena & Úbeda-Flores 2001, 2004):

* the shuffles of M returned for C(a, b) = theta equal the formulas of
  Thm 3.2.3, pass through (a, b, theta), are copulas and contain random
  checkerboard copulas with the same value at (a, b);
* their Kendall's tau / Spearman's rho closed forms agree with independent
  numerical integration along the support;
* random checkerboard copulas (BivCheckPi.generate_diverse, BivCheckMin,
  BivCheckW) with tau = t (rho = t) lie pointwise between T_t^L and T_t^U
  (P_t^L and P_t^U), also for tau >= t / tau <= t;
* the bounds are attained pointwise: at every (a, b) the shuffle
  C_U^{a,b,theta*} with theta* = T_t^L(a, b) has tau = t (and similarly for
  the upper bound and for rho);
* all four bounds are copulas (no negative rectangle volume on fine grids).
"""

import numpy as np
import pytest

import copul as cp
from copul.theory.bounds import (
    BoundsResult,
    MeasureBoundCopula,
    ShuffleOfM,
    blomqvist_beta_bounds,
    bounds_given_diagonal,
    bounds_given_measure,
    bounds_given_value,
    check_frechet_bounds,
    diagonal_upper_bound,
    frechet_lower,
    frechet_upper,
    kendall_tau_bounds,
    kendall_tau_lower_bound,
    kendall_tau_upper_bound,
    point_value_lower,
    point_value_upper,
    spearman_rho_bounds,
    spearman_rho_cubic_root,
    spearman_rho_lower_bound,
    spearman_rho_upper_bound,
)
from copul.theory.diagonal import DiagonalCopula, diagonal_section
from copul.theory.quasi import NumericQuasiCopula, is_copula, two_increasing_defect

G = np.linspace(0.0, 1.0, 41)
U, V = np.meshgrid(G, G, indexing="ij")
TOL = 1e-12


@pytest.fixture(scope="module")
def random_copulas():
    pis = cp.BivCheckPi.generate_diverse(n_samples=40, grid_size=(2, 10), rng=11)
    return (
        pis + [cp.BivCheckMin(C.matr) for C in pis[:20]] + [cp.BivCheckW(C.matr) for C in pis[20:]]
    )


@pytest.fixture(scope="module")
def measures(random_copulas):
    return [
        (C, C.cdf_vectorized(U, V), C.kendalls_tau(), C.spearmans_rho()) for C in random_copulas
    ]


# ---------------------------------------------------------------------------
# Frechet-Hoeffding bounds
# ---------------------------------------------------------------------------


def test_frechet_helpers():
    assert frechet_lower(0.7, 0.6) == pytest.approx(0.3)
    assert frechet_upper(0.7, 0.6) == pytest.approx(0.6)
    assert isinstance(frechet_lower(0.2, 0.3), float)
    assert frechet_lower(G, 0.5).shape == G.shape
    for C in (cp.Clayton(2), cp.Frank(-6), cp.LowerFrechet(), cp.UpperFrechet()):
        assert check_frechet_bounds(C, m=40)
    assert not check_frechet_bounds(lambda u, v: np.minimum(1.5 * u * v, 1.0), m=40)


def test_random_copulas_within_frechet_bounds(measures):
    for _, z, _, _ in measures:
        assert np.all(z >= frechet_lower(U, V) - TOL)
        assert np.all(z <= frechet_upper(U, V) + TOL)


# ---------------------------------------------------------------------------
# shuffles of M with arbitrary strips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C",
    [
        ShuffleOfM([(0.0, 0.5, 0.5, 1), (0.5, 0.0, 0.5, 1)]),
        ShuffleOfM.from_permutation([0.2, 0.5, 0.3], [2, 0, 1], flips=[True, False, True]),
        ShuffleOfM.from_permutation([0.15, 0.25, 0.1, 0.5], [1, 3, 0, 2], flips=[0, 1, 1, 0]),
        ShuffleOfM([(0.0, 0.0, 1.0, -1)]),
    ],
    ids=repr,
)
def test_shuffle_of_m_closed_forms(C):
    # uniform margins and copula axioms
    assert np.allclose(C.cdf(G, 1.0), G, atol=1e-14)
    assert np.allclose(C.cdf(1.0, G), G, atol=1e-14)
    assert is_copula(C, m=40)
    # closed forms vs independent integrals along the support V = phi(U)
    x = (np.arange(200_000) + 0.5) / 200_000
    phi = C.support(x)
    assert C.kendalls_tau() == pytest.approx(4 * np.mean(C.cdf_vectorized(x, phi)) - 1, abs=1e-6)
    assert C.spearmans_rho() == pytest.approx(12 * np.mean(x * phi) - 3, abs=1e-8)
    assert C.spearmans_footrule() == pytest.approx(C.spearmans_footrule(method="numeric"), abs=1e-7)
    assert C.ginis_gamma() == pytest.approx(C.ginis_gamma(method="numeric"), abs=1e-7)
    assert C.chatterjees_xi() == 1.0
    # h-functions are indicators of the support
    assert C.cond_distr_1(0.37, float(C.support(0.37)) + 1e-9) == 1.0
    assert C.cond_distr_1(0.37, float(C.support(0.37)) - 1e-9) == 0.0
    # samples lie on the support
    s = C.rvs(500, random_state=0)
    assert np.allclose(s[:, 1], C.support(s[:, 0]))


def test_shuffle_of_m_tail_coefficients():
    C = bounds_given_value(0.3, 0.6, 0.2).upper  # diagonal strips at both corners
    assert C.lambda_L() == pytest.approx(1.0)
    assert C.lambda_U() == pytest.approx(1.0)
    assert bounds_given_value(0.3, 0.6, 0.2).lower.lambda_L() == pytest.approx(0.0)
    assert ShuffleOfM([(0.0, 0.5, 0.5, 1), (0.5, 0.0, 0.5, 1)]).lambda_U() == pytest.approx(0.0)


def test_shuffle_of_m_validation():
    with pytest.raises(ValueError, match="partition"):
        ShuffleOfM([(0.0, 0.0, 0.5, 1), (0.6, 0.5, 0.4, 1)])
    with pytest.raises(ValueError, match="direction"):
        ShuffleOfM([(0.0, 0.0, 1.0, 2)])
    with pytest.raises(ValueError):
        ShuffleOfM([])
    with pytest.raises(ValueError, match="permutation"):
        ShuffleOfM.from_permutation([0.5, 0.5], [0, 0])


# ---------------------------------------------------------------------------
# bounds given C(a, b) = theta (Nelsen 2006, Thm 3.2.3)
# ---------------------------------------------------------------------------


def _random_points(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a, b = rng.random(2)
        lo, hi = max(a + b - 1, 0), min(a, b)
        out.append((a, b, lo + rng.random() * (hi - lo)))
    return out


@pytest.mark.parametrize(("a", "b", "theta"), _random_points(12, 0))
def test_point_value_bounds_are_the_shuffles_of_theorem_323(a, b, theta):
    res = bounds_given_value(a, b, theta)
    lo, up = res
    assert isinstance(lo, ShuffleOfM) and isinstance(up, ShuffleOfM)
    assert np.allclose(lo.cdf(U, V), point_value_lower(U, V, a, b, theta), atol=1e-13)
    assert np.allclose(up.cdf(U, V), point_value_upper(U, V, a, b, theta), atol=1e-13)
    assert lo.cdf(a, b) == pytest.approx(theta, abs=1e-13)
    assert up.cdf(a, b) == pytest.approx(theta, abs=1e-13)
    assert res.lower_is_copula and res.upper_is_copula
    assert two_increasing_defect(lo, m=60).volume >= -TOL
    assert two_increasing_defect(up, m=60).volume >= -TOL
    # closed forms of tau and rho of the extremal shuffles
    p, q, r = a - theta, b - theta, 1 - a - b + theta
    assert up.kendalls_tau() == pytest.approx(1 - 4 * p * q, abs=1e-12)
    assert lo.kendalls_tau() == pytest.approx(4 * theta * r - 1, abs=1e-12)
    assert up.spearmans_rho() == pytest.approx(1 - 6 * p * q * (p + q), abs=1e-12)
    assert lo.spearmans_rho() == pytest.approx(6 * theta * r * (theta + r) - 1, abs=1e-12)


def test_point_value_bounds_contain_random_copulas(random_copulas):
    rng = np.random.default_rng(5)
    for C in random_copulas[::3]:
        a, b = rng.random(2)
        res = bounds_given_value(a, b, C.cdf(a, b))
        assert res.contains(C, m=40, tol=TOL)


def test_point_value_validation():
    with pytest.raises(ValueError, match="theta"):
        bounds_given_value(0.5, 0.5, 0.6)
    with pytest.raises(ValueError):
        bounds_given_value(1.2, 0.5, 0.3)


def test_blomqvist_bounds():
    for beta in (-0.8, -0.2, 0.0, 0.5, 1.0):
        res = blomqvist_beta_bounds(beta)
        lo, up = res
        assert lo.blomqvists_beta() == pytest.approx(beta, abs=1e-12)
        assert up.blomqvists_beta() == pytest.approx(beta, abs=1e-12)
        assert res.extra["beta"] == beta
    C = cp.Frank(3)
    assert blomqvist_beta_bounds(C.blomqvists_beta()).contains(C, m=30)
    with pytest.raises(ValueError):
        blomqvist_beta_bounds(1.5)


# ---------------------------------------------------------------------------
# bounds given Kendall's tau / Spearman's rho (Nelsen et al. 2001)
# ---------------------------------------------------------------------------


def test_tau_bounds_contain_random_copulas(measures):
    for _, z, tau, _ in measures:
        assert np.all(kendall_tau_lower_bound(U, V, tau) <= z + TOL)
        assert np.all(z <= kendall_tau_upper_bound(U, V, tau) + TOL)
        # one-sided versions: tau(C) >= t resp. tau(C) <= t
        t_lo, t_hi = max(tau - 0.3, -1.0), min(tau + 0.3, 1.0)
        assert np.all(kendall_tau_lower_bound(U, V, t_lo) <= z + TOL)
        assert np.all(z <= kendall_tau_upper_bound(U, V, t_hi) + TOL)


def test_rho_bounds_contain_random_copulas(measures):
    for _, z, _, rho in measures:
        assert np.all(spearman_rho_lower_bound(U, V, rho) <= z + TOL)
        assert np.all(z <= spearman_rho_upper_bound(U, V, rho) + TOL)
        t_lo, t_hi = max(rho - 0.3, -1.0), min(rho + 0.3, 1.0)
        assert np.all(spearman_rho_lower_bound(U, V, t_lo) <= z + TOL)
        assert np.all(z <= spearman_rho_upper_bound(U, V, t_hi) + TOL)


@pytest.mark.parametrize(
    "C",
    [cp.Clayton(1.5), cp.Frank(-3), cp.Gaussian(0.7), cp.GumbelHougaard(2.0)],
    ids=lambda C: type(C).__name__,
)
def test_measure_bounds_contain_families(C):
    assert kendall_tau_bounds(C.kendalls_tau()).contains(C, m=30, tol=1e-9)
    assert spearman_rho_bounds(C.spearmans_rho()).contains(C, m=30, tol=1e-9)


ATTAIN_POINTS = [(0.3, 0.5), (0.7, 0.2), (0.5, 0.5), (0.85, 0.9), (0.1, 0.95), (0.6, 0.75)]


def _mixture_measure(A, B, key):
    r"""Exact tau / rho of lam*A + (1-lam)*B (two shuffles of M) as a function of lam.

    rho is linear; tau = 4 int C dC - 1 is quadratic in lam with
    int X dY = int_0^1 X(u, phi_Y(u)) du for a shuffle of M Y.
    """
    if key == "rho":
        ra, rb = A.spearmans_rho(), B.spearmans_rho()
        return lambda lam: lam * ra + (1 - lam) * rb
    x = (np.arange(200_000) + 0.5) / 200_000

    def integral(X, Y):
        return float(np.mean(X.cdf_vectorized(x, Y.support(x))))

    iaa, ibb, cross = integral(A, A), integral(B, B), integral(A, B) + integral(B, A)
    return lambda lam: 4 * (lam**2 * iaa + lam * (1 - lam) * cross + (1 - lam) ** 2 * ibb) - 1


def _attain_by_mixture(a, b, th, t, key):
    """A member of {C(a,b) = th} with measure t (convex combination of C_L, C_U)."""
    A, B = bounds_given_value(a, b, th)
    f = _mixture_measure(A, B, key)
    assert f(1.0) - 1e-9 <= t <= f(0.0) + 1e-9
    l0, l1 = 0.0, 1.0  # the measure is decreasing in lam (C_L <= C_U)
    for _ in range(50):
        mid = 0.5 * (l0 + l1)
        if f(mid) > t:
            l0 = mid
        else:
            l1 = mid
    lam = 0.5 * (l0 + l1)
    assert f(lam) == pytest.approx(t, abs=1e-9)
    assert lam * A.cdf(a, b) + (1 - lam) * B.cdf(a, b) == pytest.approx(th, abs=1e-12)


@pytest.mark.parametrize("t", [-0.7, -0.25, 0.1, 0.45, 0.85])
@pytest.mark.parametrize("key", ["tau", "rho"])
def test_bounds_are_attained_pointwise(key, t):
    """At every (a, b) some copula with measure t attains the bound.

    If the bound is not W(a, b) (resp. M(a, b)), the shuffle
    C_U^{a,b,theta*} (resp. C_L^{a,b,theta*}) itself has measure t;
    otherwise a convex combination of C_L and C_U through (a, b, theta*) does.
    """
    lower = kendall_tau_lower_bound if key == "tau" else spearman_rho_lower_bound
    upper = kendall_tau_upper_bound if key == "tau" else spearman_rho_upper_bound
    meas = "kendalls_tau" if key == "tau" else "spearmans_rho"
    n_shuffle = 0
    for a, b in ATTAIN_POINTS:
        w, m = max(a + b - 1, 0), min(a, b)
        th = lower(a, b, t)
        if th > w + 1e-12:  # attained by C_U^{a,b,theta*}
            C = bounds_given_value(a, b, th).upper
            assert getattr(C, meas)() == pytest.approx(t, abs=1e-10)
            assert C.cdf(a, b) == pytest.approx(th, abs=1e-12)
            n_shuffle += 1
        else:
            _attain_by_mixture(a, b, th, t, key)
        th = upper(a, b, t)
        if th < m - 1e-12:  # attained by C_L^{a,b,theta*}
            C = bounds_given_value(a, b, th).lower
            assert getattr(C, meas)() == pytest.approx(t, abs=1e-10)
            assert C.cdf(a, b) == pytest.approx(th, abs=1e-12)
            n_shuffle += 1
        else:
            _attain_by_mixture(a, b, th, t, key)
    assert n_shuffle >= 1


@pytest.mark.parametrize("t", [-0.85, -0.4, 0.0, 0.35, 0.9])
@pytest.mark.parametrize("key", ["tau", "rho"])
def test_measure_bounds_are_copulas(key, t):
    res = bounds_given_measure(key, t)
    assert res.lower_is_copula and res.upper_is_copula
    for C in res:
        assert isinstance(C, MeasureBoundCopula)
        d = two_increasing_defect(C, m=150)
        assert d.volume >= -TOL
        assert is_copula(C, m=60)


def test_measure_bound_h_functions_integrate_to_cdf():
    from copul.measures.quadrature import integrate_1d

    for key in ("tau", "rho"):
        for side in ("lower", "upper"):
            C = MeasureBoundCopula(key, 0.3 if side == "lower" else -0.3, side)
            for u, v in [(0.3, 0.8), (0.7, 0.4), (0.55, 0.5)]:
                val, _ = integrate_1d(lambda s: C.cond_distr_1(s, np.full_like(s, v)), 0.0, u)
                assert val == pytest.approx(C.cdf(u, v), abs=1e-8)
                val, _ = integrate_1d(lambda s: C.cond_distr_2(np.full_like(s, u), s), 0.0, v)
                assert val == pytest.approx(C.cdf(u, v), abs=1e-8)
            x = C.rvs(2000, random_state=1)
            assert x.shape == (2000, 2)
            assert abs(np.mean(x[:, 1]) - 0.5) < 0.03


def test_closed_form_special_cases():
    W, M = frechet_lower(U, V), frechet_upper(U, V)
    for lo_f, up_f in (
        (kendall_tau_lower_bound, kendall_tau_upper_bound),
        (spearman_rho_lower_bound, spearman_rho_upper_bound),
    ):
        assert np.allclose(lo_f(U, V, -1.0), W, atol=1e-12)
        assert np.allclose(lo_f(U, V, 1.0), M, atol=1e-12)
        assert np.allclose(up_f(U, V, -1.0), W, atol=1e-12)
        assert np.allclose(up_f(U, V, 1.0), M, atol=1e-12)
        # reflection: upper_t(u, v) = u - lower_{-t}(u, 1 - v)
        for t in (-0.5, 0.2, 0.7):
            assert np.allclose(up_f(U, V, t), U - lo_f(U, 1 - V, -t), atol=1e-12)
    # T_t^U = M for t >= 0
    for t in (0.0, 0.3, 0.9):
        assert np.allclose(kendall_tau_upper_bound(U, V, t), M, atol=1e-12)
    # explicit value of T_t^L at the centre
    assert kendall_tau_lower_bound(0.5, 0.5, 0.5) == pytest.approx((1 - np.sqrt(0.5)) / 2)
    with pytest.raises(ValueError):
        kendall_tau_lower_bound(0.5, 0.5, 1.5)


def test_rho_cubic_root():
    rng = np.random.default_rng(2)
    a, b, th = rng.random(500), rng.random(500), 2 * rng.random(500)
    s = spearman_rho_cubic_root(a, b, th)
    d = np.abs(a - b) / 2
    assert np.all(s >= d - 1e-15)
    assert np.allclose(s**3 - d**2 * s - th / 12, 0.0, atol=1e-14)
    # both branches of the closed form: D = 81 theta^2 - 27 (a - b)^6 >= 0 and < 0
    disc = 81 * th**2 - 27 * (a - b) ** 6
    assert np.any(disc >= 0) and np.any(disc < 0)
    for ai, bi, ti in [(0.9, 0.1, 0.01), (0.3, 0.6, 0.5), (0.5, 0.5, 1.0)]:
        roots = np.roots([1.0, 0.0, -((ai - bi) ** 2) / 4, -ti / 12])
        assert spearman_rho_cubic_root(ai, bi, ti) == pytest.approx(
            max(r.real for r in roots if abs(r.imag) < 1e-9), abs=1e-14
        )


# ---------------------------------------------------------------------------
# front end
# ---------------------------------------------------------------------------


def test_bounds_given_measure_front_end():
    res = bounds_given_measure("kendall", 0.4)
    assert isinstance(res, BoundsResult)
    assert res.extra == {"key": "tau", "value": 0.4}
    assert "Nelsen" in res.reference
    lo, up = res
    assert repr(lo) == "T^L_{0.4}" and repr(up) == "T^U_{0.4}"
    assert bounds_given_measure("spearman", -0.2).extra["key"] == "rho"
    beta = bounds_given_measure("beta", 0.2)
    assert beta.lower.cdf(0.5, 0.5) == pytest.approx(0.3)
    C = cp.Frank.from_measure("tau", 0.4)
    assert res.contains(C, m=25, tol=1e-9)
    with pytest.raises(NotImplementedError, match="supported"):
        bounds_given_measure("xi", 0.3)


# ---------------------------------------------------------------------------
# bounds given the diagonal section
# ---------------------------------------------------------------------------


def test_diagonal_bounds_contain_random_copulas(random_copulas):
    for C in random_copulas[::2]:
        res = bounds_given_diagonal(diagonal_section(C), check=False, m=30)
        assert res.contains(C, m=40, tol=TOL)
        S = type(C)((C.matr + C.matr.T) / 2)
        res_s = bounds_given_diagonal(diagonal_section(S), symmetric=True, check=False)
        assert res_s.contains(S, m=40, tol=TOL)


def test_diagonal_upper_bound_is_a_proper_quasi_copula():
    """A_delta for delta(t) = t^2 has V([3/8, 5/8]^2) = 25/64 + 9/64 - 2*3/8 = -7/32."""
    res = bounds_given_diagonal(lambda t: t**2, m=40)
    A = res.upper
    assert isinstance(A, NumericQuasiCopula)
    assert res.lower_is_copula and not res.upper_is_copula
    assert A.is_quasi_copula(m=40)
    assert A.volume((0.375, 0.625, 0.375, 0.625)) == pytest.approx(-7 / 32, abs=1e-12)
    assert two_increasing_defect(A, m=40).volume <= -7 / 32 + 1e-12
    # closed form for delta(t) = t^2
    hat_max = np.where(
        (np.minimum(U, V) <= 0.5) & (np.maximum(U, V) >= 0.5),
        0.25,
        np.maximum(U - U**2, V - V**2),
    )
    expected = np.minimum(np.minimum(U, V), np.maximum(U, V) - hat_max)
    assert np.allclose(A.cdf(U, V), expected, atol=1e-12)
    # all bounds have diagonal delta
    t = np.linspace(0, 1, 51)
    for C in (res.lower, res.upper, diagonal_upper_bound(lambda s: s**2)):
        assert np.allclose(C.cdf(t, t), t**2, atol=1e-13)


def test_symmetric_diagonal_bounds():
    res = bounds_given_diagonal(cp.Clayton(2), symmetric=True)
    assert res.lower_is_copula and res.upper_is_copula
    assert isinstance(res.upper, DiagonalCopula)
    assert res.contains(cp.Clayton(2), m=30, tol=1e-12)
    # the diagonal of W: only W is between the bounds on the diagonal
    res_w = bounds_given_diagonal(lambda t: np.maximum(2 * t - 1, 0), m=30)
    assert res_w.upper_is_copula  # A_{delta_W} is a copula
    with pytest.raises(ValueError, match="not a diagonal"):
        bounds_given_diagonal(lambda t: t**3)
