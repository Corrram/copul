"""Tests for copul.theory.diagonal (diagonal sections, Bertino and diagonal copulas)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import copul as cp
from copul.theory.diagonal import (
    BertinoCopula,
    Diagonal,
    DiagonalCopula,
    IntervalMin,
    as_diagonal,
    bertino_copula,
    blomqvist_from_diagonal,
    check_diagonal,
    copula_with_diagonal,
    diagonal_copula,
    diagonal_section,
    footrule_from_diagonal,
    gini_from_diagonals,
    is_diagonal,
    opposite_diagonal,
    tail_dependence_from_diagonal,
)
from copul.theory.quasi import is_copula, two_increasing_defect

G = np.linspace(0.0, 1.0, 41)
U, V = np.meshgrid(G, G, indexing="ij")

FAMILIES = [
    cp.Clayton(2),
    cp.GumbelHougaard(2.5),
    cp.Frank(-5),
    cp.Joe(2),
    cp.Gaussian(0.4),
]


def _piecewise(t):
    # a diagonal with an interior minimum of t - delta(t) and a kink
    t = np.asarray(t, dtype=float)
    return np.where(t < 0.5, 0.25 * t, np.minimum(t, 0.125 + 1.75 * (t - 0.5)))


DELTAS = {
    "pi": lambda t: t**2,
    "W": lambda t: np.maximum(2 * t - 1, 0),
    "M": lambda t: t,
    "piecewise": _piecewise,
    "clayton": diagonal_section(cp.Clayton(2)),
    "frank_neg": diagonal_section(cp.Frank(-5)),
}


@pytest.fixture(scope="module")
def checkerboards():
    pis = cp.BivCheckPi.generate_diverse(n_samples=12, grid_size=(2, 9), rng=21)
    return pis + [cp.BivCheckMin(C.matr) for C in pis[:6]]


# ---------------------------------------------------------------------------
# diagonal sections and validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", FAMILIES, ids=lambda C: type(C).__name__)
def test_diagonal_and_opposite_diagonal_sections(C):
    d = diagonal_section(C)
    om = opposite_diagonal(C)
    t = np.linspace(0, 1, 23)
    assert np.allclose(d(t), C.cdf(t, t), atol=1e-12)
    assert np.allclose(om(t), C.cdf(t, 1 - t), atol=1e-12)
    assert isinstance(d(0.3), float) and isinstance(om(0.3), float)
    chk = check_diagonal(d)
    assert chk.is_diagonal and chk.above_lower_ok
    assert is_diagonal(C)  # a copula is replaced by its diagonal section


def test_checkerboard_diagonals_are_diagonals(checkerboards):
    for C in checkerboards:
        assert is_diagonal(diagonal_section(C), m=500)


@pytest.mark.parametrize(
    ("func", "failing"),
    [
        (np.sqrt, "below_identity_ok"),
        (lambda t: t**3, "lipschitz_ok"),
        (lambda t: np.maximum(0.0, 3 * t - 2), "lipschitz_ok"),
        (lambda t: 2 * t - t**2, "below_identity_ok"),
        (lambda t: 0.9 * t**2, "endpoints_ok"),
        (lambda t: np.clip(t**2 - 0.05 * np.sin(12 * np.pi * t), 0, 1), "increasing_ok"),
    ],
)
def test_invalid_diagonals(func, failing):
    chk = check_diagonal(func)
    assert not chk.is_diagonal
    assert getattr(chk, failing) is False
    assert chk.max_violation > 0
    with pytest.raises(ValueError, match="not a diagonal"):
        BertinoCopula(func)
    with pytest.raises(ValueError, match="not a diagonal"):
        DiagonalCopula(func)


def test_diagonal_object():
    d = Diagonal(lambda t: t**2, derivative=lambda t: 2 * t, name="t^2")
    assert repr(d) == "t^2"
    assert d(0.5) == pytest.approx(0.25)
    assert d(np.array([0.2, 0.4])) == pytest.approx([0.04, 0.16])
    assert d.derivative(0.3) == pytest.approx(0.6)
    assert d.hat(0.5) == pytest.approx(0.25)
    assert Diagonal(lambda t: t**2).derivative(0.3) == pytest.approx(0.6, abs=1e-8)
    assert d.is_valid()
    assert d.spearmans_footrule() == pytest.approx(0.0, abs=1e-12)  # footrule(Pi) = 0
    assert d.blomqvists_beta() == pytest.approx(0.0)
    assert d.lambda_L() == pytest.approx(0.0, abs=1e-8)
    assert d.lambda_U() == pytest.approx(0.0, abs=1e-6)
    assert isinstance(d.bertino(), BertinoCopula)
    assert isinstance(d.diagonal_copula(), DiagonalCopula)
    assert as_diagonal(d) is d
    assert isinstance(as_diagonal(cp.Clayton(2)), Diagonal)
    with pytest.raises(TypeError):
        as_diagonal(3.0)
    ax = d.plot()
    assert ax is not None
    plt.close("all")


def test_interval_min_accuracy():
    f = lambda t: np.sin(7 * t) * np.exp(-t) + 0.3 * t
    im = IntervalMin(f)
    rng = np.random.default_rng(0)
    x, y = rng.random(200), rng.random(200)
    got = im(x, y)
    fine = np.linspace(0, 1, 200_001)
    for a, b, g in zip(np.minimum(x, y), np.maximum(x, y), got):
        sel = (fine >= a) & (fine <= b)
        ref = min(f(np.array([a, b])).min(), f(fine[sel]).min() if sel.any() else np.inf)
        assert ref - 1e-9 <= g <= ref + 1e-12
    assert im(0.3, 0.3) == pytest.approx(f(np.array([0.3]))[0])


# ---------------------------------------------------------------------------
# Bertino copula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(DELTAS))
def test_bertino_copula_is_a_copula_with_diagonal_delta(name):
    d = as_diagonal(DELTAS[name])
    B = bertino_copula(d)
    t = np.linspace(0, 1, 101)
    assert np.allclose(B.cdf(t, t), d(t), atol=1e-14)
    assert is_copula(B, m=60)
    assert two_increasing_defect(B, m=60).volume >= -1e-12
    assert B.is_symmetric
    assert np.allclose(B.cdf(U, V), B.cdf(V, U), atol=1e-14)


def test_bertino_copula_special_cases():
    W, M = cp.LowerFrechet(), cp.UpperFrechet()
    assert np.allclose(bertino_copula(DELTAS["W"]).cdf(U, V), W.cdf(U, V), atol=1e-14)
    assert np.allclose(bertino_copula(DELTAS["M"]).cdf(U, V), M.cdf(U, V), atol=1e-14)
    # delta(t) = t^2: the min of t - t^2 over an interval is attained at an endpoint
    B = bertino_copula(DELTAS["pi"])
    expected = np.minimum(U, V) - np.minimum(U - U**2, V - V**2)
    assert np.allclose(B.cdf(U, V), expected, atol=1e-14)


@pytest.mark.parametrize("C", FAMILIES, ids=lambda C: type(C).__name__)
def test_bertino_is_the_smallest_copula_with_given_diagonal(C):
    B = bertino_copula(C)
    assert np.all(B.cdf(U, V) <= C.cdf(U, V) + 1e-12)


def test_bertino_lower_bound_on_checkerboards(checkerboards):
    for C in checkerboards:
        B = bertino_copula(diagonal_section(C), check=False)
        assert np.all(B.cdf_vectorized(U, V) <= C.cdf_vectorized(U, V) + 1e-12)


def test_bertino_h_functions_integrate_to_cdf():
    from copul.measures.quadrature import integrate_1d

    B = bertino_copula(DELTAS["piecewise"])
    for u, v in [(0.3, 0.8), (0.7, 0.2), (0.45, 0.55), (0.9, 0.95)]:
        val, _ = integrate_1d(lambda s: B.cond_distr_1(s, np.full_like(s, v)), 0.0, u)
        assert val == pytest.approx(B.cdf(u, v), abs=1e-7)


def test_bertino_footrule_and_tails_agree_with_copula():
    C = cp.Clayton(2)
    B = bertino_copula(C)
    assert B.spearmans_footrule() == pytest.approx(C.spearmans_footrule(), abs=1e-8)
    assert B.blomqvists_beta() == pytest.approx(C.blomqvists_beta(), abs=1e-12)
    lam_l, lam_u = tail_dependence_from_diagonal(diagonal_section(B))
    assert lam_l == pytest.approx(2 ** (-1 / 2), abs=1e-6)
    assert lam_u == pytest.approx(0.0, abs=1e-6)


def test_bertino_sampling_reproduces_the_diagonal():
    d = as_diagonal(DELTAS["piecewise"])
    x = bertino_copula(d).rvs(20_000, random_state=3)
    from scipy.stats import kstest

    assert kstest(x[:, 0], "uniform").pvalue > 0.01
    assert kstest(x[:, 1], "uniform").pvalue > 0.01
    for t in (0.25, 0.5, 0.75, 0.9):
        emp = np.mean((x[:, 0] <= t) & (x[:, 1] <= t))
        assert emp == pytest.approx(d(t), abs=0.015)


# ---------------------------------------------------------------------------
# Fredricks-Nelsen diagonal copula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(DELTAS))
def test_diagonal_copula_is_a_symmetric_copula_with_diagonal_delta(name):
    d = as_diagonal(DELTAS[name])
    K = diagonal_copula(d)
    t = np.linspace(0, 1, 101)
    assert np.allclose(K.cdf(t, t), d(t), atol=1e-14)
    assert is_copula(K, m=60)
    assert K.is_symmetric
    assert np.allclose(K.cdf(U, V), K.cdf(V, U), atol=1e-14)
    # B_delta <= K_delta
    assert np.all(bertino_copula(d).cdf(U, V) <= K.cdf(U, V) + 1e-12)


@pytest.mark.parametrize(
    "C",
    [
        cp.Clayton(3),
        cp.Frank(4),
        cp.Gaussian(-0.5),
        cp.Plackett(0.3),
        cp.FarlieGumbelMorgenstern(0.8),
    ],
    ids=lambda C: type(C).__name__,
)
def test_diagonal_copula_is_the_largest_symmetric_copula(C):
    K = diagonal_copula(C)
    assert np.all(C.cdf(U, V) <= K.cdf(U, V) + 1e-10)


def test_diagonal_copula_bound_on_symmetric_checkerboards(checkerboards):
    for C in checkerboards:
        S = type(C)((C.matr + C.matr.T) / 2)
        K = diagonal_copula(diagonal_section(S), check=False)
        assert np.all(S.cdf_vectorized(U, V) <= K.cdf_vectorized(U, V) + 1e-12)


def test_diagonal_copula_h_functions():
    from copul.measures.quadrature import integrate_1d

    K = diagonal_copula(Diagonal(lambda t: t**2, derivative=lambda t: 2 * t))
    for u, v in [(0.3, 0.8), (0.7, 0.2), (0.5, 0.6)]:
        val, _ = integrate_1d(lambda s: K.cond_distr_1(s, np.full_like(s, v)), 0.0, u)
        assert val == pytest.approx(K.cdf(u, v), abs=1e-8)
        val, _ = integrate_1d(lambda s: K.cond_distr_2(np.full_like(s, u), s), 0.0, v)
        assert val == pytest.approx(K.cdf(u, v), abs=1e-8)
    assert K.spearmans_footrule() == pytest.approx(0.0, abs=1e-8)


def test_copula_with_diagonal_dispatch():
    d = DELTAS["pi"]
    assert isinstance(copula_with_diagonal(d), BertinoCopula)
    assert isinstance(copula_with_diagonal(d, kind="diagonal"), DiagonalCopula)
    with pytest.raises(ValueError):
        copula_with_diagonal(d, kind="other")


# ---------------------------------------------------------------------------
# dependence quantities from diagonal sections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", FAMILIES, ids=lambda C: type(C).__name__)
def test_footrule_gini_beta_from_diagonals(C):
    d = diagonal_section(C)
    assert footrule_from_diagonal(d) == pytest.approx(C.spearmans_footrule(), abs=1e-7)
    assert gini_from_diagonals(C) == pytest.approx(C.ginis_gamma(), abs=1e-7)
    assert gini_from_diagonals(d, opposite_diagonal(C)) == pytest.approx(C.ginis_gamma(), abs=1e-7)
    assert blomqvist_from_diagonal(d) == pytest.approx(C.blomqvists_beta(), abs=1e-10)


def test_footrule_from_diagonal_on_checkerboards(checkerboards):
    for C in checkerboards[:6]:
        assert footrule_from_diagonal(diagonal_section(C)) == pytest.approx(
            C.spearmans_footrule(), abs=1e-8
        )
        assert gini_from_diagonals(C) == pytest.approx(C.ginis_gamma(), abs=1e-8)


def test_gini_requires_omega_for_plain_diagonals():
    with pytest.raises(ValueError):
        gini_from_diagonals(lambda t: t**2)


@pytest.mark.parametrize(
    ("C", "lam_l", "lam_u"),
    [
        (cp.Clayton(2), 2 ** (-1 / 2), 0.0),
        (cp.GumbelHougaard(2.5), 0.0, 2 - 2 ** (1 / 2.5)),
        (cp.Joe(2), 0.0, 2 - 2 ** (1 / 2)),
        (cp.Frank(3), 0.0, 0.0),
        (cp.UpperFrechet(), 1.0, 1.0),
    ],
    ids=lambda x: type(x).__name__ if hasattr(x, "cdf") else None,
)
def test_tail_dependence_from_diagonal(C, lam_l, lam_u):
    """lambda_L = delta'(0+), lambda_U = 2 - delta'(1-)."""
    got_l, got_u = tail_dependence_from_diagonal(C)
    assert got_l == pytest.approx(lam_l, abs=1e-6)
    assert got_u == pytest.approx(lam_u, abs=1e-6)
