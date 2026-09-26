"""Tests for copul.search (random checkerboards and counterexample search)."""

import numpy as np
import pytest

from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_w import BivCheckW
from copul.optim.shapes import satisfies_shape
from copul.regions.catalog import rho_max_given_xi
from copul.search import (
    Counterexample,
    check_inequality,
    find_counterexample,
    random_checkerboards,
    random_mass_matrix,
    si_rearrangement,
)


def test_random_checkerboards_deterministic_and_typed():
    a = [c.matr for c in random_checkerboards(5, grid=(2, 9), rng=7)]
    b = [c.matr for c in random_checkerboards(5, grid=(2, 9), rng=7)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for kind, cls in (("pi", BivCheckPi), ("min", BivCheckMin), ("w", BivCheckW)):
        cops = list(random_checkerboards(4, grid=5, kind=kind, rng=1))
        assert len(cops) == 4
        assert all(type(c) is cls and c.matr.shape == (5, 5) for c in cops)
        for c in cops:
            np.testing.assert_allclose(c.matr.sum(axis=0), 0.2, atol=1e-10)
            np.testing.assert_allclose(c.matr.sum(axis=1), 0.2, atol=1e-10)


def test_infinite_iterator_and_conditions():
    it = random_checkerboards(grid=(2, 6), rng=0)
    first = [next(it) for _ in range(3)]
    assert len(first) == 3
    for c in random_checkerboards(20, grid=(2, 8), condition="si", rng=2):
        assert satisfies_shape(c.matr, "si", tol=1e-12)
    for c in random_checkerboards(10, grid=(2, 8), condition="sd", rng=2):
        assert satisfies_shape(c.matr, "sd", tol=1e-12)
    for c in random_checkerboards(10, grid=(2, 8), condition="exchangeable", rng=3):
        np.testing.assert_allclose(c.matr, c.matr.T)
    for c in random_checkerboards(10, grid=(2, 8), condition="radially_symmetric", rng=3):
        np.testing.assert_allclose(c.matr, c.matr[::-1, ::-1])
    pos = list(random_checkerboards(5, grid=4, condition=lambda C: C.spearmans_rho() > 0, rng=4))
    assert all(c.spearmans_rho() > 0 for c in pos)
    with pytest.raises(ValueError):
        next(random_checkerboards(1, condition="tp2"))
    with pytest.raises(RuntimeError):
        next(random_checkerboards(1, condition=lambda C: False, max_tries=3))


def test_mass_matrix_helpers():
    P = random_mass_matrix(5, 3, rng=0)
    assert P.shape == (3, 5)
    np.testing.assert_allclose(P.sum(axis=1), 1 / 3, atol=1e-12)
    np.testing.assert_allclose(P.sum(axis=0), 1 / 5, atol=1e-12)
    S = si_rearrangement(P)
    np.testing.assert_allclose(S.sum(axis=0), 1 / 5, atol=1e-12)
    assert satisfies_shape(S, "si", tol=1e-12)


def test_true_inequalities_hold():
    rep = check_inequality("footrule", -0.5, ">=", n_iter=300, rng=0)
    assert rep.holds and rep.counterexample is None
    assert rep.n_samples == 300 and rep.n_violations == 0
    assert rep.max_violation <= 1e-10
    rep = check_inequality("rho", 1.0, "<=", n_iter=200, kind="min", rng=1)
    assert rep.holds
    # Ansari--Rockel: |rho| <= M(xi), with a callable left-hand side
    rep = check_inequality(
        lambda C: abs(C.spearmans_rho()) - float(rho_max_given_xi(C.chatterjees_xi())),
        0.0,
        "<=",
        n_iter=200,
        grid=(2, 12),
        rng=2,
    )
    assert rep.holds, rep.max_violation
    # SI implies xi <= footrule (<= sqrt(xi))
    assert check_inequality("xi", "footrule", "<=", n_iter=200, condition="si", rng=3).holds
    assert find_counterexample("footrule", "xi", ">=", n_iter=100, condition="si", rng=4) is None


def test_false_inequalities_are_refuted_quickly():
    ce = find_counterexample("xi", "rho", "<=", n_iter=100, rng=0)
    assert isinstance(ce, Counterexample)
    assert ce.lhs > ce.rhs and ce.margin == pytest.approx(ce.lhs - ce.rhs)
    assert ce.copula.chatterjees_xi() == pytest.approx(ce.lhs, abs=1e-10)
    assert ce.matrix is not None
    ce2 = find_counterexample("tau", "rho", "<=", n_iter=300, rng=1)
    assert ce2 is not None and ce2.margin > 0
    assert ce2.copula.kendalls_tau() - ce2.copula.spearmans_rho() == pytest.approx(
        ce2.margin, abs=1e-10
    )
    ce3 = find_counterexample("rho", 0.0, "==", n_iter=10, rng=2)
    assert ce3 is not None and ce3.margin == pytest.approx(abs(ce3.lhs))


def test_refinement_improves_or_keeps_maximum():
    raw = check_inequality("tau", "rho", "<=", n_iter=50, refine=False, rng=5)
    ref = check_inequality("tau", "rho", "<=", n_iter=50, refine=True, rng=5)
    assert ref.max_violation >= raw.max_violation - 1e-15
    assert ref.argmax.kendalls_tau() - ref.argmax.spearmans_rho() == pytest.approx(
        ref.max_violation, abs=1e-10
    )


def test_custom_samplers_and_errors():
    from copul.family.frechet.frechet import Frechet

    fams = [Frechet(alpha=a, beta=0.0) for a in (0.1, 0.5, 0.9)]
    rep = check_inequality(
        lambda C: float(C.spearmans_rho()),
        lambda C: float(C.alpha),
        "==",
        sampler=fams,
        n_iter=3,
    )
    assert rep.holds and rep.n_samples == 3
    rng = np.random.default_rng(0)
    rep2 = check_inequality(
        "rho",
        -1.0,
        ">=",
        sampler=lambda: BivCheckPi(random_mass_matrix(4, rng=rng)),
        n_iter=10,
    )
    assert rep2.holds and rep2.n_samples == 10
    with pytest.raises(ValueError):
        check_inequality("rho", "xi", "!=")
