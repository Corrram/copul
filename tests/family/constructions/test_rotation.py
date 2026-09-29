import numpy as np
import pytest

import copul as cp
from copul.family.constructions import (
    TransformedCopula,
    khoudraji,
    reflect,
    rotate,
    survival,
    transpose,
)
from tests.family.numeric_copula_checks import (
    check_axioms,
    check_conditionals,
    check_density,
    check_sampling,
    closed_vs_numeric,
)

U = np.array([0.1, 0.3, 0.5, 0.72, 0.9])
V = np.array([0.8, 0.2, 0.5, 0.35, 0.95])


def _clayton():
    return cp.Clayton(2)


def _asym():
    # non-exchangeable, absolutely continuous
    return khoudraji(cp.BivIndependenceCopula(), cp.Clayton(4), 0.35, 0.95)


def _ref(C, u, v):
    return np.array([float(C.cdf(a, b)) for a, b in zip(u, v)])


TRANSFORMS = {
    "rot90": lambda C: rotate(C, 90),
    "rot180": lambda C: rotate(C, 180),
    "rot270": lambda C: rotate(C, 270),
    "refl_u": lambda C: reflect(C, "u"),
    "refl_v": lambda C: reflect(C, "v"),
    "transpose": transpose,
    "antidiagonal": lambda C: reflect(C, "antidiagonal"),
}


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_axioms_conditionals_density_sampling(name):
    R = TRANSFORMS[name](_asym())
    check_axioms(R)
    check_conditionals(R)
    check_density(R, total=False)
    check_sampling(R)


def test_explicit_formulas():
    C = _asym()
    c = C.cdf_vectorized
    np.testing.assert_allclose(rotate(C, 90).cdf(U, V), V - c(V, 1 - U), atol=1e-14)
    np.testing.assert_allclose(rotate(C, 180).cdf(U, V), U + V - 1 + c(1 - U, 1 - V), atol=1e-14)
    np.testing.assert_allclose(rotate(C, 270).cdf(U, V), U - c(1 - V, U), atol=1e-14)
    np.testing.assert_allclose(reflect(C, "u").cdf(U, V), V - c(1 - U, V), atol=1e-14)
    np.testing.assert_allclose(reflect(C, "v").cdf(U, V), U - c(U, 1 - V), atol=1e-14)
    np.testing.assert_allclose(transpose(C).cdf(U, V), c(V, U), atol=1e-14)
    np.testing.assert_allclose(survival(C).cdf(U, V), rotate(C, 180).cdf(U, V), atol=1e-14)


def test_symbolic_component():
    C = _clayton()
    R = rotate(C, 180)
    ref = U + V - 1 + _ref(C, 1 - U, 1 - V)
    np.testing.assert_allclose(R.cdf(U, V), ref, atol=1e-12)
    assert isinstance(R.cdf(0.3, 0.4), float)
    assert R.cdf(u=0.3, v=0.4) == R.cdf(0.3, 0.4)


def test_group_structure():
    C = _asym()
    assert rotate(rotate(C, 180), 180) is C
    assert rotate(rotate(C, 90), 270) is C
    assert reflect(reflect(C, "u"), "u") is C
    assert transpose(transpose(C)) is C
    assert rotate(C, 0) is C and rotate(C, 360) is C
    r = rotate(rotate(C, 90), 90)
    assert isinstance(r, TransformedCopula) and r.angle == 180
    # rot90 = reflection in u after transposition
    np.testing.assert_allclose(
        rotate(C, 90).cdf(U, V), reflect(transpose(C), "u").cdf(U, V), atol=1e-14
    )
    assert rotate(C, -90).angle == 270


def test_rotation_equals_reflection_for_exchangeable():
    C = _clayton()
    np.testing.assert_allclose(rotate(C, 90).cdf(U, V), reflect(C, "u").cdf(U, V), atol=1e-12)
    np.testing.assert_allclose(rotate(C, 270).cdf(U, V), reflect(C, "v").cdf(U, V), atol=1e-12)
    # but not for non-exchangeable copulas
    K = _asym()
    assert not np.allclose(rotate(K, 90).cdf(U, V), reflect(K, "u").cdf(U, V))


def test_repr_and_sign():
    C = _clayton()
    assert "90" in repr(rotate(C, 90))
    assert rotate(C, 90).sign == -1 and rotate(C, 180).sign == 1
    assert reflect(C, "u").sign == -1 and transpose(C).sign == 1
    assert "transpose" in repr(transpose(C))


ALL_KEYS = ["rho", "tau", "footrule", "gamma", "beta", "nu", "xi", "xi_2", "lambda_l", "lambda_u"]

EXPECTED_CLOSED = {
    "rot90": {"rho", "tau", "footrule", "gamma", "beta", "xi", "xi_2"},
    "rot180": set(ALL_KEYS),
    "rot270": {"rho", "tau", "footrule", "gamma", "beta", "xi", "xi_2"},
    "refl_u": {"rho", "tau", "footrule", "gamma", "beta", "nu", "xi", "xi_2"},
    "refl_v": {"rho", "tau", "footrule", "gamma", "beta", "nu", "xi", "xi_2"},
    "transpose": set(ALL_KEYS) - {"nu"},
    "antidiagonal": set(ALL_KEYS) - {"nu"},
}


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_measure_relations_against_numerics(name):
    """Every closed relation agrees with the numerical engine (non-exchangeable base)."""
    R = TRANSFORMS[name](_asym())
    closed = closed_vs_numeric(R, ALL_KEYS, tol=1e-7)
    assert set(closed) == EXPECTED_CLOSED[name]


def test_tail_swap_and_xi_swap():
    C = _clayton()
    th = 2.0
    S = rotate(C, 180)
    assert S.lambda_U() == pytest.approx(2 ** (-1 / th), abs=1e-12)
    assert S.lambda_L() == pytest.approx(0.0, abs=1e-12)
    K = _asym()
    xi1, xi2 = K.chatterjees_xi(), K.chatterjees_xi(condition_on_y=True)
    assert abs(xi1 - xi2) > 1e-3
    assert transpose(K).chatterjees_xi() == pytest.approx(xi2, abs=1e-12)
    assert transpose(K).chatterjees_xi(condition_on_y=True) == pytest.approx(xi1, abs=1e-12)


def test_nu_relations_clayton():
    C = _clayton()
    rho, nu = C.spearmans_rho(), C.blests_nu()
    assert reflect(C, "v").blests_nu() == pytest.approx(-nu, abs=1e-12)
    assert reflect(C, "u").blests_nu() == pytest.approx(nu - 2 * rho, abs=1e-12)
    assert survival(C).blests_nu() == pytest.approx(2 * rho - nu, abs=1e-12)
    # footrule relation for concordance-reversing symmetries
    phi, gam = C.spearmans_footrule(), C.ginis_gamma()
    assert rotate(C, 90).spearmans_footrule() == pytest.approx(phi - 1.5 * gam, abs=1e-12)


def test_frechet_bounds_map_to_each_other():
    M = cp.UpperFrechet()
    W = reflect(M, "u")
    np.testing.assert_allclose(W.cdf(U, V), np.maximum(U + V - 1, 0), atol=1e-12)
    assert W.spearmans_rho() == pytest.approx(-1.0, abs=1e-10)
    x = W.rvs(100, random_state=0)
    np.testing.assert_allclose(x[:, 0] + x[:, 1], 1.0, atol=1e-12)


def test_errors():
    with pytest.raises(ValueError):
        rotate(_clayton(), 45)
    with pytest.raises(ValueError):
        reflect(_clayton(), "w")
    with pytest.raises(ValueError):
        rotate(cp.Clayton(), 90)  # free parameter
    assert not rotate(cp.UpperFrechet(), 90).is_absolutely_continuous
