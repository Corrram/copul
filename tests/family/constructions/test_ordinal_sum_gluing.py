import numpy as np
import pytest

import copul as cp
from copul.family.constructions import (
    GluingCopula,
    OrdinalSumCopula,
    gluing,
    markov_product,
    ordinal_sum,
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
KEYS = ["rho", "tau", "footrule", "beta", "lambda_l", "lambda_u"]


def _cdf(C, u, v):
    return np.array([float(C.cdf(a, b)) for a, b in zip(u, v)])


# ------------------------------------------------------------------ ordinal sums


def test_ordinal_sum_definition():
    A = cp.Clayton(2)
    C = ordinal_sum([(0.2, 0.6, A)])
    assert isinstance(C, OrdinalSumCopula)
    u, v = np.array([0.3, 0.5, 0.1, 0.7]), np.array([0.4, 0.25, 0.9, 0.65])
    ell = 0.4
    inside = 0.2 + ell * _cdf(A, (u[:2] - 0.2) / ell, (v[:2] - 0.2) / ell)
    np.testing.assert_allclose(C.cdf(u[:2], v[:2]), inside, atol=1e-12)
    np.testing.assert_allclose(C.cdf(u[2:], v[2:]), np.minimum(u[2:], v[2:]), atol=1e-12)


def test_ordinal_sum_copula_properties():
    C = ordinal_sum([(0.1, 0.4, cp.Clayton(2)), (0.5, 1.0, cp.Frank(4))])
    assert not C.is_absolutely_continuous
    check_axioms(C)
    check_sampling(C)


def test_ordinal_sum_full_cover_is_absolutely_continuous():
    C = ordinal_sum([(0.0, 0.3, cp.Frank(-3)), (0.3, 1.0, cp.Clayton(1))])
    assert C.is_absolutely_continuous
    check_axioms(C)
    check_conditionals(C)
    check_density(C, total=False)


def test_ordinal_sum_single_component_is_identity():
    A = cp.Frank(3)
    np.testing.assert_allclose(ordinal_sum([(0, 1, A)]).cdf(U, V), _cdf(A, U, V), atol=1e-12)


def test_ordinal_sum_independence_pieces():
    Pi = cp.BivIndependenceCopula()
    C = ordinal_sum([(0.0, 0.5, Pi), (0.5, 1.0, Pi)])
    assert C.spearmans_rho() == pytest.approx(0.75, abs=1e-12)
    assert C.kendalls_tau() == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    "parts",
    [
        [(0.1, 0.4, "clayton"), (0.5, 1.0, "frank")],
        [(0.0, 0.6, "gumbel"), (0.6, 1.0, "clayton")],
    ],
)
def test_ordinal_sum_relations_against_numerics(parts):
    fam = {"clayton": cp.Clayton(2), "frank": cp.Frank(4), "gumbel": cp.GumbelHougaard(3)}
    C = ordinal_sum([(a, b, fam[k]) for a, b, k in parts])
    closed = closed_vs_numeric(C, KEYS, tol=5e-8)
    assert set(closed) == set(KEYS)


def test_ordinal_sum_errors():
    with pytest.raises(ValueError):
        ordinal_sum([(0.0, 0.5, cp.Clayton(2)), (0.4, 1.0, cp.Clayton(2))])
    with pytest.raises(ValueError):
        ordinal_sum([(0.5, 0.5, cp.Clayton(2))])
    with pytest.raises(ValueError):
        ordinal_sum([])


# ------------------------------------------------------------------ gluing


def test_gluing_definition_and_properties():
    A, B = cp.Clayton(2), cp.Frank(-4)
    G = gluing(A, B, 0.45)  # (0.45 is not a grid point of the h-function checks)
    assert isinstance(G, GluingCopula)
    lo = U <= 0.45
    ref = np.where(
        lo,
        0.45 * _cdf(A, np.minimum(U / 0.45, 1), V),
        0.45 * V + 0.55 * _cdf(B, np.maximum((U - 0.45) / 0.55, 0), V),
    )
    np.testing.assert_allclose(G.cdf(U, V), ref, atol=1e-12)
    check_axioms(G)
    check_conditionals(G)
    check_density(G, total=False)
    check_sampling(G)


def test_gluing_independence_is_independence():
    Pi = cp.BivIndependenceCopula()
    np.testing.assert_allclose(gluing(Pi, Pi, 0.3).cdf(U, V), U * V, atol=1e-12)


def test_gluing_relations_against_numerics():
    G = gluing(cp.Clayton(2), cp.Frank(4), 0.4)
    closed = closed_vs_numeric(G, ["rho", "tau", "beta"])
    assert set(closed) == {"rho", "tau", "beta"}


def test_gluing_second_axis():
    A, B = cp.Clayton(2), cp.Frank(4)
    G = gluing(A, B, 0.3, axis="v")
    lo = V <= 0.3
    ref = np.where(
        lo,
        0.3 * _cdf(A, U, np.minimum(V / 0.3, 1)),
        0.3 * U + 0.7 * _cdf(B, U, np.maximum((V - 0.3) / 0.7, 0)),
    )
    np.testing.assert_allclose(G.cdf(U, V), ref, atol=1e-12)
    assert G.spearmans_rho() == pytest.approx(
        0.09 * A.spearmans_rho() + 0.49 * B.spearmans_rho(), abs=1e-12
    )
    assert isinstance(transpose(G), GluingCopula)


def test_gluing_errors():
    with pytest.raises(ValueError):
        gluing(cp.Clayton(2), cp.Frank(1), 1.0)
    with pytest.raises(ValueError):
        gluing(cp.Clayton(2), cp.Frank(1), 0.5, axis="w")


def test_markov_product_reexport():
    P = cp.BivCheckPi([[1, 0], [0, 1]])
    assert markov_product(P, P) is not None
