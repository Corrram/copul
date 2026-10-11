"""``validate_copula`` for expressions that are undefined on the boundary and
for families without a symbolic cdf expression."""

import pytest

import copul as cp


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0])
def test_gumbel_barnett_is_valid(theta):
    # u v exp(-theta log u log v) is nan at (0, 1) and (1, 0) as an expression
    ok, details = cp.GumbelBarnett(theta).validate_copula(return_details=True)
    assert ok
    assert details["n_boundary_limits"] > 0


def test_custom_gumbel_barnett_from_cdf():
    family = cp.from_cdf("u*v*exp(-theta*log(u)*log(v))")
    assert family(theta=0.5).validate_copula()
    assert not family(theta=1.5).validate_copula()  # not 2-increasing


def test_invalid_expression_stays_invalid():
    family = cp.from_cdf("u*v + theta*(u**2 - u)*(v**2 - v)")
    assert family(theta=0.5).validate_copula()
    assert not family(theta=2).validate_copula()


@pytest.mark.parametrize(
    "copula",
    [
        cp.Clayton(2),
        cp.Frank(3),
        cp.GumbelHougaard(2),
        cp.GumbelHougaardEV(2.5),
        cp.Galambos(1.5),
        cp.XiNuBoundaryCopula(b=0.5),
    ],
    ids=lambda c: type(c).__name__,
)
def test_families_without_symbolic_cdf_expression(copula):
    assert copula.validate_copula(m=12)
