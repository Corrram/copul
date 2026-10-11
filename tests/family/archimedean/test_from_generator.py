"""``cp.from_generator``: Archimedean copulas from a user-supplied generator."""

import numpy as np
import pytest
import sympy

import copul as cp

PTS = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.2], [0.05, 0.95]])


def test_clayton_generator_matches_family():
    family = cp.from_generator("(t**(-theta) - 1)/theta")
    assert family.params  # free parameter theta
    C = family(theta=2)
    assert C.is_fully_specified()
    np.testing.assert_allclose(C.cdf(PTS), cp.Clayton(2).cdf(PTS), atol=1e-12)
    np.testing.assert_allclose(family(2).cdf(PTS), cp.Clayton(2).cdf(PTS), atol=1e-12)
    assert C.kendalls_tau() == pytest.approx(0.5, abs=1e-10)
    assert C.spearmans_rho() == pytest.approx(cp.Clayton(2).spearmans_rho(), abs=1e-8)
    assert C.is_absolutely_continuous


def test_gumbel_generator_matches_family():
    C = cp.from_generator("(-log(t))**theta")(theta=2)
    np.testing.assert_allclose(C.cdf(PTS), cp.GumbelHougaard(2).cdf(PTS), atol=1e-12)
    assert C.kendalls_tau() == pytest.approx(0.5, abs=1e-8)


def test_parameter_free_generator_is_independence():
    C = cp.from_generator("-log(t)")
    assert C.params == []
    np.testing.assert_allclose(C.cdf(PTS), PTS[:, 0] * PTS[:, 1], atol=1e-12)


def test_non_strict_generator():
    # Nelsen (4.2.7): phi(t) = -log(theta t + 1 - theta), phi(0) = log(1/(1 - theta))
    family = cp.from_generator("-log(theta*t + 1 - theta)", theta_interval=sympy.Interval(0, 1))
    C = family(theta=0.5)
    assert not C.is_absolutely_continuous
    np.testing.assert_allclose(C.cdf(PTS), cp.Nelsen7(0.5).cdf(PTS), atol=1e-12)


def test_other_variable_and_parameter_names():
    family = cp.from_generator("(x**(-alpha) - 1)/alpha", params=["alpha"])
    np.testing.assert_allclose(family(theta=1).cdf(PTS), cp.Clayton(1).cdf(PTS), atol=1e-12)


def test_classmethod_delegates():
    C = cp.Clayton.from_generator("(t**(-theta) - 1)/theta")(theta=2)
    np.testing.assert_allclose(C.cdf(PTS), cp.Clayton(2).cdf(PTS), atol=1e-12)


@pytest.mark.parametrize("bad", ["(t**(-a) - 1)/b", "u*v"])
def test_invalid_generators_raise(bad):
    with pytest.raises(ValueError):
        cp.from_generator(bad)
