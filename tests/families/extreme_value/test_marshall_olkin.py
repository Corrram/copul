import numpy as np
import pytest

from copul.families.extreme_value.marshall_olkin import MarshallOlkin


@pytest.fixture
def marshall_olkin_copula():
    return MarshallOlkin(1 / 3, 1)


def test_mo_spearmans_rho(marshall_olkin_copula):
    rho = marshall_olkin_copula.spearmans_rho()
    assert np.isclose(rho, 3 / 7)


def test_mo_kendalls_tau(marshall_olkin_copula):
    tau = marshall_olkin_copula.kendalls_tau()
    assert np.isclose(tau, 1 / 3)


def test_mo_chatterjees_xi(marshall_olkin_copula):
    xi = marshall_olkin_copula.chatterjees_xi()
    assert np.isclose(xi, 1 / 6)


def test_mo_chatterjees_xi_with_argument(marshall_olkin_copula):
    xi = MarshallOlkin().chatterjees_xi(1 / 3, 1)
    assert np.isclose(xi, 1 / 6)


def test_mo_kendalls_tau_with_argument(marshall_olkin_copula):
    tau = MarshallOlkin().kendalls_tau(1 / 3, 1)
    assert np.isclose(tau, 1 / 3)


def test_mo_spearmans_rho_with_argument(marshall_olkin_copula):
    rho = MarshallOlkin().spearmans_rho(1 / 3, 1)
    assert np.isclose(rho, 3 / 7)
