import numpy as np
import pytest

from copul import Nelsen7


@pytest.mark.parametrize("theta, expected", [(0, -1), (1, 0)])
def test_nelsen7_rho(theta, expected):
    nelsen = Nelsen7()(theta)
    rho = nelsen.spearmans_rho()
    assert np.isclose(rho, expected)


@pytest.mark.parametrize("theta, expected", [(0, -1), (1, 0)])
def test_nelsen7_tau(theta, expected):
    nelsen = Nelsen7(theta)
    tau = nelsen.kendalls_tau()
    assert np.isclose(tau, expected)
