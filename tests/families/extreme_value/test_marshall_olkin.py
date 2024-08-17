import numpy as np

from copul.families.extreme_value.marshall_olkin import MarshallOlkin


def test_marshall_olkin():
    cop = MarshallOlkin(1 / 3, 1)
    rho = cop.rho()
    assert np.isclose(rho, 3 / 7)
