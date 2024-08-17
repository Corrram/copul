import numpy as np

from copul import CuadrasAuge


def test_cuadras_auge():
    cop = CuadrasAuge(0.5)
    xi = cop.xi()
    assert np.isclose(xi, 1 / 6)
