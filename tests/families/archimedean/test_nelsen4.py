import numpy as np

from copul import Nelsen4


def test_nelsen4():
    nelsen4 = Nelsen4(0.5)
    result = nelsen4.lambda_L()
    assert np.isclose(result, 0)
