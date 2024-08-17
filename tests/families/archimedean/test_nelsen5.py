import numpy as np

from copul import Nelsen5


def test_nelsen5():
    nelsen = Nelsen5(0.5)
    result = nelsen.lambda_U()
    assert np.isclose(result, 0)
