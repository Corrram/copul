import numpy as np

from copul.families.archimedean import Nelsen3


def test_nelsen3():
    nelsen3 = Nelsen3(0.5)
    result = nelsen3.xi()
    assert np.isclose(result, 0.0225887222397811)
