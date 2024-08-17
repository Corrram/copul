import numpy as np

from copul.families.archimedean import Nelsen2


def test_nelsen2():
    nelsen2 = Nelsen2(2)
    result = nelsen2.generator(0.5)
    assert np.isclose(result, 0.25)
