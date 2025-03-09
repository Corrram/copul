import numpy as np
from copul import Nelsen12


def test_nelsen12():
    nelsen = Nelsen12(1.5)
    inv_gen = float(nelsen.inv_generator(0.5))
    assert np.isclose(inv_gen, 0.613511790435691, atol=1e-5)
