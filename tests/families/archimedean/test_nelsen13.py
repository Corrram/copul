import numpy as np

from copul import Nelsen13


def test_nelsen13_lower_orthant_ordered():
    nelsen = Nelsen13(0.5)
    nelsen2 = Nelsen13(1.5)

    def func(u, v):
        return nelsen.cdf(u, v) - nelsen2.cdf(u, v)

    linspace = np.linspace(0.01, 0.99, 10)
    grid2d = np.meshgrid(linspace, linspace)
    values = np.array(
        [func(u, v) for u, v in zip(grid2d[0].flatten(), grid2d[1].flatten())]
    )
    assert np.all(values <= 0)
