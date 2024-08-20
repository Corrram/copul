import numpy as np
import pytest

from copul.families.archimedean import Nelsen2


def test_nelsen2_generator():
    nelsen2 = Nelsen2(2)
    result = nelsen2.generator(0.5)
    assert np.isclose(result, 0.25)


def test_nelsen2_scatter():
    nelsen2 = Nelsen2(1)
    try:
        nelsen2.scatter_plot()
    except Exception as e:
        pytest.fail(f"plot_cdf() raised an exception: {e}")