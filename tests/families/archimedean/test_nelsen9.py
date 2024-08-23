import matplotlib
import numpy as np
import pytest

from copul import Nelsen9

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_nelsen9_plot_cdf():
    nelsen = Nelsen9(2)

    # Call plot_cdf and simply ensure it does not raise an exception
    try:
        nelsen.plot(cond_distr_2=nelsen.cond_distr_2)
    except Exception as e:
        pytest.fail(f"plot_cdf() raised an exception: {e}")


def test_gumbel_barnett_cond_distr_1():
    nelsen = Nelsen9(0.5)
    result = nelsen.cond_distr_1(0.3, 0.4)
    assert np.isclose(result, 0.27683793816935376)
    result = nelsen.cdf(0.5, 0.5)
    assert np.isclose(result, 0.19661242613985133)
