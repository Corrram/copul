import matplotlib
import pytest

from copul import Nelsen8

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_nelsen8():
    nelsen = Nelsen8(2)

    # Call plot_cdf and simply ensure it does not raise an exception
    try:
        nelsen.plot(cond_distr_1=nelsen.cond_distr_1)
    except Exception as e:
        pytest.fail(f"plot_cdf() raised an exception: {e}")
