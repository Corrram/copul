import matplotlib
import pytest

from copul import Plackett

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_plackett():
    cop = Plackett(0.1)
    try:
        cop.plot(cop.cond_distr_1, cop.cond_distr_2)
    except Exception as e:
        pytest.fail(f"plot_cdf() raised an exception: {e}")
