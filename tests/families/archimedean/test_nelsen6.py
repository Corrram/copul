import matplotlib
import pytest

from copul import Nelsen6

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_nelsen6():
    nelsen = Nelsen6(1.5)

    # Call plot_cdf and simply ensure it does not raise an exception
    try:
        nelsen.plot_cdf()
    except Exception as e:
        pytest.fail(f"plot_cdf() raised an exception: {e}")
