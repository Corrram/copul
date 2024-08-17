import matplotlib
import pytest

from copul import Nelsen14

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_nelsen14_generator_plot():
    nelsen = Nelsen14(4)

    # Call plot_cdf and simply ensure it does not raise an exception
    try:
        nelsen.plot_generator()
    except Exception as e:
        pytest.fail(f"scatter_plot() raised an exception: {e}")
