import matplotlib
import pytest

from copul import Nelsen15

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_nelsen15_scatter():
    nelsen = Nelsen15(4)

    # Call plot_cdf and simply ensure it does not raise an exception
    try:
        nelsen.scatter_plot()
    except Exception as e:
        pytest.fail(f"The scatter_plot function raised an exception: {e}")
