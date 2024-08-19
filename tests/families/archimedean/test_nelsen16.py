import matplotlib
import pytest

from copul import Nelsen16

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_nelsen16_scatter():
    nelsen = Nelsen16()

    # Call plot_cdf and simply ensure it does not raise an exception
    try:
        nelsen.plot_rank_correlations(10, 10)
    except Exception as e:
        pytest.fail(f"The scatter_plot function raised an exception: {e}")
