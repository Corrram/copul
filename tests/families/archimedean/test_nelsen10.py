import matplotlib
import pytest

from copul import Nelsen10

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_nelsen10():
    nelsen = Nelsen10(0.5)

    # Call plot_cdf and simply ensure it does not raise an exception
    try:
        nelsen.plot_pdf()
    except Exception as e:
        pytest.fail(f"plot_cdf() raised an exception: {e}")
