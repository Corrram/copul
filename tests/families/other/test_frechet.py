import matplotlib
import pytest

from copul import LowerFrechet, UpperFrechet, IndependenceCopula

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


@pytest.mark.parametrize("cop", [LowerFrechet(), UpperFrechet(), IndependenceCopula()])
def test_frechet_scatter(cop):
    # Call plot_cdf and simply ensure it does not raise an exception
    try:
        cop.plot_cdf()
    except Exception as e:
        pytest.fail(f"plot_cdf() raised an exception: {e}")
