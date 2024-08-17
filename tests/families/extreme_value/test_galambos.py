import matplotlib
import pytest

from copul import Galambos

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_galambos_pickands_func():
    cop = Galambos(0.5)
    pickands = cop.pickands(0.5)
    assert pickands == 0.875


def test_galambos_pickands_plot():
    try:
        Galambos().plot_pickands(delta=[0.5, 1, 2])
    except Exception as e:
        pytest.fail(f"plot_pickands() raised an exception: {e}")
