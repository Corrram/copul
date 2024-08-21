import pytest

import copul


def test_rvs_for_clayton():
    clayton = copul.archimedean.Clayton(2)
    samples = clayton.rvs(1000)
    assert samples.shape == (1000, 2)


def test_nelsen2_scatter():
    copula = copul.Clayton(0.5)
    try:
        copula.scatter_plot(10)
    except Exception as e:
        pytest.fail(f"plot_cdf() raised an exception: {e}")
