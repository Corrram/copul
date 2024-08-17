import numpy as np

from copul.families.extreme_value.gumbel_hougaard import GumbelHougaard


def test_gumbel_hougaard():
    cop = GumbelHougaard(2)
    tau = cop.tau()
    assert np.isclose(tau, 0.5)
