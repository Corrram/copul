import numpy as np

from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula


def test_from_generator_with_galambos():
    pickands = "1 - (t ** (-delta) + (1 - t) ** (-delta)) ** (-1 / delta)"
    copula_family = ExtremeValueCopula.from_pickands(pickands)
    copula = copula_family(2)
    result = copula.pickands(0.5)
    assert np.isclose(result, 0.6464466094067263)


def test_from_generator_with_galambos_with_different_var_name():
    pickands = "1 - (x ** (-delta) + (1 - x) ** (-delta)) ** (-1 / delta)"
    copula_family = ExtremeValueCopula.from_pickands(pickands, "delta")
    copula = copula_family(2)
    result = copula.pickands(0.5)
    assert np.isclose(result, 0.6464466094067263)
