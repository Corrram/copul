import numpy as np

import copul


def test_squared_checkerboard():
    clayton = copul.Families.CLAYTON.value(2)
    checkerboarder = copul.Checkerboarder(3, 3)
    ccop = checkerboarder.compute_check_copula(clayton)
    assert ccop.matr.shape == (3, 3)
    assert ccop.matr.sum() == 1.0


def test_rectangular_checkerboard():
    clayton = copul.Families.CLAYTON.value(2)
    checkerboarder = copul.Checkerboarder(3, 10)
    ccop = checkerboarder.compute_check_copula(clayton)
    assert ccop.matr.shape == (3, 10)
    assert np.isclose(ccop.matr.sum(), 1.0)


def test_xi_computation():
    np.random.seed(121)
    copula = copul.Families.NELSEN7.value(0.5)
    checkerboarder = copul.Checkerboarder(10)
    ccop = checkerboarder.compute_check_copula(copula)
    orig_xi = copula.chatterjees_xi()
    xi = ccop.chatterjees_xi()
    assert xi <= orig_xi
