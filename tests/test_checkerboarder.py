import numpy as np

import copul


def test_squared_checkerboard():
    clayton = copul.Families.CLAYTON.value(2)
    checkerboarder = copul.Checkerboarder(3)
    ccop = checkerboarder.compute_check_pi(clayton)
    assert ccop.matr.shape == (3, 3)
    assert ccop.matr.sum() == 1.0


def test_rectangular_checkerboard():
    clayton = copul.Families.CLAYTON.value(2)
    checkerboarder = copul.Checkerboarder([3, 10])
    ccop = checkerboarder.compute_check_pi(clayton)
    assert ccop.matr.shape == (3, 10)
    matr_sum = ccop.matr.sum()
    assert np.isclose(matr_sum, 1.0)


def test_xi_computation():
    np.random.seed(121)
    copula = copul.Families.NELSEN7.value(0.5)
    checkerboarder = copul.Checkerboarder(10)
    ccop = checkerboarder.compute_check_pi(copula)
    orig_xi = copula.chatterjees_xi()
    xi = ccop.chatterjees_xi(1_000)
    assert 0.5 * orig_xi <= xi <= orig_xi
