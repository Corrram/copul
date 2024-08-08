import copul


def test_squared_checkerboard():
    clayton = copul.Families.CLAYTON.value(theta=2)
    checkerboarder = copul.Checkerboarder(3, 3)
    ccop = checkerboarder.compute_check_copula(clayton)
    assert ccop.matr.shape == (3, 3)
    assert ccop.matr.sum() == 1.0


def test_rectangular_checkerboard():
    clayton = copul.Families.CLAYTON.value(theta=2)
    checkerboarder = copul.Checkerboarder(3, 10)
    ccop = checkerboarder.compute_check_copula(clayton)
    assert ccop.matr.shape == (3, 10)
    assert ccop.matr.sum() == 1.0


def test_xi_computation():
    copula = copul.Families.NELSEN7.value(theta=0.5)
    checkerboarder = copul.Checkerboarder(20, 20)
    ccop = checkerboarder.compute_check_copula(copula)
    orig_xi = copula.xi()
    xi = ccop.xi()
    assert xi <= orig_xi


if __name__ == "__main__":
    test_xi_computation()
    test_squared_checkerboard()
    test_rectangular_checkerboard()
