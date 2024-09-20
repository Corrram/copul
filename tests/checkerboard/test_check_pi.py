import numpy as np

from copul.checkerboard.check_pi import CheckPi


def test_multivar_checkerboard():
    # 3-dim matrix
    matr = np.ndarray((2, 2, 2))
    matr[0, 0, 0] = 0.5
    matr[0, 0, 1] = 0.5
    matr[0, 1, 0] = 0.5
    matr[0, 1, 1] = 0.5
    matr[1, 0, 0] = 0.5
    matr[1, 0, 1] = 0.5
    matr[1, 1, 0] = 0.5
    matr[1, 1, 1] = 0.5
    copula = CheckPi(matr)
    u = (0.5, 0.5, 0.5)
    assert copula.cdf(*u) == 0.125
    assert copula.pdf(*u) == 0.125
    assert copula.cond_distr(1, u) == 0.25
    assert copula.cond_distr(2, u) == 0.25
    assert copula.cond_distr(3, u) == 0.25
    # ToDo - add tests for non-boundary values and implement cdf and pdf accordingly


def test_2d_check_pi_rvs():
    np.random.seed(1)
    ccop = CheckPi([[1, 2], [2, 1]])
    n = 1_000
    samples = ccop.rvs(n)
    n_lower_empirical = sum([(sample < (0.5, 0.5)).all() for sample in samples])
    n_upper_empirical = sum([(sample > (0.5, 0.5)).all() for sample in samples])
    theoretical_ratio = 1 / 6 * n
    assert n_lower_empirical < 1.5 * theoretical_ratio
    assert n_upper_empirical < 1.5 * theoretical_ratio


def test_3d_check_pi_rvs():
    np.random.seed(1)
    ccop = CheckPi([[[1, 2], [2, 1]], [[1, 2], [2, 1]]])
    n = 1_000
    samples = ccop.rvs(n)
    n_lower_empirical = sum([(sample < (0.5, 0.5, 0.5)).all() for sample in samples])
    n_upper_empirical = sum([(sample > (0.5, 0.5, 0.5)).all() for sample in samples])
    theoretical_ratio = 1 / 12 * n
    assert 0.5 * theoretical_ratio < n_lower_empirical < 1.5 * theoretical_ratio
    assert 0.5 * theoretical_ratio < n_upper_empirical < 1.5 * theoretical_ratio
