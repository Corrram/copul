import numpy as np

from copul.families.other.multivar_checkerboard_copula import MultivarCheckerboardCopula


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
    copula = MultivarCheckerboardCopula(matr)
    assert copula.cdf(0.5, 0.5, 0.5) == 0.125
    assert copula.pdf(0.5, 0.5, 0.5) == 0.125
    # ToDo - add tests for non-boundary values and implement cdf and pdf accordingly


if __name__ == "__main__":
    test_multivar_checkerboard()
