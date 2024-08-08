import numpy as np

from copul import CheckerboardCopula


def test_checkerboard():
    matr = np.array([[0.5, 0.5], [0.5, 0.5]])
    copula = CheckerboardCopula(matr)
    assert copula.cdf(0.5, 0.5) == 0.25


if __name__ == "__main__":
    test_checkerboard()
