import numpy as np

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula


def test_from_generator_with_nelsen2_and_specific_theta():  # theta = 2
    copula = ArchimedeanCopula.from_generator("(1 - t) ** 2")
    assert copula.generator(0.5) == 0.25
    cdf = copula.cdf(0.5, 0.5)
    assert np.isclose(cdf, 0.2928932188134524)


def test_from_generator_with_nelsen2():
    copula = ArchimedeanCopula.from_generator("(1 - t) ** theta")
    assert copula(2).generator(0.5) == 0.25
    cdf = copula(2).cdf(0.5, 0.5)
    assert np.isclose(cdf, 0.2928932188134524)
