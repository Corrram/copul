import numpy as np

from copul.families.copula import Copula


def test_from_cdf_with_plackett():
    plackett_cdf = (
        "((theta - 1)*(u + v) - sqrt(-4*theta*u*v*(theta - 1) + "
        "((theta - 1)*(u + v) + 1)**2) + 1)/(2*(theta - 1))"
    )
    copula_family = Copula.from_cdf(plackett_cdf)
    copula = copula_family(0.1)
    result = copula.cdf(0.5, 0.5)
    assert np.isclose(result, 0.12012653667602105)


def test_from_cdf_with_gumbel_barnett():
    cdf = "u*v*exp(-theta*ln(u)*ln(v))"
    copula_family = Copula.from_cdf(cdf)
    copula = copula_family(0.1)
    result = copula.cdf(0.5, 0.5)
    assert np.isclose(result, 0.2382726524420907)
