import random

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


def test_from_cdf_with_gumbel_barnett_different_var_names():
    np.random.seed(42)
    cdf = "x*y*exp(-0.5*ln(x)*ln(y))"
    copula_family = Copula.from_cdf(cdf)
    copula = copula_family()
    result = copula.cdf(0.5, 0.5)
    assert np.isclose(result, 0.19661242613985133)
    pdf = copula.pdf(0.5, 0.5)
    assert np.isclose(pdf, 1.0328132803599177)
    cd1_func = copula.cond_distr_1()
    cd1 = cd1_func(0.4, 0.3)
    assert np.isclose(cd1, 0.27683793816935376)
    cd2 = copula.cond_distr_2(0.4, 0.3)
    assert np.isclose(cd2, 0.33597451772973175)
    random.seed(1)
    sample_data = copula.rvs(3)
    expected = np.array(
        [[0.89660326, 0.13436424], [0.17197592, 0.76377462], [0.42514613, 0.49543509]]
    )
    assert np.allclose(sample_data, expected)
    copula.scatter_plot()
    copula.plot_cdf()
    copula.plot_pdf()


def test_from_cdf_with_gumbel_barnett_different_var_names_and_theta():
    cdf = "x*y*exp(-theta*ln(x)*ln(y))"
    copula_family = Copula.from_cdf(cdf, "theta")
    copula = copula_family(0.5)
    result = copula.cdf(0.5, 0.5)
    assert np.isclose(result, 0.19661242613985133)
