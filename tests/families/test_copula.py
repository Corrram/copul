import numpy as np

import copul
from copul.families.copula import Copula


def test_cond_distr_of_ind_copula():
    copula: Copula = copul.from_cdf("x*y*z")
    cond_distr = copula.cond_distr(2)
    assert cond_distr(0.5, 0.5) == 0.25


def test_cond_distr_direct_eval_of_ind_copula():
    copula: Copula = copul.from_cdf("x*y*z")
    u = [0.5, 0.5]
    cond_distr = copula.cond_distr(2, u)
    assert cond_distr == 0.25


def test_copula_pdf():
    copula: Copula = copul.from_cdf("x*y*z")
    pdf = copula.pdf()
    assert pdf.func == 1
    evaluated_pdf = pdf(0.5, 0.5, 0.5)
    assert np.isclose(evaluated_pdf.evalf(), 1)
