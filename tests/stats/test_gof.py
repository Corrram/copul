import numpy as np
import pytest

import copul as cp
from copul import stats as cs
from copul.stats._adapters import sample


@pytest.fixture(scope="module")
def clayton_data():
    return sample(cp.Clayton(theta=3), 300, random_state=0)


def test_gof_statistic_definition(clayton_data):
    U = cs.pseudo_obs(clayton_data)
    cop = cp.Clayton(theta=3)
    cn = np.array([np.mean((U[:, 0] <= a) & (U[:, 1] <= b)) for a, b in U])
    ct = np.array([float(cop.cdf(a, b)) for a, b in U])
    assert cs.gof_statistic(cop, U, "cvm") == pytest.approx(np.sum((cn - ct) ** 2))
    assert cs.gof_statistic(cop, U, "ks") == pytest.approx(np.sqrt(300) * np.max(np.abs(cn - ct)))


def test_gof_correct_vs_wrong_family(clayton_data):
    ok = cs.gof_test(cp.Clayton, clayton_data, n_boot=40, fit_method="itau", random_state=1)
    bad = cs.gof_test(cp.GumbelHougaard, clayton_data, n_boot=40, fit_method="itau", random_state=1)
    assert ok.pvalue > 0.05
    assert bad.pvalue < 0.03
    assert bad.reject()
    assert ok.n_boot == 40
    assert ok.boot_statistics.shape == (40,)
    assert "Clayton" in repr(ok)
    again = cs.gof_test(cp.Clayton, clayton_data, n_boot=40, fit_method="itau", random_state=1)
    assert again.pvalue == ok.pvalue  # seeded


def test_gof_from_fitresult_ks_and_no_refit(clayton_data):
    res = cs.fit(cp.Gaussian, clayton_data, method="itau")
    g = cs.gof_test(res, clayton_data, statistic="ks", n_boot=30, random_state=0)
    assert g.fit is res
    assert g.statistic_name == "ks"
    assert 0 < g.pvalue < 1
    g2 = cs.gof_test(res, clayton_data, n_boot=30, refit=False, random_state=0)
    assert not g2.refit
    with pytest.raises(ValueError):
        cs.gof_test(res, clayton_data, statistic="ad")
    with pytest.raises(ValueError):
        cs.gof_test(res, clayton_data, method="multiplier")


@pytest.mark.parametrize(
    "true, fam",
    [(cp.Frank(theta=5), cp.Frank), (cp.Gaussian(rho=0.6), cp.Gaussian)],
)
def test_gof_mle_refit(true, fam):
    X = sample(true, 400, random_state=3)
    r = cs.gof_test(fam, X, n_boot=60, random_state=0)
    assert r.fit.method == "mle"
    assert r.pvalue > 0.05
    wrong = cs.gof_test(cp.Clayton, X, n_boot=60, random_state=0)
    assert wrong.pvalue < 0.05
