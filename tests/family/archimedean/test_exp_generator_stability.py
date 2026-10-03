"""Overflow-free numerics of Nelsen 4.2.19 / 4.2.20 near the lower tail."""

import numpy as np
import pytest

import copul as cp


def _reference(family, theta, u, v):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    if family == "Nelsen19":
        g, ginv = (lambda t: theta / t), (lambda L: theta / L)
    else:
        g, ginv = (lambda t: t ** (-theta)), (lambda L: L ** (-1 / theta))

    def C(a, b):
        return ginv(mp.log(mp.e ** g(a) + mp.e ** g(b) - mp.e ** g(mp.mpf(1))))

    a, b = mp.mpf(u), mp.mpf(v)
    return float(C(a, b)), float(mp.diff(lambda x: C(x, b), a))


@pytest.mark.parametrize("family", ["Nelsen19", "Nelsen20"])
@pytest.mark.parametrize("theta", [0.7, 2.0])
@pytest.mark.parametrize("u,v", [(1e-3, 0.5), (0.005, 0.4), (0.3, 0.7), (0.02, 0.021)])
def test_cdf_and_h1_match_high_precision(family, theta, u, v):
    cop = getattr(cp, family)(theta)
    c_ref, h_ref = _reference(family, theta, u, v)
    assert cop.cdf(u, v) == pytest.approx(c_ref, rel=1e-10, abs=1e-15)
    assert cop.cond_distr_1(u, v) == pytest.approx(h_ref, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("family", ["Nelsen19", "Nelsen20"])
def test_no_spurious_zeros_near_lower_tail(family):
    cop = getattr(cp, family)(1.0)
    u = np.geomspace(1e-6, 0.1, 30)
    vals = cop.cdf(u, np.full_like(u, 0.5))
    assert np.all(vals > 0)
    np.testing.assert_allclose(vals, u, rtol=1e-3)
