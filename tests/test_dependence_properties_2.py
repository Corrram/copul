"""Tests for the dependence properties added for the 'Dependence properties
of bivariate copula families 2' paper: tail monotonicity (LTD/RTI), corner
set monotonicity (LCSD/RCSI), symmetry checks, and tail orders."""

import numpy as np
import pytest

import copul as cp


class TestTailMonotonicity:
    def test_clayton_positive_is_ltd_and_rti(self):
        C = cp.Clayton(2)
        assert C.is_ltd(n_grid=12)
        assert C.is_rti(n_grid=12)

    def test_frank_negative_is_not_ltd(self):
        C = cp.Frank(-5)
        assert not C.is_ltd(n_grid=12)

    def test_independence_is_ltd_and_rti(self):
        C = cp.BivIndependenceCopula()
        assert C.is_ltd(n_grid=10)
        assert C.is_rti(n_grid=10)


class TestCornerSetMonotonicity:
    def test_clayton_positive_is_lcsd_and_rcsi(self):
        # Clayton with theta >= 0 is TP2, hence LCSD and RCSI
        C = cp.Clayton(2)
        assert C.is_lcsd(n_grid=20)
        assert C.is_rcsi(n_grid=20)

    def test_verifier_class_directly(self):
        verifier = cp.CornerSetVerifier(n_grid=20)
        assert verifier.is_lcsd(cp.GumbelHougaard(2))
        assert verifier.is_rcsi(cp.GumbelHougaard(2))

    def test_frank_negative_is_not_lcsd(self):
        C = cp.Frank(-8)
        assert not C.is_lcsd(n_grid=20)


class TestSymmetries:
    def test_archimedean_exchangeable(self):
        assert cp.Clayton(2).is_exchangeable(n_grid=8)

    def test_marshall_olkin_asymmetric(self):
        C = cp.MarshallOlkin(0.2, 0.8)
        assert not C.is_exchangeable(n_grid=8)

    def test_frank_radially_symmetric(self):
        assert cp.Frank(3).is_radially_symmetric(n_grid=8)

    def test_fgm_radially_symmetric(self):
        assert cp.FarlieGumbelMorgenstern(0.5).is_radially_symmetric(n_grid=8)

    def test_clayton_not_radially_symmetric(self):
        assert not cp.Clayton(2).is_radially_symmetric(n_grid=8)

    def test_gaussian_radially_symmetric(self):
        assert cp.Gaussian(0.5).is_radially_symmetric(n_grid=8)


class TestTailOrder:
    def test_gaussian_tail_order(self):
        # kappa = 2 / (1 + rho), Hua & Joe (2011)
        res = cp.Gaussian(0.5).tail_order()
        assert res["lower"] == pytest.approx(4 / 3, rel=1e-6)
        assert res["upper"] == pytest.approx(4 / 3, rel=1e-6)

    def test_clayton_tail_order(self):
        res = cp.Clayton(2).tail_order()
        assert res["lower"] == pytest.approx(1.0, abs=1e-9)
        assert res["upper"] == pytest.approx(2.0, rel=1e-2)

    def test_gumbel_hougaard_tail_order(self):
        # kappa_L = 2^{1/theta}, kappa_U = 1
        res = cp.GumbelHougaard(2).tail_order()
        assert res["lower"] == pytest.approx(np.sqrt(2), rel=1e-3)
        assert res["upper"] == pytest.approx(1.0, abs=1e-9)

    def test_frank_tail_quadrant_independence(self):
        res = cp.Frank(3).tail_order()
        assert res["lower"] == pytest.approx(2.0, rel=1e-2)
        assert res["upper"] == pytest.approx(2.0, rel=1e-2)


class TestMeasuresNumericFallback:
    def test_footrule_clayton_numeric(self):
        val = float(cp.Clayton(2).spearman_footrule())
        # psi = 6 * int C(t,t) dt - 2 with C(t,t) = t (2 - t^2)^{-1/2}
        from scipy.integrate import quad

        expected = 6 * quad(lambda t: t / np.sqrt(2 - t**2), 0, 1)[0] - 2
        assert val == pytest.approx(expected, rel=1e-6)

    def test_gini_gamma_independence_zero(self):
        val = float(cp.BivIndependenceCopula().gini_gamma())
        assert val == pytest.approx(0.0, abs=1e-8)

    def test_blomqvist_beta_frechet_bounds(self):
        assert float(cp.UpperFrechet().blomqvists_beta()) == pytest.approx(1.0)
        assert float(cp.LowerFrechet().blomqvists_beta()) == pytest.approx(-1.0)
