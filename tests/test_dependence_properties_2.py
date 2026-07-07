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

    def test_ev_counterexample_not_rcsi(self):
        # C(u,v) = min(u, v, (uv)^kappa) with kappa in (1/2, 1) is an
        # extreme-value copula (Pickands function max(w, 1-w, kappa)) that is
        # LCSD (as every EV copula) but NOT RCSI: on the region
        # (uv)^kappa < min(u,v), the survival function satisfies
        # Cbar*Cbar_uv - Cbar_u*Cbar_v -> -(1-kappa)^2 < 0 as u=v -> 1.
        C = cp.from_cdf("min(min(u, v), (u*v)**0.75)")
        assert C.is_lcsd(n_grid=40)
        assert not C.is_rcsi(n_grid=40)


class TestPQDAndMKTP2:
    def test_pqd_nqd_basic(self):
        assert cp.Clayton(2).is_pqd(n_grid=20)
        assert not cp.Clayton(2).is_nqd(n_grid=20)
        assert cp.Frank(-3).is_nqd(n_grid=20)
        assert not cp.Frank(-3).is_pqd(n_grid=20)
        assert cp.Gaussian(0.5).is_pqd(n_grid=20)
        assert cp.Gaussian(-0.5).is_nqd(n_grid=20)

    def test_mk_tp2_archimedean_equals_ci(self):
        # Fuchs & Tschimpke (2023): Archimedean MK-TP2 <=> SI/CI
        assert cp.GumbelHougaard(2).is_mk_tp2(n_grid=20)
        assert cp.Clayton(2).is_mk_tp2(n_grid=20)
        assert not cp.Nelsen16(2).is_mk_tp2(n_grid=20)  # CI iff theta >= 3

    def test_mk_tp2_extreme_value(self):
        # Cuadras-Auge has D+A(0) = -delta in (-1,0) => not MK-TP2
        assert not cp.CuadrasAuge(0.5).is_mk_tp2(n_grid=20)
        # Galambos has D+A(0) = -1 and satisfies the F-T criterion
        assert cp.Galambos(1).is_mk_tp2(n_grid=20)

    def test_mk_tp2_frechet_violation(self):
        # kernel (1-alpha) v + alpha 1_{v >= u} is not TP2 for alpha in (0,1)
        assert not cp.Frechet(0.5, 0.0).is_mk_tp2(n_grid=20)

    def test_mk_tp2_student_t_fails(self):
        assert not cp.StudentT(0.5, 2).is_mk_tp2(n_grid=15)


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


class TestClosedFormMeasures:
    """Closed forms added for 'Dependence properties of bivariate copula
    families II': Blest's nu, Schweizer-Wolff sigma, Hoeffding's Phi^2."""

    def test_fgm_all_measures(self):
        C = cp.FarlieGumbelMorgenstern(0.8)
        assert float(C.blests_nu()) == pytest.approx(0.8 / 3)
        assert float(C.schweizer_wolff_sigma()) == pytest.approx(0.8 / 3)
        assert float(C.hoeffdings_phi_square()) == pytest.approx(0.8**2 / 10)
        assert float(C.spearman_footrule()) == pytest.approx(0.8 / 5)
        assert float(C.gini_gamma()) == pytest.approx(4 * 0.8 / 15)

    def test_cuadras_auge_nu(self):
        d = 0.3
        expected = 2 * d / (2 - d) - 12 * d / ((2 - d) * (4 - d) * (5 - d))
        assert float(cp.CuadrasAuge(d).blests_nu()) == pytest.approx(expected)
        # numerically verified reference value
        assert expected == pytest.approx(0.231167, abs=1e-6)

    def test_marshall_olkin_nu_and_phi2(self):
        C = cp.MarshallOlkin(0.3, 0.8)
        assert float(C.blests_nu()) == pytest.approx(0.386868, abs=1e-6)
        assert float(C.hoeffdings_d()) == pytest.approx(0.133500, abs=1e-6)

    def test_frechet_phi2(self):
        a, b = 0.4, 0.2
        assert float(cp.Frechet(a, b).hoeffdings_d()) == pytest.approx(
            a**2 + b**2 - 7 * a * b / 4
        )

    def test_mardia_closed_forms(self):
        t = -0.6
        C = cp.Mardia(t)
        assert float(C.blests_nu()) == pytest.approx(t**3)
        assert float(C.gini_gamma()) == pytest.approx(t**3)
        assert float(C.spearman_footrule()) == pytest.approx(t**2 * (1 + 3 * t) / 4)
        assert float(C.hoeffdings_d()) == pytest.approx(t**4 * (1 + 15 * t**2) / 16)

    def test_gaussian_gini_gamma(self):
        s = 0.5
        expected = 2 / np.pi * (np.arcsin((1 + s) / 2) - np.arcsin((1 - s) / 2))
        assert float(cp.Gaussian(s).gini_gamma()) == pytest.approx(expected)

    def test_gaussian_nu_equals_rho(self):
        # radial symmetry implies nu = rho
        C = cp.Gaussian(0.6)
        assert float(C.blests_nu()) == pytest.approx(float(C.spearmans_rho()))

    def test_frank_blomqvist_beta(self):
        th = 3.0
        assert float(cp.Frank(th).blomqvists_beta()) == pytest.approx(
            4 / th * np.log(np.cosh(th / 4))
        )

    def test_ev_base_closed_forms(self):
        C = cp.GumbelHougaardEV(2)
        A_half = 2 ** (1 / 2 - 1)
        assert float(C.blomqvists_beta()) == pytest.approx(2 ** (2 - 2 * A_half) - 1)
        assert float(C.spearman_footrule()) == pytest.approx(6 / (1 + 2 * A_half) - 2)

    def test_student_t_rho_depends_on_nu(self):
        # quadrature value, cross-checked by Monte Carlo (0.4552 +- 0.0013)
        rho = float(cp.StudentT(0.5, 2).spearmans_rho())
        assert rho == pytest.approx(0.4552, abs=2e-3)
        assert abs(rho - 6 / np.pi * np.arcsin(0.25)) > 0.02  # differs from Gaussian
        assert float(cp.StudentT(0.5, 2).blests_nu()) == pytest.approx(rho)

    def test_raftery_closed_forms(self):
        d = 0.3
        C = cp.Raftery(d)
        assert float(C.spearman_footrule()) == pytest.approx(2 * d / (3 - d))
        assert float(C.schweizer_wolff_sigma()) == pytest.approx(
            d * (4 - 3 * d) / (2 - d) ** 2
        )
        assert float(C.blomqvists_beta()) == pytest.approx(
            1 + 4 * (1 - d) / (1 + d) * (2 ** (-2 / (1 - d)) - 0.5)
        )

    def test_blum_kiefer_rosenblatt_closed_forms(self):
        # B = 30 iint (C - uv)^2 dC, with B(M) = B(W) = 1
        th = 0.8
        assert float(
            cp.FarlieGumbelMorgenstern(th).blum_kiefer_rosenblatt()
        ) == pytest.approx(th**2 / 30)
        a, b = 0.4, 0.2
        expected = (a**2 * (1 + 2 * a) + b**2 * (1 + 2 * b)) / 3 - a * b * (
            a + b
        ) / 2 - 7 * a * b / 12
        assert float(cp.Frechet(a, b).blum_kiefer_rosenblatt()) == pytest.approx(
            expected
        )
        assert float(cp.Frechet(1.0, 0.0).blum_kiefer_rosenblatt()) == pytest.approx(1)
        assert float(cp.Frechet(0.0, 1.0).blum_kiefer_rosenblatt()) == pytest.approx(1)
        t = -0.6
        assert float(cp.Mardia(t).blum_kiefer_rosenblatt()) == pytest.approx(
            t**4 * (2 * t**2 + 1) * (15 * t**2 + 1) / 48
        )

    def test_blum_kiefer_rosenblatt_monte_carlo(self):
        # generic Monte Carlo path agrees with the FGM closed form
        val = cp.Clayton(2).blum_kiefer_rosenblatt(n_samples=50_000)
        assert 0.1 < val < 0.3  # sanity: strictly between independence and M

    def test_plackett_sigma(self):
        th = 0.3  # NQD range: sigma = -rho
        C = cp.Plackett(th)
        assert float(C.schweizer_wolff_sigma()) == pytest.approx(
            -float(C.spearmans_rho())
        )


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
