"""Population atlas definitions, independent analytic and numerical checks."""

import math

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import quad
from scipy.special import ndtri

import copul as cp
from copul.measures import (
    ATLAS_MEASURES,
    NumericCopula,
    compute,
    get_measure,
    measures_from_cdf,
    measures_from_h,
    numeric_backend,
)
from copul.measures.atlas import KEYS, _hellinger
from copul.measures.numeric import evaluate


@pytest.mark.parametrize("key", ATLAS_MEASURES)
@pytest.mark.parametrize(
    "factory, endpoint",
    [
        (cp.BivIndependenceCopula, "at_Pi"),
        (cp.UpperFrechet, "at_M"),
        (cp.LowerFrechet, "at_W"),
    ],
)
def test_atlas_endpoints(key, factory, endpoint):
    assert len(ATLAS_MEASURES) == len(set(ATLAS_MEASURES)) == 23
    assert compute(factory(), key) == pytest.approx(getattr(get_measure(key), endpoint), abs=2e-6)


FGM = {
    "eta": lambda t: t / 3,
    "normal_score": lambda t: t / np.pi,
    "savage_score": lambda t: t / 4,
    "omega": lambda t: abs(t) / np.sqrt(15),
    "chi": lambda t: t * t / 9,
    "integrated_r2": lambda t: t * t / 18,
    "hoeffding_d": lambda t: t * t / 30,
    "tau_star": lambda t: t * t / 25,
    "silvey": lambda t: abs(t) / 8,
}


@pytest.mark.parametrize("t", [-0.8, 0.0, 0.7])
@pytest.mark.parametrize("key", FGM)
def test_fgm_against_analytic_integrals(t, key):
    copula = cp.FarlieGumbelMorgenstern(t)
    expected = FGM[key](t)
    assert compute(copula, key, method="numeric") == pytest.approx(expected, abs=3e-8)
    assert getattr(copula, get_measure(key).method_name)() == pytest.approx(expected, abs=3e-8)


@pytest.mark.parametrize("key", ["eta", "omega", "chi", "integrated_r2", "hoeffding_d", "tau_star"])
def test_symbolic_fgm(key):
    c = cp.FarlieGumbelMorgenstern()
    result = compute(c, key, method="symbolic")
    assert isinstance(result, sp.Expr)
    assert float(result.subs(c.theta, 0.7)) == pytest.approx(FGM[key](0.7))
    with pytest.raises(ValueError, match="free parameters"):
        compute(c, key, method="numeric")


def test_legacy_naming_is_preserved_and_unambiguous_name_is_added():
    c = cp.FarlieGumbelMorgenstern(0.6)
    assert c.hoeffding_d() == pytest.approx(0.6**2 / 30)
    assert c.hoeffdings_d() == pytest.approx(0.6**2 / 10)
    assert compute(c, "phi2") == pytest.approx(c.hoeffdings_d())
    assert compute(c, "bkr") == pytest.approx(c.hoeffding_d())
    assert compute(c, "rho_s") == pytest.approx(c.spearmans_rho())


@pytest.mark.parametrize("rho", [-0.7, 0.4])
@pytest.mark.parametrize("key", ["normal_score", "linfoot", "hellinger", "maximal_correlation"])
def test_gaussian_calibration(rho, key):
    assert compute(cp.Gaussian(rho), key) == pytest.approx(
        rho if key == "normal_score" else abs(rho)
    )


@pytest.mark.parametrize("key", sorted(KEYS - {"maximal_correlation"}))
def test_pure_numeric_api_matches_object(key):
    c = cp.FarlieGumbelMorgenstern(0.6)
    be = numeric_backend(c)
    value = measures_from_cdf(be.cdf, key, h1=be.get("h1"), h2=be.get("h2"), pdf=be.get("pdf"))
    assert value == pytest.approx(compute(c, key), abs=3e-7)


def test_maximal_correlation_resolution_and_diagnostics():
    # The FGM operator is rank one: R=|theta| ||1-2u||_2^2=|theta|/3.
    c = cp.FarlieGumbelMorgenstern(0.7)
    low = compute(c, "R", max_grid=32, full_output=True)
    high = compute(c, "R", max_grid=128, full_output=True)
    assert low.value < high.value < 0.7 / 3
    assert high.value == pytest.approx(0.7 / 3, abs=2e-5)
    assert "not an error bound" in high.info["source"]["maximal_correlation"]
    assert high.error > 0
    for invalid in [True, 3, 24, 32.0, -2]:
        with pytest.raises(ValueError, match="max_grid"):
            c.maximal_correlation(max_grid=invalid)


@pytest.mark.parametrize(
    "key", ["eta", "omega", "chi", "integrated_r2", "normal_score", "savage_score"]
)
@pytest.mark.parametrize("sign", [0, 1, -1])
def test_rectangular_checkerboard_against_generic_integrals(key, sign):
    p = np.array([[0.25, 0.125, 0.125], [1 / 12, 5 / 24, 5 / 24]])
    c = cp.BivCheckMixed(p, np.full(p.shape, sign))
    actual = compute(c, key)
    if sign and key in ("normal_score", "savage_score", "integrated_r2"):
        # Independent scalar quadrature split at every conditional jump.
        be = numeric_backend(c)

        def outer(v):
            j = min(int(3 * v), 2)
            b = 3 * v - j
            local = b if sign == 1 else 1 - b
            points = sorted({0.0, 0.5, 1.0, local / 2, (1 + local) / 2})

            def inner(u):
                if key == "integrated_r2":
                    return (float(be.get("h1")(u, v)) - v) ** 2 / (v * (1 - v))
                diff = float(be.cdf(u, v)) - u * v
                if key == "savage_score":
                    return diff / ((1 - u) * (1 - v))
                return diff * 2 * np.pi * np.exp((ndtri(u) ** 2 + ndtri(v) ** 2) / 2)

            return quad(inner, 0, 1, points=points, epsabs=1e-8)[0]

        expected = quad(outer, 0, 1, points=[1 / 3, 2 / 3], epsabs=1e-7)[0]
    else:
        expected, _ = evaluate(key, numeric_backend(c), rtol=1e-6, atol=1e-7)
    assert actual == pytest.approx(expected, abs=2e-6)


def test_checkerboard_maximal_correlation():
    # Binary symmetric channel with contrast t has singular values 1, |t|.
    for t in [0.0, 0.3, -0.8, 1.0]:
        p = np.array([[1 + t, 1 - t], [1 - t, 1 + t]]) / 4
        result = compute(cp.BivCheckPi(p), "R", full_output=True)
        assert result.value == pytest.approx(abs(t), abs=1e-14)
        assert result.error == 0


@pytest.mark.parametrize("sign", [[[1, -1], [0, 1]], [[1, -1], [-1, 1]]])
def test_mixed_checkerboard_maximal_correlation_against_partitions(sign):
    from copul.measures.atlas import _maximal

    c = cp.BivCheckMixed([[0.35, 0.15], [0.15, 0.35]], sign)
    # On this aligned grid the even centered local step functions are
    # already represented by the finite partition.
    reference, _ = _maximal(c.cdf, 16)
    assert c.maximal_correlation() == pytest.approx(reference, abs=1e-12)


def test_checkerboard_score_integrals():
    p = np.array([[0.4, 0.1], [0.1, 0.4]])
    c = cp.BivCheckPi(p)
    normal = np.array([quad(ndtri, i / 2, (i + 1) / 2)[0] * 2 for i in range(2)])
    savage = np.array([quad(lambda u: -np.log1p(-u), i / 2, (i + 1) / 2)[0] * 2 for i in range(2)])
    assert c.normal_score_correlation() == pytest.approx(normal @ p @ normal)
    assert c.savage_score_correlation() == pytest.approx(savage @ p @ savage - 1)


@pytest.mark.parametrize("t", [1e-12, 0.2, 0.6, 1.0])
def test_mixed_frechet_divergences(t):
    c = cp.Frechet(t, 0)
    assert c.linfoot_correlation() == 1
    assert c.silvey_delta() == pytest.approx(t, abs=1e-15)
    affinity = math.sqrt(1 - t)
    assert c.hellinger_correlation() == pytest.approx(_hellinger(affinity))
    assert c.integrated_r2() == pytest.approx(t * t)


@pytest.mark.parametrize("sign", [np.zeros((2, 2)), np.ones((2, 2)), [[0, 1], [-1, 0]]])
def test_checkerboard_divergences_with_singular_cells(sign):
    p = np.array([[0.35, 0.15], [0.15, 0.35]])
    c = cp.BivCheckMixed(p, sign)
    ac = np.asarray(sign) == 0
    density = 4 * p * ac
    affinity = np.sqrt(density).mean()
    singular_mass = np.sum(p[~ac])
    assert c.silvey_delta() == pytest.approx((np.abs(density - 1).mean() + singular_mass) / 2)
    assert c.hellinger_correlation() == pytest.approx(_hellinger(affinity))
    expected_l = 1.0 if singular_mass else math.sqrt(1 - math.exp(-2 * np.sum(p * np.log(4 * p))))
    assert c.linfoot_correlation() == pytest.approx(expected_l)


def test_hellinger_stable_limits():
    assert _hellinger(0) == 1
    assert _hellinger(1) == 0
    assert math.isfinite(_hellinger(1e-30))
    assert 0 < _hellinger(1 - 1e-14) < 1e-5


@pytest.mark.parametrize("key", ["chi", "integrated_r2"])
def test_directional_coefficients_on_asymmetric_checkerboard(key):
    p = np.array([[0.25, 0.125, 0.125], [1 / 12, 5 / 24, 5 / 24]])
    c, ct = cp.BivCheckPi(p), cp.BivCheckPi(p.T)
    assert compute(c, key, condition_on_y=True) == pytest.approx(compute(ct, key), abs=1e-12)
    assert abs(compute(c, key) - compute(ct, key)) > 1e-4
    method = getattr(c, get_measure(key).method_name)
    assert method(condition_on_y=True) == pytest.approx(compute(ct, key))


def test_generic_directional_coefficients():
    # Polynomial copula with different regressions in the two directions.
    u, v = sp.symbols("u v", positive=True)
    from copul.wrapper.cdf_wrapper import CDFWrapper

    class Polynomial(cp.BivCopula):
        @property
        def cdf(self):
            return CDFWrapper(u * v + sp.Rational(1, 2) * u * (1 - u) * (1 - 2 * u) * v * (1 - v))

    class Transposed(cp.BivCopula):
        @property
        def cdf(self):
            return CDFWrapper(u * v + sp.Rational(1, 2) * v * (1 - v) * (1 - 2 * v) * u * (1 - u))

    c, ct = Polynomial(), Transposed()
    for key in ["chi", "integrated_r2"]:
        assert compute(c, key, condition_on_y=True) == pytest.approx(compute(ct, key), abs=1e-9)
        assert abs(compute(c, key) - compute(ct, key)) > 1e-4


def test_h_callable_new_conditional_measures():
    h = lambda u, v: v + 0.7 * v * (1 - v) * (1 - 2 * u)
    out = measures_from_h(h, ["chi", "integrated_r2"])
    assert out["chi"] == pytest.approx(0.7**2 / 9)
    assert out["integrated_r2"] == pytest.approx(0.7**2 / 18)


def test_fgm_divergence_independent_integrals():
    t = 0.7

    # Products of centered uniforms: E[g(XY)] = -int_0^1 log(z)[g(z)+g(-z)] dz / 2.
    # Here X,Y are uniform on [-1,1].
    def expectation(f):
        return quad(lambda z: -np.log(z) * (f(1 + t * z) + f(1 - t * z)) / 2, 0, 1, epsabs=1e-12)[0]

    information = expectation(lambda c: c * np.log(c))
    affinity = expectation(np.sqrt)
    c = cp.FarlieGumbelMorgenstern(t)
    assert c.linfoot_correlation() == pytest.approx(np.sqrt(-np.expm1(-2 * information)), abs=1e-9)
    assert c.hellinger_correlation() == pytest.approx(_hellinger(affinity), abs=1e-9)


def test_tiny_singular_mass_generic_linfoot():
    from copul.measures.atlas import evaluate as atlas_evaluate

    funcs = NumericCopula(pdf=lambda u, v: np.zeros_like(u) + 1 - 1e-12)
    funcs._ac = False
    assert atlas_evaluate("linfoot", funcs, 1e-8, 1e-10) == (1.0, 0.0)


@pytest.mark.parametrize("key", ["linfoot", "hellinger", "silvey"])
def test_divergences_reject_unknown_density_decomposition(key):
    with pytest.raises(NotImplementedError, match="explicit"):
        measures_from_cdf(np.minimum, key)


@pytest.mark.parametrize("rho", [0.01, 0.5, 0.99])
def test_hellinger_gaussian_affinity_transformation(rho):
    b = (16 * (1 - rho * rho) / (4 - rho * rho) ** 2) ** 0.25
    assert _hellinger(b) == pytest.approx(rho, abs=1e-12)
