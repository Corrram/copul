"""
Universal property tests of the numerical API of every bivariate copula.

Each representative of ``tests/properties/representatives.py`` (all members
of :class:`copul.family_list.Families`, checkerboards, boundary families and
negative-dependence variants) is checked for

* groundedness and uniform margins, 2-increasingness, Fréchet bounds;
* conditional distributions: range, monotonicity, agreement with finite
  differences of the CDF, generalized inverses;
* densities (absolutely continuous copulas): non-negativity, ``logpdf``,
  agreement with mixed finite differences, total mass one;
* sampling: uniform margins, empirical CDF, Kendall's tau, reproducibility;
* call conventions and agreement of the symbolic and numerical routes.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from scipy import stats

from copul.exceptions import PropertyUnavailableException
from tests.properties.representatives import IDS, instance

pytestmark = pytest.mark.filterwarnings("ignore")

GRID = np.linspace(0.0, 1.0, 31)
U, V = np.meshgrid(GRID, GRID, indexing="ij")
_RNG = np.random.default_rng(20260929)
UR, VR = _RNG.uniform(0.02, 0.98, (2, 400))
WR = _RNG.random(400)
N_SAMPLES = 20_000


def _is_ac(cop) -> bool:
    try:
        return bool(cop.is_absolutely_continuous)
    except Exception:
        return True


@pytest.fixture(scope="module", params=IDS)
def cop(request):
    return instance(request.param)


@pytest.fixture(scope="module")
def samples(cop):
    return cop.rvs(N_SAMPLES, random_state=42)


# ---------------------------------------------------------------------------
# distribution function
# ---------------------------------------------------------------------------


def test_grounded_and_uniform_margins(cop):
    c = cop.cdf(U, V)
    assert np.abs(c[:, 0]).max() <= 1e-12  # C(u, 0) = 0
    assert np.abs(c[0, :]).max() <= 1e-12  # C(0, v) = 0
    assert np.abs(c[:, -1] - GRID).max() <= 1e-10  # C(u, 1) = u
    assert np.abs(c[-1, :] - GRID).max() <= 1e-10  # C(1, v) = v


def test_two_increasing(cop):
    c = cop.cdf(U, V)
    mass = c[1:, 1:] - c[:-1, 1:] - c[1:, :-1] + c[:-1, :-1]
    assert mass.min() >= -1e-10
    assert np.isclose(mass.sum(), 1.0, atol=1e-10)


def test_frechet_bounds(cop):
    c = cop.cdf(U, V)
    assert np.all(c >= np.maximum(U + V - 1.0, 0.0) - 1e-12)
    assert np.all(c <= np.minimum(U, V) + 1e-12)


def test_survival_function(cop):
    s = cop.survival_function(UR, VR)
    assert np.allclose(s, 1.0 - UR - VR + cop.cdf(UR, VR), atol=1e-12)
    assert cop.survival_function(0.0, 0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# conditional distributions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", [1, 2])
def test_cond_distr_range_and_monotonicity(cop, which):
    h = getattr(cop, f"cond_distr_{which}")(U, V)
    assert not np.isnan(h).any()
    assert h.min() >= -1e-12 and h.max() <= 1.0 + 1e-12
    # nondecreasing in the non-conditioned argument
    assert np.diff(h, axis=1 if which == 1 else 0).min() >= -1e-9


@pytest.mark.parametrize("which", [1, 2])
def test_cond_distr_matches_cdf_differences(cop, which):
    step = 1e-6
    if which == 1:
        fd = (cop.cdf(UR + step, VR) - cop.cdf(UR - step, VR)) / (2 * step)
    else:
        fd = (cop.cdf(UR, VR + step) - cop.cdf(UR, VR - step)) / (2 * step)
    h = getattr(cop, f"cond_distr_{which}")(UR, VR)
    # singular copulas: the difference quotient straddles a jump with
    # probability of order `step`, hence the robust criterion
    assert np.mean(np.abs(h - fd) > 1e-4) <= 0.01


@pytest.mark.parametrize("which", [1, 2])
def test_cond_distr_inverse(cop, which):
    """``cond_distr_i_inv`` is the (generalized) inverse of ``cond_distr_i``."""
    x = UR if which == 1 else VR
    q = getattr(cop, f"cond_distr_{which}_inv")(x, WR)
    assert np.all((q >= 0) & (q <= 1))
    h = getattr(cop, f"cond_distr_{which}")

    def at(y):
        return h(x, y) if which == 1 else h(y, x)

    # h(q) >= w and h(q - delta) <= w (smallest such point)
    assert np.mean(at(q) < WR - 1e-7) <= 0.01
    assert np.mean(at(np.clip(q - 1e-7, 0, 1)) > WR + 1e-7) <= 0.01
    if _is_ac(cop):
        assert np.median(np.abs(at(q) - WR)) < 1e-8


# ---------------------------------------------------------------------------
# density
# ---------------------------------------------------------------------------


def test_density(cop):
    if not _is_ac(cop):
        pytest.skip("not absolutely continuous")
    p = cop.pdf(UR, VR)
    assert not np.isnan(p).any() and p.min() >= 0.0
    logp = cop.logpdf(UR, VR)
    assert np.allclose(np.exp(logp), p, rtol=1e-8, atol=1e-12)
    step = 1e-4
    fd = (
        cop.cdf(UR + step, VR + step)
        - cop.cdf(UR + step, VR - step)
        - cop.cdf(UR - step, VR + step)
        + cop.cdf(UR - step, VR - step)
    ) / (4 * step * step)
    assert np.mean(np.abs(p - fd) > 1e-3 * (1.0 + np.abs(fd))) <= 0.02
    assert cop.pdf(-0.1, 0.5) == 0.0 and cop.logpdf(0.5, 1.2) == -np.inf


def test_density_integrates_to_one(cop):
    if not _is_ac(cop):
        pytest.skip("not absolutely continuous")
    from copul.measures.quadrature import integrate_2d

    total = integrate_2d(lambda a, b: cop.pdf(a, b), rtol=1e-6, atol=1e-7)
    total = total[0] if isinstance(total, tuple) else total
    if abs(total - 1.0) < 2e-3:
        return
    # densities concentrating at a corner (e.g. lambda_L = 1 for BB2) defeat
    # quadrature on the full square: compare the density's mass on
    # [d, 1]^2 with the C-volume 1 - 2d + C(d, d) of that rectangle instead
    d = 0.05
    part = integrate_2d(
        lambda a, b: (1 - d) ** 2 * cop.pdf(d + (1 - d) * a, d + (1 - d) * b),
        rtol=1e-6,
        atol=1e-7,
    )
    part = part[0] if isinstance(part, tuple) else part
    assert abs(part - (1 - 2 * d + float(cop.cdf(d, d)))) < 2e-3


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_rvs_uniform_margins(samples):
    assert samples.shape == (N_SAMPLES, 2)
    assert np.all((samples >= 0) & (samples <= 1))
    for j in range(2):
        assert stats.kstest(samples[:, j], "uniform").pvalue > 1e-3


def test_rvs_empirical_cdf(cop, samples):
    g = np.linspace(0.05, 0.95, 10)
    a, b = np.meshgrid(g, g, indexing="ij")
    emp = np.mean(
        (samples[:, 0][None, None, :] <= a[..., None])
        & (samples[:, 1][None, None, :] <= b[..., None]),
        axis=-1,
    )
    assert np.abs(emp - cop.cdf(a, b)).max() < 0.02


def test_rvs_kendalls_tau(cop, samples):
    tau_hat = stats.kendalltau(samples[:, 0], samples[:, 1]).statistic
    tau = float(cop.kendalls_tau())
    # asymptotic s.e. of the U-statistic: 4 sqrt(Var(2 C(U,V) - U - V) / n)
    c = cop.cdf(samples[:, 0], samples[:, 1])
    se = 4.0 * np.sqrt(np.var(2 * c - samples[:, 0] - samples[:, 1]) / N_SAMPLES) + 1e-3
    assert abs(tau_hat - tau) <= 4.0 * se


def test_rvs_reproducible(cop):
    a = cop.rvs(50, random_state=7)
    assert np.array_equal(a, cop.rvs(50, random_state=7))
    b = cop.rvs(50, random_state=np.random.default_rng(7))
    assert b.shape == (50, 2)


# ---------------------------------------------------------------------------
# call conventions and symbolic/numeric agreement
# ---------------------------------------------------------------------------


def test_call_conventions(cop):
    pts = np.column_stack([UR[:5], VR[:5]])
    for name in ("cdf", "cond_distr_1", "cond_distr_2", "survival_function"):
        f = getattr(cop, name)
        vec = f(UR[:5], VR[:5])
        assert isinstance(vec, np.ndarray) and vec.shape == (5,)
        assert np.allclose(f(pts), vec)
        scalar = f(UR[0], VR[0])
        assert isinstance(scalar, float)
        assert scalar == pytest.approx(vec[0])
        assert f(u=UR[0], v=VR[0]) == pytest.approx(vec[0])
        assert np.allclose(f(UR[:5, None], VR[None, :3]).shape, (5, 3))
    # clipping
    assert cop.cdf(-0.5, 0.3) == 0.0
    assert cop.cdf(1.5, 0.3) == pytest.approx(0.3)


_SYMBOLIC_POINTS = [(0.3, 0.6), (0.7, 0.2), (0.42, 0.61), (0.85, 0.9)]


@pytest.mark.parametrize("name", ["cdf", "cond_distr_1", "cond_distr_2", "pdf"])
def test_symbolic_and_numeric_agree(cop, name):
    """Where a family has a symbolic expression in (u, v), it matches the numerics."""
    try:
        attr = getattr(cop, name)
        wrapper = attr() if callable(attr) else attr
    except (PropertyUnavailableException, NotImplementedError, TypeError, ValueError):
        pytest.skip("no symbolic form")
    expr = getattr(wrapper, "func", None)
    if not isinstance(expr, sp.Expr):
        pytest.skip("no symbolic form")
    syms = {str(s): s for s in expr.free_symbols}
    if set(syms) - {"u", "v"} or expr.has(sp.Integral, sp.Subs, sp.Derivative):
        pytest.skip("symbolic form not closed")
    u_sym, v_sym = syms.get("u", sp.Symbol("u")), syms.get("v", sp.Symbol("v"))
    for a, b in _SYMBOLIC_POINTS:
        sym = complex(sp.N(expr.subs({u_sym: a, v_sym: b})))
        assert sym.real == pytest.approx(getattr(cop, name)(a, b), rel=1e-6, abs=1e-9)
