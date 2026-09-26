"""Registry, API and validation (a)+(b) of the registered exact regions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import copul.regions as cr
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.optim.checkerboard_formulas import measure_values
from copul.regions._numeric import numeric_measures
from copul.regions.catalog import (
    nu_max_given_rho,
    nu_max_given_xi,
    rho_max_given_xi,
    xi_nu_parametric,
)
from copul.schur_order import bounds_from_xi as bx
from copul.search import si_rearrangement

ALL = cr.available(None)


def test_registry_contents_and_lookup():
    assert cr.available() == [("rho", "nu"), ("xi", "beta"), ("xi", "nu"), ("xi", "rho")]
    assert ("xi", "footrule", "si") in ALL
    reg = cr.get("Chatterjee", "Spearman")
    assert reg is cr.get("xi", "rho")
    sw = cr.get("rho", "xi")
    assert isinstance(sw, cr.SwappedRegion) and (sw.x, sw.y) == ("rho", "xi")
    assert sw is cr.get("rho", "xi")  # cached
    with pytest.raises(KeyError):
        cr.get("xi", "tau")
    with pytest.raises(KeyError):
        cr.get("xi", "footrule")  # only the SI region is exact
    with pytest.raises(KeyError):
        cr.register(cr.get("xi", "rho"))
    for key in ALL:
        r = cr.get(*key)
        assert r.reference and r.source and r.key == key
        assert "(" in repr(r) and r.title.startswith("$")


def test_vectorised_formulas_match_package_scalars():
    xs = np.linspace(0, 1, 101)
    np.testing.assert_allclose(
        rho_max_given_xi(xs), [bx.rho_max_given_xi(float(x)) for x in xs], atol=1e-14
    )
    inner = xs[1:-1]
    np.testing.assert_allclose(
        nu_max_given_xi(inner),
        [bx.nu_bounds_from_xi(float(x))[1] for x in inner],
        atol=1e-9,
    )
    # parametric closed forms from the V-threshold class
    from copul.family.other.v_threshold_copula import VThresholdCopula

    for mu in (0.0, 0.3, 1.0, 1.4, 2.0):
        c = VThresholdCopula(mu=mu)
        assert nu_max_given_rho(c.spearmans_rho()) == pytest.approx(c.blests_nu(), abs=1e-12)
    x1, n1 = xi_nu_parametric(np.array([1.0]))
    assert x1[0] == pytest.approx(32 / 105) and n1[0] == pytest.approx(76 / 105)


@pytest.mark.parametrize("key", ALL)
def test_boundary_shape_and_key_points(key):
    reg = cr.get(*key)
    xs = np.linspace(*reg.x_range, 301)
    lo, up = reg.lower(xs), reg.upper(xs)
    assert np.all(lo <= up + 1e-12)
    assert np.isnan(reg.upper(reg.x_range[1] + 0.1))
    assert reg.contains(xs, 0.5 * (lo + up)).all()
    assert not reg.contains(xs, up + 1e-6).any()
    assert not reg.contains(xs, lo - 1e-6).any()
    assert not reg.contains(reg.x_range[0] - 1e-6, 0.0)
    for p in reg.key_points:
        assert reg.contains(p.x, p.y, tol=1e-9), p
        if p.copula is not None:
            c = p.copula()
            nm = numeric_measures(c, (reg.x, reg.y), N=200)
            assert nm[reg.x] == pytest.approx(p.x, abs=1e-2)
            assert nm[reg.y] == pytest.approx(p.y, abs=1e-3)
    poly = reg.sample_boundary(50)
    assert poly.shape == (101, 2) and np.allclose(poly[0], poly[-1])


# (a) boundary copulas attain the boundary --------------------------------
SIDES = [(k, s) for k in ALL for s in ("upper", "lower") if cr.get(*k).has_boundary_family[s]]


@pytest.mark.parametrize("key,side", SIDES)
def test_boundary_copulas_attain_boundary(key, side):
    reg = cr.get(*key)
    a, b = reg.x_range
    for x in np.linspace(a, b, 6)[1:-1]:
        C = reg.boundary_copula(x, side)
        nm = numeric_measures(C, (reg.x, reg.y), N=300)
        target = reg.upper(x) if side == "upper" else reg.lower(x)
        singular = key == ("xi", "footrule", "si")  # Frechet mixtures have an M part
        assert nm[reg.x] == pytest.approx(x, abs=1e-2 if singular else 2e-4), (x, nm)
        assert nm[reg.y] == pytest.approx(float(target), abs=1e-4), (x, nm)


def test_boundary_copulas_own_closed_forms():
    reg = cr.get("xi", "rho")
    for x in (0.1, 0.3, 0.8):
        up, lo = reg.boundary_copula(x, "upper"), reg.boundary_copula(x, "lower")
        assert float(up.chatterjees_xi()) == pytest.approx(x, abs=1e-12)
        assert float(up.spearmans_rho()) == pytest.approx(float(reg.upper(x)), abs=1e-12)
        assert float(lo.spearmans_rho()) == pytest.approx(float(reg.lower(x)), abs=1e-12)
    reg = cr.get("xi", "nu")
    for x in (0.1, 0.5):
        c = reg.boundary_copula(x, "upper")
        assert float(c.chatterjees_xi()) == pytest.approx(x, abs=1e-9)
        assert float(c.blests_nu()) == pytest.approx(float(reg.upper(x)), abs=1e-9)
        assert float(reg.boundary_copula(x, "lower").blests_nu()) == pytest.approx(
            float(reg.lower(x)), abs=1e-9
        )
    reg = cr.get("rho", "nu")
    for r in (-0.5, 0.0, 0.5):
        c = reg.boundary_copula(r, "upper")
        assert c.spearmans_rho() == pytest.approx(r, abs=1e-10)
        assert c.blests_nu() == pytest.approx(float(reg.upper(r)), abs=1e-10)
    assert reg.boundary_copula(0.2, "lower") is None
    assert type(reg.boundary_copula(1.0, "upper")).__name__ == "UpperFrechet"
    with pytest.raises(ValueError):
        reg.boundary_copula(2.0)
    with pytest.raises(ValueError):
        reg.boundary_copula(0.0, "middle")


# (b) random checkerboards lie inside --------------------------------------
@pytest.fixture(scope="module")
def random_pi():
    cops = BivCheckPi.generate_diverse(2000, grid_size=(2, 60), rng=20260926)
    vals = []
    for c in cops:
        vals.append(
            {
                "xi": c.chatterjees_xi(),
                "rho": c.spearmans_rho(),
                "nu": c.blests_nu(),
                "footrule": c.spearmans_footrule(),
                "beta": c.blomqvists_beta(),
            }
        )
    return cops, vals


@pytest.mark.parametrize("key", [k for k in ALL if k[2] == "all"])
def test_random_checkerboards_inside(key, random_pi):
    reg = cr.get(*key)
    cops, vals = random_pi
    for kind, values in (
        ("pi", vals),
        ("min", [measure_values(c.matr, "min", (reg.x, reg.y)) for c in cops]),
        ("w", [measure_values(c.matr, "w", (reg.x, reg.y)) for c in cops]),
    ):
        x = np.array([v[reg.x] for v in values])
        y = np.array([v[reg.y] for v in values])
        margin = reg.margin(x, y)
        assert np.all(margin <= 1e-9), (kind, margin.max(), int(np.argmax(margin)))


def test_random_si_checkerboards_inside_si_region(random_pi):
    reg = cr.get("xi", "footrule", "si")
    cops, _ = random_pi
    vals = [measure_values(si_rearrangement(c.matr), "pi", ("xi", "footrule")) for c in cops]
    x = np.array([v["xi"] for v in vals])
    y = np.array([v["footrule"] for v in vals])
    assert np.all(reg.margin(x, y) <= 1e-9)


# swapped regions ------------------------------------------------------------
@pytest.mark.parametrize("key", [("xi", "rho"), ("xi", "nu"), ("rho", "nu")])
def test_swapped_region_consistency(key):
    base = cr.get(*key)
    sw = cr.get(key[1], key[0])
    rng = np.random.default_rng(0)
    p = rng.uniform(*base.x_range, 500)
    q = rng.uniform(-1, 1, 500)
    np.testing.assert_array_equal(sw.contains(q, p), base.contains(p, q))
    qs = np.linspace(sw.x_range[0], sw.x_range[1], 41)[1:-1]
    lo, up = sw.lower(qs), sw.upper(qs)
    assert np.all(lo <= up)
    # the inverted boundaries lie on the boundary of the original region
    assert np.all(np.abs(base.margin(lo, qs)) < 1e-9)
    assert np.all(np.abs(base.margin(up, qs)) < 1e-9)
    assert sw.boundary_copula(qs[0]) is None


def test_plots():
    ax = cr.get("xi", "rho").plot()
    assert ax.get_xlabel() == "Chatterjee's $\\xi$"
    assert len(ax.collections) >= 1
    cr.get("nu", "rho").plot(ax=ax, fill=False, mark=False, color="red")
    fig = cr.plot_grid(ncols=2)
    assert len([a for a in fig.axes if a.get_visible()]) == len(ALL)
    with cr.paper_style():
        fig2 = cr.plot_grid([("xi", "rho"), ("xi", "footrule", "si")], ncols=3)
    assert len(fig2.axes) == 2
    plt.close("all")


def test_xi_beta_region_published_bound():
    """|beta|^3 <= 2 xi (arXiv:2606.30033): random checkerboards lie inside,
    and a simple tent-shaped checkerboard attains equality."""
    import copul.regions as regions

    reg = regions.get("xi", "beta")
    cops = BivCheckPi.generate_diverse(500, grid_size=(2, 30), rng=7)
    for c in cops:
        assert reg.contains(c.chatterjees_xi(), c.blomqvists_beta(), tol=1e-9)
    for n in (4, 8):
        P = np.ones((n, n)) / n**2
        h = n // 2
        P[:h, h - 1], P[:h, h] = 2 / n**2, 0.0
        P[h:, h - 1], P[h:, h] = 0.0, 2 / n**2
        c = BivCheckPi(P)
        assert reg.upper(c.chatterjees_xi()) == pytest.approx(c.blomqvists_beta(), abs=1e-12)
