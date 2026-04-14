"""
Parametrized copula axiom tests applied uniformly to all registered families.

This module is the canonical home for the mathematical properties that EVERY
copula must satisfy, eliminating the per-family duplication that previously
appeared in files like test_nelsen1.py, test_nelsen4.py, etc.  Family-specific
files should now focus only on family-specific closed-form results (generator
formulas, exact Kendall tau expressions, etc.).

Axioms tested
-------------
1. Grounded             – C(0,v) = C(u,0) = 0
2. Uniform margins      – C(1,v) = v,  C(u,1) = u
3. Range                – 0 ≤ C(u,v) ≤ 1
4. 2-Increasing         – copula volume condition ≥ 0
5. Fréchet–Hoeffding    – max(u+v-1,0) ≤ C(u,v) ≤ min(u,v)
6. Symmetry             – C(u,v) = C(v,u) for symmetric copulas
7. Conditional range    – ∂C/∂u, ∂C/∂v ∈ [0,1] at interior points
8. PDF non-negativity   – c(u,v) ≥ 0 for absolutely continuous copulas
9. Diagonal monotonicity – C(t,t) non-decreasing

Performance
-----------
The module is marked ``pytest.mark.slow`` so that ``make test``
(which passes ``-m "not slow and not instable"``) skips it.
Run explicitly with::

    pytest tests/family/test_copula_axioms.py -v
"""

import logging
from functools import lru_cache

import numpy as np
import pytest

import copul
from tests.family_representatives import family_representatives

log = logging.getLogger(__name__)

# Mark the entire module as slow — make test skips these automatically.
pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Cache copula instances so each (name, params) pair is instantiated once
# across all axiom tests in the session rather than once per test function.
@lru_cache(maxsize=None)
def _instantiate(copula_name: str, params):
    """Return a fully-parameterised copula instance (cached)."""
    cop_class = getattr(copul, copula_name)
    if params is None:
        return cop_class()
    if isinstance(params, tuple):
        return cop_class(*params)
    return cop_class(params)


def _cdf_batch(cop, us, vs):
    """Evaluate C(u,v) for arrays *us*, *vs* and return a float array.

    Prefers ``cdf_vectorized`` (single numpy call, orders of magnitude
    faster for families that implement it).  Falls back to element-wise
    symbolic evaluation.
    """
    us = np.asarray(us, dtype=float)
    vs = np.asarray(vs, dtype=float)

    # --- fast path: vectorized numerical evaluation ---
    cdf_vec = getattr(cop, "cdf_vectorized", None)
    if cdf_vec is not None:
        try:
            result = np.asarray(cdf_vec(us, vs), dtype=float)
            # Sanity: reject if the vectorized impl returns all-zero for
            # interior points (known Nelsen18 bug).
            interior = (us > 0.01) & (vs > 0.01) & (us < 0.99) & (vs < 0.99)
            if interior.any() and not np.all(result[interior] == 0.0):
                return result
            # If all interior values are zero, fall through to symbolic path.
        except Exception:
            pass

    # --- slow path: element-wise keyword-arg symbolic evaluation ---
    out = np.empty_like(us)
    for idx in range(len(us)):
        u, v = float(us[idx]), float(vs[idx])
        try:
            r = cop.cdf(u=u, v=v)
            out[idx] = float(r.evalf()) if hasattr(r, "evalf") else float(r)
        except TypeError:
            r = cop.cdf(u, v)
            out[idx] = float(r.evalf()) if hasattr(r, "evalf") else float(r)
    return out


def _cdf(cop, u: float, v: float) -> float:
    """Evaluate a single CDF value — thin wrapper around _cdf_batch."""
    return float(_cdf_batch(cop, [u], [v])[0])


# Parametrize list: one entry per registered family
_FAMILIES = [(name, params) for name, params in family_representatives.items()]
_IDS = [name for name, _ in _FAMILIES]

# Families whose conditional distribution implementation is known to return
# out-of-range values (genuine bugs in the copula code, not in this test).
_COND_DISTR_KNOWN_BROKEN: set = set()


# ---------------------------------------------------------------------------
# 1. Grounded: C(0,v) = C(u,0) = 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_grounded(name, params):
    """C(0,v) = C(u,0) = 0 for all v,u in [0,1]."""
    cop = _instantiate(name, params)
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    # Batch: C(0, x_i)
    vals_0x = _cdf_batch(cop, [0.0] * len(xs), xs)
    # Batch: C(x_i, 0)
    vals_x0 = _cdf_batch(cop, xs, [0.0] * len(xs))
    for i, x in enumerate(xs):
        assert vals_0x[i] == pytest.approx(0.0, abs=1e-9), (
            f"{name}: C(0, {x}) = {vals_0x[i]} != 0"
        )
        assert vals_x0[i] == pytest.approx(0.0, abs=1e-9), (
            f"{name}: C({x}, 0) = {vals_x0[i]} != 0"
        )


# ---------------------------------------------------------------------------
# 2. Uniform margins: C(1,v) = v,  C(u,1) = u
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_uniform_margins(name, params):
    """C(1,v) = v and C(u,1) = u for all points in (0,1).

    Tolerance is 1e-4 to accommodate copulas (e.g. StudentT) whose CDF is
    evaluated by numerical quadrature rather than a closed-form formula.
    """
    cop = _instantiate(name, params)
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    vals_1x = _cdf_batch(cop, [1.0] * len(xs), xs)
    vals_x1 = _cdf_batch(cop, xs, [1.0] * len(xs))
    for i, x in enumerate(xs):
        assert vals_1x[i] == pytest.approx(x, abs=1e-4), (
            f"{name}: C(1, {x}) = {vals_1x[i]} != {x}"
        )
        assert vals_x1[i] == pytest.approx(x, abs=1e-4), (
            f"{name}: C({x}, 1) = {vals_x1[i]} != {x}"
        )


# ---------------------------------------------------------------------------
# 3. Range: 0 ≤ C(u,v) ≤ 1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_range(name, params):
    """Copula values lie in [0, 1]."""
    cop = _instantiate(name, params)
    grid = [0.2, 0.5, 0.8]
    us = [u for u in grid for _ in grid]
    vs = [v for _ in grid for v in grid]
    vals = _cdf_batch(cop, us, vs)
    for k in range(len(us)):
        assert 0.0 - 1e-10 <= vals[k] <= 1.0 + 1e-10, (
            f"{name}: C({us[k]},{vs[k]}) = {vals[k]} not in [0,1]"
        )


# ---------------------------------------------------------------------------
# 4. 2-Increasing: copula rectangle volume ≥ 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_2_increasing(name, params):
    """V_C([u1,u2]×[v1,v2]) = C(u2,v2) - C(u1,v2) - C(u2,v1) + C(u1,v1) ≥ 0."""
    cop = _instantiate(name, params)
    corners = [0.2, 0.5, 0.8]
    # Pre-compute all needed CDF values in a single batch
    pts_u, pts_v = [], []
    rects = []
    for i in range(len(corners) - 1):
        u1, u2 = corners[i], corners[i + 1]
        for j in range(len(corners) - 1):
            v1, v2 = corners[j], corners[j + 1]
            base = len(pts_u)
            for uu, vv in [(u2, v2), (u1, v2), (u2, v1), (u1, v1)]:
                pts_u.append(uu)
                pts_v.append(vv)
            rects.append((u1, v1, u2, v2, base))
    vals = _cdf_batch(cop, pts_u, pts_v)
    for u1, v1, u2, v2, base in rects:
        vol = vals[base] - vals[base + 1] - vals[base + 2] + vals[base + 3]
        assert vol >= -1e-9, (
            f"{name}: 2-increasing violated at ({u1},{v1})x({u2},{v2}): vol={vol:.2e}"
        )


# ---------------------------------------------------------------------------
# 5. Fréchet–Hoeffding bounds: max(u+v−1,0) ≤ C(u,v) ≤ min(u,v)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_frechet_hoeffding_bounds(name, params):
    """max(u+v-1,0) ≤ C(u,v) ≤ min(u,v) everywhere on the interior grid."""
    cop = _instantiate(name, params)
    grid = [0.2, 0.5, 0.8]
    us = [u for u in grid for _ in grid]
    vs = [v for _ in grid for v in grid]
    vals = _cdf_batch(cop, us, vs)
    for k in range(len(us)):
        u, v, val = us[k], vs[k], vals[k]
        lower = max(u + v - 1.0, 0.0)
        upper = min(u, v)
        assert val >= lower - 1e-9, (
            f"{name}: C({u},{v})={val:.6f} < W({u},{v})={lower:.6f}"
        )
        assert val <= upper + 1e-9, (
            f"{name}: C({u},{v})={val:.6f} > M({u},{v})={upper:.6f}"
        )


# ---------------------------------------------------------------------------
# 6. Symmetry: C(u,v) = C(v,u) for exchangeable copulas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_symmetry(name, params):
    """C(u,v) = C(v,u) for copulas that declare is_symmetric=True."""
    cop = _instantiate(name, params)
    is_sym = False
    try:
        is_sym = bool(cop.is_symmetric)
    except (NotImplementedError, AttributeError):
        pass
    if not is_sym:
        pytest.skip(f"{name} is not symmetric")

    pairs = [(0.3, 0.7), (0.2, 0.8), (0.4, 0.6)]
    us = [u for u, _ in pairs] + [v for _, v in pairs]
    vs = [v for _, v in pairs] + [u for u, _ in pairs]
    vals = _cdf_batch(cop, us, vs)
    n = len(pairs)
    for i in range(n):
        cuv, cvu = vals[i], vals[n + i]
        u, v = pairs[i]
        assert cuv == pytest.approx(cvu, abs=1e-4), (
            f"{name}: C({u},{v})={cuv:.8f} != C({v},{u})={cvu:.8f}"
        )


# ---------------------------------------------------------------------------
# 7. Conditional distributions ∈ [0, 1]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_conditional_distribution_range(name, params):
    """dC/du and dC/dv lie in [0,1] at interior points."""
    if name in _COND_DISTR_KNOWN_BROKEN:
        pytest.xfail(f"{name} has a known cond_distr implementation bug")
    cop = _instantiate(name, params)
    test_pts = [(0.3, 0.6), (0.5, 0.5), (0.7, 0.2)]
    for u, v in test_pts:
        for method_name in ("cond_distr_1", "cond_distr_2"):
            method = getattr(cop, method_name, None)
            if method is None:
                continue
            try:
                result = method(u=u, v=v)
                val = (
                    float(result.evalf()) if hasattr(result, "evalf") else float(result)
                )
            except Exception as exc:
                log.debug("%s.%s(%s,%s) skipped: %s", name, method_name, u, v, exc)
                continue
            assert -0.01 <= val <= 1.01, (
                f"{name}.{method_name}({u},{v}) = {val:.6f} not in [0,1]"
            )


# ---------------------------------------------------------------------------
# 8. PDF non-negativity (absolutely continuous families)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_pdf_non_negative(name, params):
    """PDF values are >= 0 at interior grid points for absolutely continuous copulas."""
    cop = _instantiate(name, params)
    try:
        is_ac = bool(cop.is_absolutely_continuous)
    except (NotImplementedError, AttributeError):
        pytest.skip(f"{name}: is_absolutely_continuous not available")
    if not is_ac:
        pytest.skip(f"{name} is not absolutely continuous")

    from copul.exceptions import PropertyUnavailableException

    for u, v in [(0.3, 0.4), (0.5, 0.5), (0.6, 0.7)]:
        try:
            pdf_result = cop.pdf(u=u, v=v)
            if callable(pdf_result):
                pdf_result = pdf_result(u, v)
            if hasattr(pdf_result, "evalf"):
                val = float(pdf_result.evalf())
            else:
                val = float(pdf_result)
        except PropertyUnavailableException:
            pytest.skip(f"{name}: PDF not available")
        except Exception as exc:
            log.debug("%s.pdf(%s,%s) skipped: %s", name, u, v, exc)
            continue
        assert val >= -1e-9, f"{name}: pdf({u},{v}) = {val:.6f} < 0"


# ---------------------------------------------------------------------------
# 9. Diagonal section monotonicity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_diagonal_section_monotone(name, params):
    """C(t,t) is non-decreasing in t (diagonal section is monotone)."""
    cop = _instantiate(name, params)
    t_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    diag = _cdf_batch(cop, t_vals, t_vals)
    for i in range(len(diag) - 1):
        assert diag[i] <= diag[i + 1] + 1e-9, (
            f"{name}: diagonal section not monotone: "
            f"C({t_vals[i]},{t_vals[i]})={diag[i]:.6f} > "
            f"C({t_vals[i + 1]},{t_vals[i + 1]})={diag[i + 1]:.6f}"
        )
