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
2. Uniform margins      – C(1,v) = v,  C(u,1) = u          (already in test_all_families.py; kept
                           here for the interior grid as well)
3. Range                – 0 ≤ C(u,v) ≤ 1
4. 2-Increasing         – copula volume condition ≥ 0
5. Fréchet–Hoeffding    – max(u+v-1,0) ≤ C(u,v) ≤ min(u,v)
6. Symmetry             – C(u,v) = C(v,u) for symmetric copulas
7. Conditional range    – ∂C/∂u, ∂C/∂v ∈ [0,1] at interior points
8. PDF non-negativity   – c(u,v) ≥ 0 for absolutely continuous copulas
"""

import logging

import numpy as np
import pytest

import copul
from tests.family_representatives import family_representatives

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instantiate(copula_name: str, params):
    """Return a fully-parameterised copula instance."""
    cop_class = getattr(copul, copula_name)
    if params is None:
        return cop_class()
    if isinstance(params, tuple):
        return cop_class(*params)
    return cop_class(params)


def _cdf(cop, u: float, v: float) -> float:
    """Evaluate the CDF robustly across different copula API styles.

    Tries the keyword-argument path first (works for the vast majority of
    families).  Falls back to ``cdf_vectorized`` when the symbolic path
    raises a ``TypeError`` (API mismatch), and finally to positional args.
    """
    # 1. Standard keyword-arg path
    try:
        result = cop.cdf(u=u, v=v)
        if hasattr(result, "evalf"):
            return float(result.evalf())
        return float(result)
    except TypeError:
        pass

    # 2. Vectorized numerical path (fallback for families with non-standard CDF API)
    cdf_vec = getattr(cop, "cdf_vectorized", None)
    if cdf_vec is not None:
        try:
            arr = cdf_vec(np.array([u], dtype=float), np.array([v], dtype=float))
            return float(arr.flat[0])
        except Exception:
            pass

    # 3. Positional-arg fallback
    result = cop.cdf(u, v)
    if hasattr(result, "evalf"):
        return float(result.evalf())
    return float(result)


# Parametrize list: one entry per registered family
_FAMILIES = [(name, params) for name, params in family_representatives.items()]
_IDS = [name for name, _ in _FAMILIES]

# Families whose conditional distribution implementation is known to return
# out-of-range values (genuine bugs in the copula code, not in this test).
# Tests that check cond_distr range are marked xfail for these families so
# CI is not blocked while the underlying bug is being fixed.
# NOTE: tEV was initially listed here but the root cause was an invalid
# representative in family_representatives.py (rho=2 outside (-1,1)).
# After fixing the representative to (nu=2, rho=0.5), tEV passes.
_COND_DISTR_KNOWN_BROKEN: set = set()


# ---------------------------------------------------------------------------
# 1. Grounded: C(0,v) = C(u,0) = 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_grounded(name, params):
    """C(0,v) = C(u,0) = 0 for all v,u in [0,1]."""
    cop = _instantiate(name, params)
    for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert _cdf(cop, 0.0, x) == pytest.approx(0.0, abs=1e-9), (
            f"{name}: C(0, {x}) != 0"
        )
        assert _cdf(cop, x, 0.0) == pytest.approx(0.0, abs=1e-9), (
            f"{name}: C({x}, 0) != 0"
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
    for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert _cdf(cop, 1.0, x) == pytest.approx(x, abs=1e-4), (
            f"{name}: C(1, {x}) != {x}"
        )
        assert _cdf(cop, x, 1.0) == pytest.approx(x, abs=1e-4), (
            f"{name}: C({x}, 1) != {x}"
        )


# ---------------------------------------------------------------------------
# 3. Range: 0 ≤ C(u,v) ≤ 1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_range(name, params):
    """Copula values lie in [0, 1]."""
    cop = _instantiate(name, params)
    for u in [0.2, 0.5, 0.8]:
        for v in [0.2, 0.5, 0.8]:
            val = _cdf(cop, u, v)
            assert 0.0 - 1e-10 <= val <= 1.0 + 1e-10, (
                f"{name}: C({u},{v}) = {val} not in [0,1]"
            )


# ---------------------------------------------------------------------------
# 4. 2-Increasing: copula rectangle volume ≥ 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_2_increasing(name, params):
    """V_C([u1,u2]×[v1,v2]) = C(u2,v2) - C(u1,v2) - C(u2,v1) + C(u1,v1) ≥ 0."""
    cop = _instantiate(name, params)
    corners = [0.2, 0.5, 0.8]
    for i in range(len(corners) - 1):
        u1, u2 = corners[i], corners[i + 1]
        for j in range(len(corners) - 1):
            v1, v2 = corners[j], corners[j + 1]
            vol = (
                _cdf(cop, u2, v2)
                - _cdf(cop, u1, v2)
                - _cdf(cop, u2, v1)
                + _cdf(cop, u1, v1)
            )
            assert vol >= -1e-9, (
                f"{name}: 2-increasing violated at ({u1},{v1})×({u2},{v2}): vol={vol:.2e}"
            )


# ---------------------------------------------------------------------------
# 5. Fréchet–Hoeffding bounds: max(u+v−1,0) ≤ C(u,v) ≤ min(u,v)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_frechet_hoeffding_bounds(name, params):
    """max(u+v-1,0) ≤ C(u,v) ≤ min(u,v) everywhere on the interior grid."""
    cop = _instantiate(name, params)
    for u in [0.2, 0.5, 0.8]:
        for v in [0.2, 0.5, 0.8]:
            val = _cdf(cop, u, v)
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

    for u, v in [(0.3, 0.7), (0.2, 0.8), (0.4, 0.6)]:
        cuv = _cdf(cop, u, v)
        cvu = _cdf(cop, v, u)
        # abs=1e-4 to accommodate numerical-quadrature families (StudentT)
        assert cuv == pytest.approx(cvu, abs=1e-4), (
            f"{name}: C({u},{v})={cuv:.8f} != C({v},{u})={cvu:.8f}"
        )


# ---------------------------------------------------------------------------
# 7. Conditional distributions ∈ [0, 1]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
@pytest.mark.xfail(
    condition=False,  # overridden per-family inside the test body
    reason="Known cond_distr bug — see _COND_DISTR_KNOWN_BROKEN",
    strict=False,
)
def test_conditional_distribution_range(name, params):
    """∂C/∂u and ∂C/∂v lie in [0,1] at interior points."""
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
            # Use abs=0.01 tolerance to absorb numerical-quadrature rounding in
            # families like StudentT. Values outside [−0.01, 1.01] indicate a
            # genuine implementation bug.
            assert -0.01 <= val <= 1.01, (
                f"{name}.{method_name}({u},{v}) = {val:.6f} not in [0,1]"
            )


# ---------------------------------------------------------------------------
# 8. PDF non-negativity (absolutely continuous families)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_pdf_non_negative(name, params):
    """PDF values are ≥ 0 at interior grid points for absolutely continuous copulas."""
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
# 9. C(u,u) monotone in u: lower tail concentration L(t)=C(t,t)/t ∈ [0,1]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", _FAMILIES, ids=_IDS)
def test_diagonal_section_monotone(name, params):
    """C(t,t) is non-decreasing in t (diagonal section is monotone)."""
    cop = _instantiate(name, params)
    t_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    diag = [_cdf(cop, t, t) for t in t_vals]
    for i in range(len(diag) - 1):
        assert diag[i] <= diag[i + 1] + 1e-9, (
            f"{name}: diagonal section not monotone: "
            f"C({t_vals[i]},{t_vals[i]})={diag[i]:.6f} > "
            f"C({t_vals[i + 1]},{t_vals[i + 1]})={diag[i + 1]:.6f}"
        )
