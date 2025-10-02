# tests/test_core_copula.py
import pytest
import sympy

from copul import Clayton, XiRhoBoundaryCopula
from copul.family.core.core_copula import CoreCopula
from copul.family.other.xi_psi_lower_jensen_bound import XiPsiLowerJensenBound


# ---------------------------------------------------------------------
# Helper subclasses for testing
# ---------------------------------------------------------------------
class IndepCopula(CoreCopula):
    """Independence copula: C(u) = ∏ u_i (no free params)."""

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    def __init__(self, dimension: int):
        super().__init__(dimension)
        # C(u) = product(u_i)
        expr = sympy.Integer(1)
        for ui in self.u_symbols:
            expr *= ui
        self._cdf_expr = expr


class PowerCopula(CoreCopula):
    """
    Simple parametric CDF in 2D:
      C(u1, u2) = u1**a * u2
    with a ∈ [0, 2]. Exposes CoreCopula's parameter & interval machinery.
    """

    # class-level defaults used by CoreCopula's __init__
    a = sympy.Symbol("a")  # name used as attribute when not set in kwargs/args
    params = [a]
    intervals = {"a": sympy.Interval(0, 2)}
    _free_symbols = {"a": a}  # for _cdf_expr auto-substitution

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        # Not symmetric unless a == 1; declare False for the class
        return False

    def __init__(self, a=None):
        super().__init__(2, **({"a": a} if a is not None else {}))
        u1, u2 = self.u_symbols
        # Template expression with symbolic 'a'; Core will substitute if self.a exists
        self._cdf_expr = (u1 ** self._free_symbols["a"]) * u2


# ---------------------------------------------------------------------
# Basic dunder & construction
# ---------------------------------------------------------------------
def test_str_returns_class_name():
    c = IndepCopula(2)
    assert str(c) == "IndepCopula"


def test_indep_copula_cdf_basic_eval():
    c = IndepCopula(2)
    # C(0.3, 0.7) = 0.21
    assert pytest.approx(c.cdf(0.3, 0.7)) == 0.21
    # Via array
    assert pytest.approx(c.cdf([0.3, 0.7])) == 0.21
    # Via named variables (standard u1, u2)
    assert pytest.approx(c.cdf(u1=0.3, u2=0.7)) == 0.21


def test_pdf_indep_is_one_everywhere():
    c = IndepCopula(2)
    pdf = c.pdf()
    # PDF is constant 1; with no remaining variables, just call with no args
    assert pytest.approx(pdf) == 1.0
    # kwargs are fine too (they'll be ignored since pdf has no free symbols)
    assert pytest.approx(c.pdf(u1=0.1, u2=0.2)) == 1.0


def test_conditional_distribution_indep():
    c = IndepCopula(2)
    # ∂/∂u1 (u1 u2) = u2  ⇒ conditioned on U1, the value at (u1,u2) is u2
    # Provide remaining variable via args after partial substitution
    assert pytest.approx(c.cond_distr(1, 0.7, u1=0.25)) == 0.7
    # Similarly for i=2: ∂/∂u2 = u1
    assert pytest.approx(c.cond_distr(2, 0.25, u2=0.7)) == 0.25


# ---------------------------------------------------------------------
# Parameter & interval plumbing
# ---------------------------------------------------------------------
def test_parameters_and_intervals_are_trimmed_on_init_and_call():
    # Initially, 'a' is free
    c = PowerCopula()  # params=['a'], intervals={'a':[0,2]}
    assert not c.is_fully_specified()
    assert set(map(str, c.parameters.keys())) == {"a"}

    # Fix 'a' via calling the instance: returns a new instance with params removed
    c2 = c(a=2.0)
    assert c2.is_fully_specified()
    assert c2.parameters == {}  # no remaining free params

    # The original remains with a free param
    assert not c.is_fully_specified()
    assert set(map(str, c.parameters.keys())) == {"a"}


def test_slice_interval_restricts_bounds_and_closes_when_set():
    c = PowerCopula()
    # initial interval [0,2]
    I0 = c.parameters["a"]
    assert float(I0.inf) == 0.0 and float(I0.sup) == 2.0
    # use truthiness instead of identity to be robust across SymPy versions
    assert (not I0.left_open) and (not I0.right_open)

    # Slice to [-0.5, 0.8] (explicit bounds should be closed)
    c.slice_interval("a", -0.5, 0.8)
    II = c.parameters["a"]
    assert float(II.inf) == -0.5 and float(II.sup) == 0.8
    assert (not II.left_open) and (not II.right_open)


def test__cdf_expr_substitutes_free_symbols_from_attributes():
    # Set a concrete 'a' so _cdf_expr uses it
    c = PowerCopula(a=2.0)
    u1, u2 = c.u_symbols
    # Evaluate: C(u1,u2)=u1**2 * u2; at (0.5, 0.8) -> 0.25 * 0.8 = 0.2
    assert pytest.approx(c.cdf(0.5, 0.8)) == 0.2


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------
def test_survival_copula_of_independence():
    c = IndepCopula(2)
    s = c.survival_copula()
    # For independence, \widehat C(u,v) = (1-u)(1-v)
    val = s.cdf(0.3, 0.7)
    assert pytest.approx(val) == (1 - 0.3) * (1 - 0.7)


def test_vertical_reflection_margin2_independence():
    c = IndepCopula(2)
    vref = c.vertical_reflection(margin=2)
    # For C(u,v)=u v, C^vee(u,v)= v - C(u,1-v) = v - u(1-v) = v - u + u v
    u, v = 0.25, 0.6
    expected = v - u + u * v
    assert pytest.approx(vref.cdf(u, v)) == expected


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def test_validate_copula_requires_fully_specified_params():
    c = PowerCopula()  # 'a' free
    with pytest.raises(ValueError, match="free parameters"):
        c.validate_copula()


def test_validate_copula_passes_for_independence_small_grid():
    c = IndepCopula(2)
    ok, details = c.validate_copula(m=6, return_details=True)
    assert ok  # np.bool_ or bool both pass

    assert details["dim"] == 2
    assert details["grid_size_per_axis"] == 7
    assert details["bounds_ok"]
    assert details["grounded_ok"]
    assert details["increasing_ok"]


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_validate_copula_fails_for_non_copula(mu):
    copula = XiPsiLowerJensenBound(mu)
    ok = copula.validate_copula(m=6)
    assert not ok  # np.bool_ or bool both pass


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_validate_copula_passes_for_copula(mu):
    copula = XiRhoBoundaryCopula(mu)
    ok = copula.validate_copula(m=6)
    assert ok


def test_clayton_copula_validation():
    Clayton(2).validate_copula()
