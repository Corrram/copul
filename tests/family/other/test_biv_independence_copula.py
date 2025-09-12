import numpy as np
import pytest
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.other.frechet import Frechet
from copul.family.other.biv_independence_copula import BivIndependenceCopula


@pytest.fixture
def independence_copula():
    """Create a basic IndependenceCopula instance for testing."""
    return BivIndependenceCopula()


def test_inheritance():
    """Test that IndependenceCopula inherits from both Frechet and ArchimedeanCopula."""
    copula = BivIndependenceCopula()
    assert isinstance(copula, Frechet)
    assert isinstance(copula, BivArchimedeanCopula)


def test_alpha_beta_properties(independence_copula):
    """Test that alpha and beta are fixed at 0 for the independence copula."""
    assert independence_copula.alpha == 0
    assert independence_copula.beta == 0
    assert independence_copula._alpha == 0
    assert independence_copula._beta == 0


def test_pickands_property(independence_copula):
    """Test that the Pickands dependence function is constant 1."""
    pickands = independence_copula.pickands
    assert isinstance(pickands, sympy.Expr)
    # Evaluate the expression at different points
    t_vals = np.linspace(0, 1, 5)  # Test points in [0, 1]
    for t in t_vals:
        # For independence copula, Pickands function should be 1 everywhere
        assert float(pickands.subs(independence_copula.t, t)) == 1


def test_generator(independence_copula):
    """Test that the generator is -log(t)."""
    generator = independence_copula.generator
    t_vals = np.linspace(0.1, 0.9, 5)  # Avoid t=0 which would make log(t) undefined

    for t in t_vals:
        expected = -np.log(t)
        result = float(generator.subs(independence_copula.t, t))
        assert abs(result - expected) < 1e-10


def test_inv_generator(independence_copula):
    """Test that the inverse generator is exp(-y)."""
    inv_generator = independence_copula.inv_generator
    y_vals = np.linspace(0, 5, 5)

    for y in y_vals:
        expected = np.exp(-y)
        result = float(inv_generator(y))
        assert abs(result - expected) < 1e-10


def test_cdf(independence_copula):
    """Test that the CDF is C(u,v) = u*v."""
    # Sample points in the unit square
    test_points = [
        (0.1, 0.2),
        (0.3, 0.7),
        (0.5, 0.5),
        (0.8, 0.9),
        (0, 0),
        (1, 1),
        (0, 1),
        (1, 0),
    ]

    for u, v in test_points:
        cdf_val = float(independence_copula.cdf(u=u, v=v))
        expected = u * v
        assert abs(cdf_val - expected) < 1e-10


def test_pdf(independence_copula):
    """Test that the PDF is constant 1 on the unit square."""
    # Sample points in the interior of the unit square
    test_points = [(0.1, 0.2), (0.3, 0.7), (0.5, 0.5), (0.8, 0.9)]

    for u, v in test_points:
        pdf_val = independence_copula.pdf(u=u, v=v)
        pdf_float = float(pdf_val)
        expected = 1.0  # Independence copula has uniform density
        assert abs(pdf_float - expected) < 1e-10


# def test_pdf_with_positional_arguments(independence_copula):
#     """Test that the PDF is constant 1 on the unit square."""
#     # Sample points in the interior of the unit square
#     test_points = [(0.1, 0.2), (0.3, 0.7), (0.5, 0.5), (0.8, 0.9)]

#     for u, v in test_points:
#         pdf_val = independence_copula.pdf(u, v)
#         pdf_float = float(pdf_val)
#         expected = 1.0  # Independence copula has uniform density
#         assert abs(pdf_float - expected) < 1e-10


def test_conditional_distributions(independence_copula):
    """Test that conditional distributions are as expected for independence."""
    # For independence copula:
    # cond_distr_1(u|v) = v (the derivative of uv with respect to u is v)
    # cond_distr_2(v|u) = u (the derivative of uv with respect to v is u)

    test_points = [(0.3, 0.7), (0.5, 0.5), (0.8, 0.9)]

    for u, v in test_points:
        cond1 = float(independence_copula.cond_distr_1(u=u, v=v))
        assert abs(cond1 - v) < 1e-10, f"cond_distr_1({u},{v}) = {cond1}, expected {v}"

        cond2 = float(independence_copula.cond_distr_2(u=u, v=v))
        assert abs(cond2 - u) < 1e-10, f"cond_distr_2({u},{v}) = {cond2}, expected {u}"


def test_tail_dependence(independence_copula):
    """Test that both tail dependence coefficients are 0 for the independence copula."""
    # Independence copula has no tail dependence
    lambda_L = independence_copula.lambda_L()
    assert lambda_L == 0, "Lower tail dependence should be 0"

    lambda_U = independence_copula.lambda_U()
    assert lambda_U == 0, "Upper tail dependence should be 0"


def test_boundary_conditions(independence_copula):
    """Test that the independence copula satisfies the boundary conditions."""
    u_vals = np.linspace(0.1, 0.9, 5)

    # C(u,0) = 0
    for u in u_vals:
        cdf_val = float(independence_copula.cdf(u=u, v=0))
        assert abs(cdf_val) < 1e-10

    # C(0,v) = 0
    for v in u_vals:
        cdf_val = float(independence_copula.cdf(u=0, v=v))
        assert abs(cdf_val) < 1e-10

    # C(u,1) = u
    for u in u_vals:
        cdf_val = float(independence_copula.cdf(u=u, v=1))
        assert abs(cdf_val - u) < 1e-10

    # C(1,v) = v
    for v in u_vals:
        cdf_val = float(independence_copula.cdf(u=1, v=v))
        assert abs(cdf_val - v) < 1e-10


def test_integration():
    """Test that the PDF integrates to 1 over the unit square."""
    # For the independence copula, this is trivial since PDF=1 everywhere
    # so the integral is just the area of the unit square, which is 1

    # But we can verify this numerically as well
    from scipy import integrate

    copula = BivIndependenceCopula()

    def pdf_func(u, v):
        return float(copula.pdf(u=u, v=v))

    result, _ = integrate.nquad(
        lambda u, v: pdf_func(u, v),
        [[0, 1], [0, 1]],
        opts={"epsabs": 1e-3, "epsrel": 1e-3},
    )

    assert abs(result - 1.0) < 0.01


def test_params():
    """Test that the copula has the expected parameters."""
    # IndependenceCopula doesn't have get_params() but we can check attributes
    copula = BivIndependenceCopula()

    # Check that the key properties are as expected
    assert copula.alpha == 0
    assert copula.beta == 0


def test_archimedean_properties():
    """Test Archimedean copula specific properties."""
    copula = BivIndependenceCopula()

    # Test that the generator is -log(t)
    t_val = 0.5
    gen_val = float(copula.generator.subs(copula.t, t_val))
    expected = -np.log(t_val)
    assert abs(gen_val - expected) < 1e-10

    # Test that the inverse generator is exp(-y)
    y_val = 1.0
    inv_gen_val = float(copula.inv_generator(y_val))
    expected = np.exp(-y_val)
    assert abs(inv_gen_val - expected) < 1e-10
