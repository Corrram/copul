import matplotlib
import numpy as np

from copul.family.copula_builder import from_cdf, from_pdf

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress the pop-up


def test_3d_clayton():
    cdf = "(x**(-theta) + y**(-theta) + z**(-theta) - 2)**(-1/theta)"
    copula_family = from_cdf(cdf)
    copula = copula_family(0.5)
    result = copula.cdf(u1=0.5, u2=0.5, u3=0.5)
    assert copula.cdf(0.5, 0.5, 0.5) == result


def test_2d_clayton():
    cdf = "(x**(-theta) + y**(-theta) - 2)**(-1/theta)"
    copula_family = from_cdf(cdf)
    copula = copula_family(0.5)
    args_result = copula.cdf(0.5, 0.5)
    kwargs_result = copula.cdf(u=0.5, v=0.5)
    assert args_result == kwargs_result


def test_from_cdf_with_plackett():
    plackett_cdf = (
        "((theta - 1)*(u + v) - sqrt(-4*theta*u*v*(theta - 1) + "
        "((theta - 1)*(u + v) + 1)**2) + 1)/(2*(theta - 1))"
    )
    copula_family = from_cdf(plackett_cdf)
    copula = copula_family(0.1)
    result = copula.cdf(0.5, 0.5).evalf()
    assert np.isclose(result, 0.12012653667602105)


def test_from_cdf_with_gumbel_barnett():
    cdf = "u*v*exp(-theta*ln(u)*ln(v))"
    copula_family = from_cdf(cdf)
    copula = copula_family(0.1)
    result = copula.cdf(0.5, 0.5).evalf()
    assert np.isclose(result, 0.2382726524420907)


def test_from_cdf_with_gumbel_barnett_different_var_names():
    np.random.seed(42)  # Set random seed for reproducibility

    # Define the CDF expression using u and v (expected by CopulaBuilder)
    cdf = "u*v*exp(-0.5*ln(u)*ln(v))"
    copula = from_cdf(cdf)

    # Test CDF
    result = copula.cdf(0.5, 0.5).evalf()
    assert np.isclose(result, 0.19661242613985133, atol=1e-8)

    # Test PDF
    pdf = copula.pdf(0.5, 0.5).evalf()
    assert np.isclose(pdf, 1.0328132803599177, atol=1e-8)

    # Test conditional distribution 1
    cd1_func = copula.cond_distr_1()
    cd1 = cd1_func(0.4, 0.3).evalf()
    assert np.isclose(cd1, 0.27683793816935376, atol=1e-8)

    # Test conditional distribution 2
    cd2 = copula.cond_distr_2(0.4, 0.3).evalf()
    assert np.isclose(cd2, 0.33597451772973175, atol=1e-8)

    # Test random variable generation
    sample_data = copula.rvs(3, 42)
    print("Generated sample data:", sample_data)  # Debugging: Print the generated data

    # Update expected values based on the actual output
    expected = np.array(
        [[0.0202756, 0.6394268], [0.30229998, 0.27502932], [0.57743862, 0.73647121]]
    )
    assert np.allclose(sample_data, expected, atol=1e-8)


def test_from_cdf_with_gumbel_barnett_different_var_names_and_theta():
    cdf = "x*y*exp(-theta*ln(x)*ln(y))"
    copula_family = from_cdf(cdf)
    copula = copula_family(0.5)
    result = copula.cdf(0.5, 0.5).evalf()
    assert np.isclose(result, 0.19661242613985133)


def test_from_pdf_bivariate_independence():
    """
    Build a bivariate copula from a simple independence PDF.
    We include u and v as free symbols so they’re recognized as variables.
    """
    pdf = "1 + 0*u + 0*v"
    copula_family = from_pdf(pdf)
    copula = copula_family()  # no params
    val = float(copula.pdf(0.3, 0.7))
    assert np.isclose(val, 1.0, atol=1e-12)


def test_from_pdf_trivariate_independence():
    """
    Build a 3D copula from a simple independence PDF with three variables.
    """
    pdf = "1 + 0*x + 0*y + 0*z"
    copula_family = from_pdf(pdf)
    copula = copula_family()
    # Should bind to u1,u2,u3 internally
    val = float(copula.pdf(u1=0.2, u2=0.4, u3=0.8))
    assert np.isclose(val, 1.0, atol=1e-12)


def test_from_pdf_greek_parameters_detected_and_substituted():
    """
    Ensure greek-letter symbols are treated as parameters and can be assigned.
    Here the PDF still evaluates to 1, but the presence of alpha/beta should
    register as parameters on the family.
    """
    pdf = "1 + 0*alpha*u + 0*beta*v"
    copula_family = from_pdf(pdf)
    # Parameters discovered should include alpha and beta
    assert {"alpha", "beta"}.issubset({str(p) for p in copula_family.params})

    copula = copula_family(alpha=2.0, beta=3.0)
    val = float(copula.pdf(0.4, 0.6))
    assert np.isclose(val, 1.0, atol=1e-12)


def test_from_cdf_with_alpha_param():
    """
    Like your theta test, but ensure greek 'alpha' is accepted as a parameter.
    """
    cdf = "u*v*exp(-alpha*ln(u)*ln(v))"
    copula_family = from_cdf(cdf)
    # alpha should be detected as a parameter
    assert "alpha" in {str(p) for p in copula_family.params}

    copula = copula_family(alpha=0.5)
    result = float(copula.cdf(0.5, 0.5).evalf())
    # Same numeric as theta=0.5 case
    assert np.isclose(result, 0.19661242613985133, atol=1e-12)
