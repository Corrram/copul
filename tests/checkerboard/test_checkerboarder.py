import numpy as np
import pandas as pd

import copul
from copul.checkerboard.checkerboarder import Checkerboarder
from tests.family_representatives import family_representatives


def test_squared_checkerboard():
    clayton = copul.Families.CLAYTON.cls(2)
    checkerboarder = copul.Checkerboarder(3)
    ccop = checkerboarder.get_checkerboard_copula(clayton)
    assert ccop.matr.shape == (3, 3)
    assert ccop.matr.sum() == 1.0


def test_rectangular_checkerboard():
    clayton = copul.Families.CLAYTON.cls(2)
    checkerboarder = copul.Checkerboarder([3, 10])
    ccop = checkerboarder.get_checkerboard_copula(clayton)
    assert ccop.matr.shape == (3, 10)
    matr_sum = ccop.matr.sum()
    assert np.isclose(matr_sum, 1.0)


def test_rectangular_checkerboard_with_n16():
    n16 = copul.Families.NELSEN16.cls(2)
    checkerboarder = copul.Checkerboarder([3, 10])
    ccop = checkerboarder.get_checkerboard_copula(n16)
    assert ccop.matr.shape == (3, 10)
    matr_sum = ccop.matr.sum()
    assert np.isclose(matr_sum, 1.0)


def test_xi_computation():
    np.random.seed(121)
    copula = copul.Families.NELSEN7.cls(0.5)
    checkerboarder = copul.Checkerboarder(10)
    ccop = checkerboarder.get_checkerboard_copula(copula)
    orig_xi = copula.chatterjees_xi()
    xi = ccop.chatterjees_xi()
    assert 0.5 * orig_xi <= xi <= orig_xi


def test_default_initialization():
    """Test the default initialization of Checkerboarder."""
    checkerboarder = copul.Checkerboarder()
    assert checkerboarder.n == [20, 20]
    assert checkerboarder.d == 2


def test_custom_dimensions():
    """Test Checkerboarder with custom dimensions."""
    checkerboarder = copul.Checkerboarder(5, dim=3)
    assert checkerboarder.n == [5, 5, 5]
    assert checkerboarder.d == 3


def test_mixed_dimensions():
    """Test Checkerboarder with mixed dimensions."""
    checkerboarder = copul.Checkerboarder([3, 4, 5])
    assert checkerboarder.n == [3, 4, 5]
    assert checkerboarder.d == 3


def test_different_copula_families():
    """Test Checkerboarder with different copula families."""
    # Test with available copula families in your implementation
    # Fixed: using the families that actually exist in your package
    for family_param in [(copul.Families.CLAYTON, 2), (copul.Families.NELSEN7, 0.5)]:
        family, param = family_param
        copula = family.cls(param)
        checkerboarder = copul.Checkerboarder(5)
        ccop = checkerboarder.get_checkerboard_copula(copula)

        # Check properties of the checkerboard copula
        assert ccop.matr.shape == (5, 5)
        assert np.isclose(ccop.matr.sum(), 1.0)


def test_from_data_bivariate():
    """Test the from_data method with bivariate data."""
    # Generate synthetic data with a known dependence structure
    np.random.seed(42)
    n_samples = 1000

    # Generate correlated normal data
    rho = 0.7
    cov_matrix = np.array([[1, rho], [rho, 1]])
    data = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=n_samples)
    df = pd.DataFrame(data, columns=["X", "Y"])

    # Create checkerboard from data
    checkerboarder = copul.Checkerboarder(10)
    ccop = checkerboarder.from_data(df)

    # Basic checks
    assert ccop.matr.shape == (10, 10)
    assert np.isclose(ccop.matr.sum(), 1.0)


def test_from_data_numpy_array():
    """Test the from_data method with a numpy array."""
    np.random.seed(42)
    data = np.random.rand(100, 2)  # Uniform random data

    checkerboarder = copul.Checkerboarder(5)
    ccop = checkerboarder.from_data(data)

    assert ccop.matr.shape == (5, 5)
    assert np.isclose(ccop.matr.sum(), 1.0)


def test_from_data_list():
    """Test the from_data method with a list."""
    np.random.seed(42)
    data = np.random.rand(100, 2).tolist()  # Uniform random data as list

    checkerboarder = copul.Checkerboarder(5)
    ccop = checkerboarder.from_data(data)

    assert ccop.matr.shape == (5, 5)
    assert np.isclose(ccop.matr.sum(), 1.0)


def test_direct_from_data():
    """Test using from_data as a module-level function."""
    np.random.seed(42)
    data = np.random.rand(100, 2)  # Uniform random data

    # Use the function directly from the module where it's defined
    from copul.checkerboard.checkerboarder import from_data

    ccop = from_data(data, checkerboard_size=5)

    assert ccop.matr.shape == (5, 5)
    assert np.isclose(ccop.matr.sum(), 1.0)


def test_from_data_trivariate_uniform_shape_and_sum():
    """3D: uniform data -> histogramdd path, correct shape & normalization."""
    np.random.seed(123)
    data = np.random.rand(2000, 3)  # i.i.d. U(0,1)
    checkerboarder = Checkerboarder(6, dim=3)  # 6x6x6
    ccop = checkerboarder.from_data(data)

    assert ccop.matr.shape == (6, 6, 6)
    assert np.isclose(ccop.matr.sum(), 1.0)
    assert np.all(ccop.matr >= 0.0)


def test_from_data_trivariate_rectangular_bins_and_boundary():
    """3D: rectangular bins & boundary ranks (==1) are handled (clipped) correctly."""
    # Construct data that guarantees some ranks equal 1.0 in each column
    np.random.seed(321)
    n = 999  # odd size to avoid ties with random seed; max rank exactly 1.0
    X = np.random.randn(n)
    Y = np.random.randn(n)
    Z = np.random.randn(n)

    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    # Rectangular grid
    n_bins = [4, 3, 5]
    checkerboarder = Checkerboarder(n_bins, dim=3)
    ccop = checkerboarder.from_data(df)

    assert ccop.matr.shape == (4, 3, 5)
    # Proper probability tensor: nonnegative, sums to 1
    assert np.isclose(ccop.matr.sum(), 1.0)
    assert np.all(ccop.matr >= 0.0)


def test_boundary_conditions_for_independence():
    """Test boundary conditions for the checkerboard approximation."""
    # Test with independence (using Clayton with parameter close to 0)
    independent = copul.Families.CLAYTON.cls(0.01)  # Almost independent
    checkerboarder = copul.Checkerboarder(5)
    ccop = checkerboarder.get_checkerboard_copula(independent)

    # For independence, all cells should be approximately equal
    expected_value = 1.0 / 25  # 5x5 grid
    # Use a higher tolerance for near-independence
    assert np.all(np.abs(ccop.matr - expected_value) < 0.1)


def test_boundary_conditions_for_lower_frechet():
    lower_frechet = copul.Families.LOWER_FRECHET.cls()
    checkerboarder = Checkerboarder(5)
    ccop = checkerboarder.get_checkerboard_copula(lower_frechet)
    matr = ccop.matr
    for i in range(5):
        for j in range(5):
            if i + j == 4:
                assert np.isclose(matr[i, j], 0.2)
            else:
                assert np.isclose(matr[i, j], 0.0)


def test_boundary_conditions_for_upper_frechet():
    lower_frechet = copul.Families.UPPER_FRECHET.cls()
    checkerboarder = Checkerboarder(5)
    ccop = checkerboarder.get_checkerboard_copula(lower_frechet)
    matr = ccop.matr
    for i in range(5):
        for j in range(5):
            if i != j:
                assert np.isclose(matr[i, j], 0.0)
            else:
                assert np.isclose(matr[i, j], 0.2)


def test_compute_pi_with_galambos():
    """Test the computation of a checkerboard copula with the
    Galambos copula."""
    param = family_representatives["Galambos"]
    galambos = copul.Families.GALAMBOS.cls(param)
    checkerboarder = Checkerboarder(50)
    ccop = checkerboarder.get_checkerboard_copula(galambos)
    assert ccop.matr.shape == (50, 50)
