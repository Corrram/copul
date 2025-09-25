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


def test_from_data_trivariate_gaussian_with_marginals_and_dependence():
    """3D: from_data should produce a valid checkerboard with ~uniform marginals
    and a dependence structure similar to the original (via Spearman)."""
    np.random.seed(42)
    n_samples = 2000

    # Trivariate correlated normal data
    Sigma = np.array(
        [
            [1.0, 0.6, 0.4],
            [0.6, 1.0, 0.5],
            [0.4, 0.5, 1.0],
        ]
    )
    data = np.random.multivariate_normal(mean=[0, 0, 0], cov=Sigma, size=n_samples)
    df = pd.DataFrame(data, columns=["X", "Y", "Z"])

    # Build 10x10x10 checkerboard from data
    n_bins = 10
    checkerboarder = Checkerboarder(n_bins, dim=3)
    ccop = checkerboarder.from_data(df)

    # --- Basic checks ---
    assert ccop.matr.shape == (n_bins, n_bins, n_bins)
    assert np.isclose(ccop.matr.sum(), 1.0)
    assert np.all(ccop.matr >= 0.0)

    # --- 1D marginals should be ~uniform ---
    # Sum over the other two axes; each should be ~ 1/n_bins
    tol_1d = (
        0.03  # loose tolerance; histogramdd + ranks won't be exactly uniform per bin
    )
    m1 = ccop.matr.sum(axis=(1, 2))
    m2 = ccop.matr.sum(axis=(0, 2))
    m3 = ccop.matr.sum(axis=(0, 1))
    expected = np.full(n_bins, 1.0 / n_bins)
    assert np.allclose(m1, expected, atol=tol_1d)
    assert np.allclose(m2, expected, atol=tol_1d)
    assert np.allclose(m3, expected, atol=tol_1d)

    # --- 2D marginals: each should sum to 1 and have uniform 1D marginals ---
    tol_2d = 0.05
    M12 = ccop.matr.sum(axis=2)  # shape (n_bins, n_bins)
    M13 = ccop.matr.sum(axis=1)
    M23 = ccop.matr.sum(axis=0)

    for M in (M12, M13, M23):
        assert np.isclose(M.sum(), 1.0)
        # its 1D marginals (row/column sums) ~ uniform
        rows = M.sum(axis=1)
        cols = M.sum(axis=0)
        assert np.allclose(rows, expected, atol=tol_2d)
        assert np.allclose(cols, expected, atol=tol_2d)

    # --- Dependence similarity via Spearman correlation ---
    # Spearman on original data equals Pearson on ranks; compute once here.
    # Use the same (0,1] pseudo-observations the checkerboarder uses.
    def _pseudo_obs(x):
        # ranks in {1,...,n} scaled by n to (0,1]
        order = np.argsort(x)
        r = np.empty_like(order, dtype=float)
        r[order] = (np.arange(len(x)) + 1) / float(len(x))
        return r

    U = np.column_stack([_pseudo_obs(df[c].values) for c in ["X", "Y", "Z"]])
    df_u = pd.DataFrame(U, columns=["U1", "U2", "U3"])
    spearman_orig = df_u.corr(method="spearman").values

    # Sample from checkerboard copula (uniform margins); compare Spearman matrices
    # If rvs is unavailable for the checkerboard, skip this part gracefully.
    S = 4000
    samples = ccop.rvs(S)
    df_s = pd.DataFrame(samples, columns=["U1", "U2", "U3"])
    spearman_ccop = df_s.corr(method="spearman").values

    # Compare off-diagonal entries (dependence); allow modest tolerance
    offdiag_idx = np.triu_indices(3, k=1)
    diff = np.abs(spearman_ccop[offdiag_idx] - spearman_orig[offdiag_idx])
    assert np.all(diff < 0.03), f"Spearman off-diagonal diffs too large: {diff}"


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
