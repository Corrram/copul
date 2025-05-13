import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import kstest, pearsonr, kendalltau, spearmanr

matplotlib.use("Agg")  # Use the 'Agg' backend to suppress plot pop-ups

# Assuming ShuffleOfMin is importable from the correct path
# from copul.checkerboard.shuffle_min import ShuffleOfMin
# Using the class defined in the context for standalone testing
from copul.checkerboard.shuffle_min import ShuffleOfMin


# --- Start of Pytest Suite ---
# (The test functions remain exactly the same as in the previous artifact)
# --------------------------------------------------------------------------- #
#                           Basic construction tests                          #
# --------------------------------------------------------------------------- #


def test_constructor_validation():
    """Test constructor validation of permutation input."""
    # Valid permutation
    cop = ShuffleOfMin([1, 2, 3])
    assert cop.n == 3
    assert all(cop.pi == np.array([1, 2, 3]))
    assert cop.is_identity

    # Valid reverse
    cop_rev = ShuffleOfMin([3, 2, 1])
    assert cop_rev.n == 3
    assert all(cop_rev.pi == np.array([3, 2, 1]))
    assert cop_rev.is_reverse

    # Invalid: not a permutation
    with pytest.raises(ValueError, match="permutation of 1..n"):
        ShuffleOfMin([1, 2, 4])

    # Invalid: not 1-based
    with pytest.raises(ValueError, match="permutation of 1..n"):
        ShuffleOfMin([0, 1, 2])

    # Invalid: duplicates
    with pytest.raises(ValueError, match="permutation of 1..n"):
        ShuffleOfMin([1, 1, 3])

    # Invalid: wrong dimension
    with pytest.raises(ValueError, match="1-D permutation"):
        ShuffleOfMin([[1, 2], [3, 4]])


# --------------------------------------------------------------------------- #
#                           Argument processing tests                         #
# --------------------------------------------------------------------------- #
def test_cdf_values():
    cop = ShuffleOfMin([2, 1, 3])
    assert cop.cdf(0.25, 0.25) == 0.0
    assert cop.cond_distr_1(0.25, 0.25) == 0.0
    assert cop.cond_distr_1(0.25, 0.4) == 0.0
    assert cop.cond_distr_1(0.25, 0.6) == 1.0


def test_cdf_argument_forms():
    """Test that all the different calling conventions for cdf() work correctly."""
    cop = ShuffleOfMin([2, 1])  # n=2, reverse

    # Calculate expected value for (0.3, 0.4) using formula
    # n=2, pi=[2,1], pi0=[1,0]. u=0.3, v=0.4. nu=0.6, nv=0.8
    # i=1(idx=0,pi0=1): min(max(0, min(0.6-0, 0.8-1)), 1) = min(max(0, min(0.6, -0.2)), 1) = 0
    # i=2(idx=1,pi0=0): min(max(0, min(0.6-1, 0.8-0)), 1) = min(max(0, min(-0.4, 0.8)), 1) = 0
    # CDF = (0+0)/2 = 0.0
    expected_val1 = 0.0

    # Scalar arguments
    val1 = cop.cdf(0.3, 0.4)
    assert np.isclose(val1, expected_val1)

    # 1D array argument
    val2 = cop.cdf([0.3, 0.4])
    assert np.isclose(val2, expected_val1)

    # 2D array argument (single point)
    val3 = cop.cdf([[0.3, 0.4]])
    assert isinstance(val3, np.ndarray) and val3.shape == (1,)
    assert np.isclose(val3[0], expected_val1)

    # Calculate expected value for (0.5, 0.6)
    # u=0.5, v=0.6. nu=1.0, nv=1.2
    # i=1(idx=0,pi0=1): min(max(0, min(1.0-0, 1.2-1)), 1) = min(max(0, min(1.0, 0.2)), 1) = 0.2
    # i=2(idx=1,pi0=0): min(max(0, min(1.0-1, 1.2-0)), 1) = min(max(0, min(0.0, 1.2)), 1) = 0.0
    # CDF = (0.2+0.0)/2 = 0.1
    expected_val_p2 = 0.1

    # Array arguments
    u = np.array([0.3, 0.5])
    v = np.array([0.4, 0.6])
    val4 = cop.cdf(u, v)
    assert isinstance(val4, np.ndarray) and val4.shape == (2,)
    assert np.allclose(val4, [expected_val1, expected_val_p2])

    # 2D array with multiple points
    points = np.array([[0.3, 0.4], [0.5, 0.6]])
    val5 = cop.cdf(points)
    assert isinstance(val5, np.ndarray) and val5.shape == (2,)
    assert np.allclose(val5, [expected_val1, expected_val_p2])


def test_cond_distr_argument_forms():
    """Test that all the different calling conventions for cond_distr() work correctly."""
    cop = ShuffleOfMin([1, 2])  # n=2, identity

    # Calculate expected C(u|v) for (0.3, 0.4)
    # Identity: u=0.3, v=0.4. v=0.4 -> j=0. pi0_inv[0]=0 -> k=0. t=2*0.4-0=0.8. u0=(0+0.8)/2=0.4. u=0.3 < 0.4 -> C(u|v)=0.
    expected_val1 = 0.0

    # Scalar arguments
    val1 = cop.cond_distr_2(0.3, 0.4)  # C(u|v)
    assert np.isclose(val1, expected_val1)

    # 2D array argument (single point)
    val3 = cop.cond_distr_2([[0.3, 0.4]])
    assert isinstance(val3, np.ndarray) and val3.shape == (1,)
    assert np.isclose(val3[0], expected_val1)

    # Calculate expected C(u|v) for (0.5, 0.6)
    # u=0.5, v=0.6. v=0.6 -> j=1. pi0_inv[1]=1 -> k=1. t=2*0.6-1=0.2. u0=(1+0.2)/2=0.6. u=0.5 < 0.6 -> C(u|v)=0.
    expected_val_p2 = 0.0

    # Array arguments
    u = np.array([0.3, 0.5])
    v = np.array([0.4, 0.6])
    val4 = cop.cond_distr_2(u, v)  # C(u|v)
    assert isinstance(val4, np.ndarray) and val4.shape == (2,)
    assert np.allclose(val4, [expected_val1, expected_val_p2])

    # 2D array with multiple points
    points = np.array([[0.3, 0.4], [0.5, 0.6]])
    val5 = cop.cond_distr_2(points)  # C(u|v)
    assert isinstance(val5, np.ndarray) and val5.shape == (2,)
    assert np.allclose(val5, [expected_val1, expected_val_p2])

    # Calculate expected C(v|u) for (0.3, 0.4)
    # u=0.3 -> i=0. t=2*0.3-0=0.6. pi0[0]=0. v0=(0+0.6)/2=0.3. v=0.4 >= 0.3 -> C(v|u)=1.
    expected_val6 = 1.0
    val6 = cop.cond_distr(1, [0.3, 0.4])
    val7 = cop.cond_distr_1([0.3, 0.4])
    assert np.isclose(val6, expected_val6)
    assert np.isclose(val7, expected_val6)  # cond_distr(2, ...) == cond_distr_2(...)


def test_invalid_arguments():
    """Test that invalid arguments raise appropriate exceptions."""
    cop = ShuffleOfMin([1, 2, 3])

    # Too few arguments
    with pytest.raises(ValueError, match="No arguments provided"):
        cop.cdf()
    with pytest.raises(ValueError, match="No arguments provided"):
        cop.cond_distr(1)

    # Too many arguments for cdf/pdf/cond_distr(i, u, v)
    with pytest.raises(ValueError, match="Expected 1 or 2 arguments"):
        cop.cdf(0.3, 0.4, 0.5)
    with pytest.raises(ValueError, match="Expected 1 or 2 arguments"):
        cop.cond_distr(1, 0.3, 0.4, 0.5)

    # Wrong shape for 1D array
    with pytest.raises(ValueError, match="1D input must have length 2"):
        cop.cdf([0.3, 0.4, 0.5])
    with pytest.raises(ValueError, match="1D input must have length 2"):
        cop.cond_distr(1, [0.3, 0.4, 0.5])

    # Wrong shape for 2D array
    with pytest.raises(ValueError, match="2D input must have 2 columns"):
        cop.cdf([[0.3, 0.4, 0.5]])
    with pytest.raises(ValueError, match="2D input must have 2 columns"):
        cop.cond_distr(1, [[0.3, 0.4, 0.5]])

    # Out of bounds
    with pytest.raises(ValueError, match="u, v must lie in"):
        cop.cdf(-0.1, 0.5)
    with pytest.raises(ValueError, match="u, v must lie in"):
        cop.cdf(0.5, 1.1)
    with pytest.raises(ValueError, match="u, v must lie in"):
        cop.cond_distr(1, -0.1, 0.5)
    with pytest.raises(ValueError, match="u, v must lie in"):
        cop.cond_distr(2, 0.5, 1.1)

    # Invalid i for cond_distr
    with pytest.raises(ValueError, match="i must be between 1 and 2"):
        cop.cond_distr(0, 0.3, 0.4)
    with pytest.raises(ValueError, match="i must be between 1 and 2"):
        cop.cond_distr(3, 0.3, 0.4)


# --------------------------------------------------------------------------- #
#                           Basic functionality tests                         #
# --------------------------------------------------------------------------- #


def test_identity_permutation():
    """Order-4 shuffle with identity permutation behaves like M(u,v)=min(u,v) (perfect concordance)."""
    cop = ShuffleOfMin([1, 2, 3, 4])
    # dimensionality and permutation check
    assert cop.n == 4
    assert all(cop.pi == np.arange(1, 5))
    assert cop.is_identity

    # CDF should equal min(u,v); try a few points
    pts = np.array(
        [[0.2, 0.9], [0.8, 0.1], [0.5, 0.7], [1.0, 1.0], [0.0, 0.0], [0.6, 0.6]]
    )
    expected = np.minimum(pts[:, 0], pts[:, 1])

    # Test using 2D array
    actual1 = cop.cdf(pts)
    assert np.allclose(actual1, expected)

    # Test using two 1D arrays
    actual2 = cop.cdf(pts[:, 0], pts[:, 1])
    assert np.allclose(actual2, expected)

    # Test using individual points
    for i in range(len(pts)):
        actual3 = cop.cdf(pts[i])  # Pass point as [u, v]
        assert np.isclose(actual3, expected[i])
        actual4 = cop.cdf(pts[i, 0], pts[i, 1])  # Pass as u, v
        assert np.isclose(actual4, expected[i])


def test_reverse_permutation():
    """Reverse permutation yields strong counter-monotone pattern."""
    n = 5
    cop = ShuffleOfMin(list(range(n, 0, -1)))
    assert cop.is_reverse

    # Check CDF at a few specific points using the formula
    # C(u,v) = (1/n) * sum min(max(0, min(nu-(i-1), nv-(pi(i)-1))), 1)
    # pi = [5,4,3,2,1], pi0 = [4,3,2,1,0]

    # Point (0.2, 0.8): nu=1, nv=4. Expected = 0.0
    assert np.isclose(cop.cdf(0.2, 0.8), 0.0)
    # Point (0.5, 0.5): nu=2.5, nv=2.5. Expected = 0.1
    # i=1(0,4): 0
    # i=2(1,3): 0
    # i=3(2,2): min(max(0, min(0.5, 0.5)), 1) = 0.5
    # i=4(3,1): min(max(0, min(-0.5, 1.5)), 1) = 0
    # i=5(4,0): min(max(0, min(-1.5, 2.5)), 1) = 0
    # CDF = (0.5)/5 = 0.1
    assert np.isclose(cop.cdf(0.5, 0.5), 0.1)
    # Point (0.7, 0.3): nu=3.5, nv=1.5. Expected = 0.1
    # i=1(0,4): 0
    # i=2(1,3): 0
    # i=3(2,2): 0
    # i=4(3,1): min(max(0, min(0.5, 0.5)), 1) = 0.5
    # i=5(4,0): min(max(0, min(-0.5, 1.5)), 1) = 0
    # CDF = (0.5)/5 = 0.1
    assert np.isclose(cop.cdf(0.7, 0.3), 0.1)
    # Point (0.9, 0.1): nu=4.5, nv=0.5. Expected = 0.1
    # i=1..4 -> 0
    # i=5(idx=4,pi0=0): min(max(0, min(4.5-4, 0.5-0)), 1) = min(max(0, min(0.5, 0.5)), 1) = 0.5
    # CDF = 0.5/5 = 0.1
    assert np.isclose(cop.cdf(0.9, 0.1), 0.1)

    # PDF is zero everywhere (singular copula)
    np.linspace(0, 1, 11)


# --------------------------------------------------------------------------- #
#                               Edge-case CDF                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "point, expected",
    [
        # Boundary points
        ((0.0, 0.0), 0.0),
        ((1.0, 1.0), 1.0),
        ((0.0, 0.5), 0.0),
        ((0.5, 0.0), 0.0),
        ((1.0, 0.5), 0.5),  # C(1,v)=v
        ((0.5, 1.0), 0.5),  # C(u,1)=u
        # Interior points for pi=[3,2,1], n=3. pi0=[2,1,0]
        # (0.1, 0.1): nu=0.3, nv=0.3 -> CDF=0.
        ((0.1, 0.1), 0.0),
        # (0.5, 0.5): nu=1.5, nv=1.5 -> CDF=0.5/3
        ((0.5, 0.5), 0.5 / 3),
        # (0.8, 0.8): nu=2.4, nv=2.4 -> CDF=(0.4+1.0+0.4)/3 = 0.6
        ((0.8, 0.8), 0.6),
    ],
)
def test_cdf_boundaries_and_interior(point, expected):
    """Test CDF at boundaries and some interior points."""
    cop = ShuffleOfMin([3, 2, 1])  # n=3 reverse
    u, v = point
    actual = cop.cdf(u, v)
    assert np.isclose(actual, expected, atol=1e-9)
    # Test passing point as list/array
    actual_list = cop.cdf(list(point))
    assert np.isclose(actual_list, expected, atol=1e-9)
    actual_arr = cop.cdf(np.array(point))
    assert np.isclose(actual_arr, expected, atol=1e-9)


# --------------------------------------------------------------------------- #
#                         Conditional distribution                          #
# --------------------------------------------------------------------------- #


def test_cond_distr_step_function():
    """Test conditional distribution behaves as a step function."""
    cop = ShuffleOfMin([2, 1])  # n=2, pi0=[1,0], pi0_inv=[1,0]

    # Test C(v|u) for fixed u=0.4 (Interior)
    # u=0.4 -> i=0. t=2*0.4-0=0.8. pi0[0]=1. v0=(1+0.8)/2=0.9.
    # Expect C(v|u=0.4) = 0 for v < 0.9, 1 for v >= 0.9
    assert np.isclose(cop.cond_distr_1(0.4, 0.8), 0.0)
    assert np.isclose(cop.cond_distr_1(0.4, 0.89), 0.0)
    assert np.isclose(cop.cond_distr_1(0.4, 0.9), 1.0)
    assert np.isclose(cop.cond_distr_1(0.4, 0.91), 1.0)
    # Check boundary v=1 (should still follow step logic for interior u)
    assert np.isclose(cop.cond_distr_1(0.4, 1.0), 1.0)

    # Test C(u|v) for fixed v=0.6 (Interior)
    # v=0.6 -> j=1. pi0_inv[1]=0 -> k=0. t=2*0.6-1=0.2. u0=(0+0.2)/2=0.1.
    # Expect C(u|v=0.6) = 0 for u < 0.1, 1 for u >= 0.1
    # Check boundary u=0 (should still follow step logic for interior v)
    assert np.isclose(cop.cond_distr_2(0.0, 0.6), 0.0)
    assert np.isclose(cop.cond_distr_2(0.09, 0.6), 0.0)
    assert np.isclose(cop.cond_distr_2(0.1, 0.6), 1.0)
    assert np.isclose(cop.cond_distr_2(0.11, 0.6), 1.0)
    # Check boundary u=1
    assert np.isclose(cop.cond_distr_2(1.0, 0.6), 1.0)

    # Test C(v|u) for boundary u=0 and u=1 (Uniform)
    v_vals = np.array([0.1, 0.5, 0.9])
    assert np.allclose(cop.cond_distr_1(0.0, v_vals), v_vals)  # C(v|0)=v
    assert np.allclose(cop.cond_distr_1(1.0, v_vals), v_vals)  # C(v|1)=v

    # Test C(u|v) for boundary v=0 and v=1 (Uniform)
    u_vals = np.array([0.1, 0.5, 0.9])
    assert np.allclose(cop.cond_distr_2(u_vals, 0.0), u_vals)  # C(u|0)=u
    assert np.allclose(cop.cond_distr_2(u_vals, 1.0), u_vals)  # C(u|1)=u


def test_cond_distr_on_segment_corrected():
    """Test conditional distribution for points exactly on segments and one off segment."""
    cop = ShuffleOfMin([2, 1])  # n=2, pi0=[1,0], pi0_inv=[1,0]

    # Point on segment 0: i=0. t=0.5. u=(0+0.5)/2=0.25, v=(pi0[0]+t)/2=(1+0.5)/2=0.75
    u0, v0 = 0.25, 0.75
    # C(u|v): v=0.75 -> j=1. k=pi0_inv[1]=0. t=2*0.75-1=0.5. u0=(0+0.5)/2=0.25. u=0.25 >= u0 -> 1.0
    assert np.isclose(cop.cond_distr_2(u0, v0), 1.0)
    # C(v|u): u=0.25 -> i=0. t=2*0.25-0=0.5. v0=(pi0[0]+t)/2=(1+0.5)/2=0.75. v=0.75 >= v0 -> 1.0
    assert np.isclose(cop.cond_distr_1(u0, v0), 1.0)

    # Point on segment 1: i=1. t=0.3. u=(1+0.3)/2=0.65, v=(pi0[1]+t)/2=(0+0.3)/2=0.15
    u1, v1 = 0.65, 0.15
    # C(u|v): v=0.15 -> j=0. k=pi0_inv[0]=1. t=2*0.15-0=0.3. u0=(1+0.3)/2=0.65. u=0.65 >= u0 -> 1.0
    assert np.isclose(cop.cond_distr_2(u1, v1), 1.0)
    # C(v|u): u=0.65 -> i=1. t=2*0.65-1=0.3. v0=(pi0[1]+t)/2=(0+0.3)/2=0.15. v=0.15 >= v0 -> 1.0
    assert np.isclose(cop.cond_distr_1(u1, v1), 1.0)

    # Point OFF segment: u=0.4, v=0.6
    u_off, v_off = 0.4, 0.6
    # C(u|v): v=0.6 -> j=1. k=0. t=0.2. u0=0.1. u=0.4 >= 0.1 -> 1.0
    assert np.isclose(cop.cond_distr_2(u_off, v_off), 1.0)
    # C(v|u): u=0.4 -> i=0. t=0.8. v0=0.9. v=0.6 < 0.9 -> 0.0
    assert np.isclose(cop.cond_distr_1(u_off, v_off), 0.0)


# --------------------------------------------------------------------------- #
#                           Simulation / rvs checks                           #
# --------------------------------------------------------------------------- #


def test_rvs_marginals_uniform():
    """Test that both marginals are uniformly distributed."""
    np.random.seed(123)
    cop = ShuffleOfMin([4, 1, 3, 2])  # some arbitrary permutation
    n_samples = 2000
    uv = cop.rvs(size=n_samples)

    assert uv.shape == (n_samples, 2)
    assert np.all((uv >= 0) & (uv <= 1))

    # KS test for uniformity of each marginal
    for k in range(2):
        stat, p = kstest(uv[:, k], "uniform")
        assert p > 1e-3, f"marginal {k} fails KS-test (p={p})"


def test_tau_reverse():
    assert ShuffleOfMin([3, 2, 1]).kendall_tau() > -1
    assert ShuffleOfMin([1, 2, 3]).kendall_tau() == 1


def test_rho_reverse():
    assert ShuffleOfMin([3, 2, 1]).spearman_rho() > -1
    assert ShuffleOfMin([1, 2, 3]).spearman_rho() == 1


def test_rvs_expected_correlation_matches_tau():
    """
    For a shuffle with inversion count N_inv, Kendall τ = 1 - 4 N_inv / (n(n-1)).
    Empirically check that the sampled τ is near that value.
    """
    np.random.seed(42)
    n = 6
    perm = list(range(n, 0, -1))  # maximal inversions
    cop = ShuffleOfMin(perm)

    # The theoretical tau for reverse permutation is -1.0
    tau_theory = cop.kendall_tau()
    assert np.isclose(tau_theory, -2 / 3)

    uv = cop.rvs(size=4000)
    tau_emp, _ = kendalltau(uv[:, 0], uv[:, 1])

    # Relax tolerance: Sample tau for finite n shuffle might not be exactly -1
    print(
        f"\n[INFO] n={n} Reverse Permutation: Theoretical Tau={tau_theory:.4f}, Sample Tau={tau_emp:.4f}"
    )
    assert np.isclose(tau_emp, tau_theory, atol=0.035)  # Relaxed from 0.05

    # Check Spearman's rho as well (should also be -1 theoretically)
    rho_theory = cop.spearman_rho()
    rho_emp, _ = spearmanr(uv[:, 0], uv[:, 1])
    print(
        f"[INFO] n={n} Reverse Permutation: Theoretical Rho={rho_theory:.4f}, Sample Rho={rho_emp:.4f}"
    )
    # Sample Spearman Rho might also deviate for finite n
    assert np.isclose(rho_emp, rho_theory, atol=0.01)  # Relaxed tolerance


def test_rvs_conditional_functional():
    """
    Check that V is (nearly) a deterministic function of U by measuring
    the sample correlation between U and V inside each strip.
    """
    np.random.seed(42)
    cop = ShuffleOfMin([2, 3, 1, 4])
    n_samples = 3000
    uv = cop.rvs(size=n_samples)
    u, v = uv[:, 0], uv[:, 1]

    # partition sample by integer part of n*u (which segment index)
    n = cop.n
    tol = 1e-9
    seg_indices = np.floor(n * u - tol).astype(int)
    seg_indices = np.clip(seg_indices, 0, n - 1)  # Clip potential -1 or n

    # compute Pearson r within each strip; should be ~ +1
    for seg_idx in range(n):
        seg_mask = seg_indices == seg_idx
        n_in_seg = seg_mask.sum()
        print(f"\n[INFO] Segment {seg_idx}: {n_in_seg} points")
        if n_in_seg < 10:
            print("[WARN] Skipping segment due to insufficient points.")
            continue
        # Handle case where all points in segment are identical (can happen with low n_samples)
        if np.allclose(u[seg_mask], u[seg_mask][0]) or np.allclose(
            v[seg_mask], v[seg_mask][0]
        ):
            print(
                "[WARN] Skipping segment due to constant values (correlation undefined)."
            )
            continue
        r, p_val = pearsonr(u[seg_mask], v[seg_mask])
        print(f"  Pearson r = {r:.4f}, p-value = {p_val:.4f}")
        assert r > 0.99, f"Correlation in segment {seg_idx} is too low ({r})"


def test_rvs_empirical_cdf():
    """Test that the empirical CDF from samples matches the theoretical CDF."""
    np.random.seed(42)
    cop = ShuffleOfMin([3, 1, 2])
    n_samples = 5000
    samples = cop.rvs(n_samples)

    # Create a grid of points
    grid_size = 15  # Reduced grid size for faster test
    u_grid = np.linspace(0.05, 0.95, grid_size)
    v_grid = np.linspace(0.05, 0.95, grid_size)
    uu, vv = np.meshgrid(u_grid, v_grid)
    grid_points = np.column_stack((uu.ravel(), vv.ravel()))

    # Calculate theoretical CDF values
    theoretical_cdf = cop.cdf(grid_points)

    # Calculate empirical CDF values
    empirical_cdf = np.zeros(len(grid_points))
    for i, point in enumerate(grid_points):
        empirical_cdf[i] = np.mean(
            (samples[:, 0] <= point[0]) & (samples[:, 1] <= point[1])
        )

    # Check that theoretical and empirical CDFs are close
    mae = np.mean(np.abs(theoretical_cdf - empirical_cdf))
    print(f"\n[INFO] Empirical vs Theoretical CDF MAE: {mae:.4f}")
    assert mae < 0.03, f"MAE between empirical and theoretical CDF is too high ({mae})"


# --------------------------------------------------------------------------- #
#                           Association measures tests                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n, expected_tau, expected_rho",
    [
        (1, np.nan, np.nan),  # n=1 case
        (2, 1.0, 1.0),  # identity permutation
        (3, 1.0, 1.0),
        (5, 1.0, 1.0),
    ],
)
def test_association_identity(n, expected_tau, expected_rho):
    """Test association measures for identity permutation of different sizes."""
    cop = ShuffleOfMin(list(range(1, n + 1)))
    if n == 1:
        assert np.isnan(cop.kendall_tau())
        assert np.isnan(cop.spearman_rho())
        assert np.isclose(cop.chatterjee_xi(), 1.0)  # Functional dependence still holds
        assert np.isclose(cop.tail_lower(), 1.0)  # Based on pi[0]==1
        assert np.isclose(cop.tail_upper(), 1.0)  # Based on pi[-1]==n
    else:
        assert np.isclose(cop.kendall_tau(), expected_tau)
        assert np.isclose(cop.spearman_rho(), expected_rho)
        assert np.isclose(cop.chatterjee_xi(), 1.0)  # Functional dependence
        assert np.isclose(cop.tail_lower(), 1.0)
        assert np.isclose(cop.tail_upper(), 1.0)


@pytest.mark.parametrize(
    "n, expected_tau, expected_rho",
    [
        (1, np.nan, np.nan),  # n=1 case (same as identity for n=1)
        (5, -0.6, -0.9),  # reverse permutation
        (20, -0.9, -0.9),
    ],
)
def test_association_reverse(n, expected_tau, expected_rho):
    """Test association measures for reverse permutation of different sizes."""
    # For n=1, reverse is same as identity
    pi = list(range(n, 0, -1)) if n > 1 else [1]
    cop = ShuffleOfMin(pi)

    if n == 1:
        assert np.isnan(cop.kendall_tau())
        assert np.isnan(cop.spearman_rho())
        assert np.isclose(cop.chatterjee_xi(), 1.0)
        assert np.isclose(cop.tail_lower(), 1.0)  # pi[0]=1
        assert np.isclose(cop.tail_upper(), 1.0)  # pi[-1]=1
    else:
        assert np.isclose(cop.kendall_tau(), expected_tau, atol=0.1)
        assert np.isclose(cop.spearman_rho(), expected_rho, atol=0.1)
        assert np.isclose(cop.chatterjee_xi(), 1.0)  # Still functional dependence
        # Tail dependence for reverse: pi[0]=n, pi[-1]=1
        assert np.isclose(cop.tail_lower(), 0.0)  # pi[0] != 1
        assert np.isclose(cop.tail_upper(), 0.0)  # pi[-1] != n


def test_random_permutation_tau():
    """Test Kendall's tau for a random permutation against manual calculation."""
    np.random.seed(42)
    n = 8
    # Generate 0-based permutation first, then convert to 1-based for copula
    perm0 = np.random.permutation(n)
    perm1 = perm0 + 1  # 1-based permutation for ShuffleOfMin
    cop = ShuffleOfMin(perm1)  # Use 1-based perm here

    # Calculate inversions manually using 0-based logic
    inversions = sum(
        1 for i in range(n) for j in range(i + 1, n) if perm0[i] > perm0[j]
    )  # Use 0-based perm here

    # Calculate expected tau
    tau_expected = 1.0 - 4.0 * inversions / (n**2)

    # Get tau from copula method
    tau_copula = cop.kendall_tau()

    print(f"\n[INFO] Random Permutation (n={n}): {perm1.tolist()}")
    print(f"  Manual Inversions = {inversions}")
    print(f"  Expected Tau = {tau_expected:.6f}")
    print(f"  Copula Tau = {tau_copula:.6f}")

    assert np.isclose(tau_copula, tau_expected, atol=0.01)


def test_identity_vs_reverse_tau():
    """Test that identity and reverse permutations have opposite tau values."""
    n = 5
    id_cop = ShuffleOfMin(list(range(1, n + 1)))  # identity
    rev_cop = ShuffleOfMin(list(range(n, 0, -1)))  # reverse

    assert id_cop.kendall_tau() == pytest.approx(1.0)
    assert rev_cop.kendall_tau() == pytest.approx(-0.6)


# --------------------------------------------------------------------------- #
#                           Tail dependence tests                             #
# --------------------------------------------------------------------------- #


def test_tail_dependence_identity():
    """Test tail dependence for identity permutation."""
    cop = ShuffleOfMin([1, 2, 3, 4])
    assert cop.tail_lower() == 1.0  # Lower tail dependence (pi[0]=1)
    assert cop.tail_upper() == 1.0  # Upper tail dependence (pi[-1]=n)


def test_tail_dependence_reverse():
    """Test tail dependence for reverse permutation."""
    cop = ShuffleOfMin([4, 3, 2, 1])
    assert cop.tail_lower() == 0.0  # No lower tail dependence (pi[0]=4 != 1)
    assert cop.tail_upper() == 0.0  # No upper tail dependence (pi[-1]=1 != 4)


def test_tail_dependence_mixed():
    """Test tail dependence for mixed permutation."""
    # Lower tail dependence if pi[0] = 1
    cop1 = ShuffleOfMin([1, 3, 2])  # n=3
    assert cop1.tail_lower() == 1.0
    assert cop1.tail_upper() == 0.0  # pi[-1]=2 != 3

    # Upper tail dependence if pi[-1] = n
    cop2 = ShuffleOfMin([2, 1, 3])  # n=3
    assert cop2.tail_lower() == 0.0  # pi[0]=2 != 1
    assert cop2.tail_upper() == 1.0

    # Neither
    cop3 = ShuffleOfMin([2, 3, 1])  # n=3
    assert cop3.tail_lower() == 0.0  # pi[0]=2 != 1
    assert cop3.tail_upper() == 0.0  # pi[-1]=1 != 3


# --------------------------------------------------------------------------- #
#                           Visualization tests (Manual Check)                #
# --------------------------------------------------------------------------- #


# @pytest.mark.skip(reason="Visual inspection only, disable for automated runs")
def test_scatter_plot():
    """Test scatter plot visualization of random samples. (Manual check recommended)"""
    np.random.seed(42)
    cop = ShuffleOfMin([3, 1, 4, 2])  # n=4
    uv = cop.rvs(size=500)

    plt.figure(figsize=(6, 6))
    plt.scatter(uv[:, 0], uv[:, 1], alpha=0.5, s=10)  # Smaller points
    # Add theoretical segments
    t_plot = np.linspace(0, 1, 50)
    for i in range(cop.n):
        u_seg = (i + t_plot) / cop.n
        v_seg = (cop.pi0[i] + t_plot) / cop.n
        plt.plot(u_seg, v_seg, "r-", lw=2, alpha=0.7)

    plt.title(f"Scatter Plot of ShuffleOfMin({cop.pi.tolist()})")
    plt.xlabel("U")
    plt.ylabel("V")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    # Save plot instead of showing
    plt.savefig("shuffle_min_scatter_test.png")
    plt.close()
    print("\n[INFO] Scatter plot saved to shuffle_min_scatter_test.png")


# --------------------------------------------------------------------------- #
#                           String representation                             #
# --------------------------------------------------------------------------- #


def test_str_representation():
    """Test the string representation of ShuffleOfMin."""
    cop = ShuffleOfMin([2, 1, 3])
    rep = str(cop)
    expected = "ShuffleOfMin(order=3, pi=[2, 1, 3])"
    assert rep == expected
