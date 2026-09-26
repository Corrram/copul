import numpy as np
import pytest

from copul.foci import codec, estimate_q, estimate_s, find_nearest_neighbors


def test_nearest_neighbour_never_self_with_repeated_points():
    X = np.array([[0.0], [0.0], [0.0], [1.0], [2.0], [5.0]])
    for seed in range(30):
        nn = find_nearest_neighbors(X, random_state=seed)
        assert not np.any(nn == np.arange(len(X)))
        assert set(nn[:3]) <= {0, 1, 2}  # duplicates point into their group
        assert nn[3] in (0, 1, 2, 4) and nn[5] == 4


def test_nearest_neighbour_repeats_are_uniform_over_other_members():
    X = np.zeros((4, 1))
    counts = np.zeros((4, 4))
    for seed in range(600):
        nn = find_nearest_neighbors(X, random_state=seed)
        counts[np.arange(4), nn] += 1
    assert np.all(np.diag(counts) == 0)
    off = counts[~np.eye(4, dtype=bool)]
    assert off.min() > 150  # expected 200 each


def test_nearest_neighbour_is_reproducible_and_does_not_touch_global_seed():
    rng = np.random.default_rng(0)
    X = np.round(rng.random((200, 2)), 1)  # many repeats and ties
    state = np.random.get_state()[1].copy()
    a = find_nearest_neighbors(X, random_state=3)
    b = find_nearest_neighbors(X, random_state=3)
    assert np.array_equal(a, b)
    assert np.array_equal(np.random.get_state()[1], state)
    assert not np.any(a == np.arange(200))


def test_estimates_are_plain_floats():
    rng = np.random.default_rng(1)
    Y = rng.random((50, 1))
    X = rng.random((50, 2))
    n = 50
    L = np.array([np.sum(y <= Y) for y in Y.ravel()], float)
    assert estimate_s(Y) == pytest.approx(np.sum(L * (n - L)) / n**3, abs=1e-15)
    assert isinstance(estimate_q(Y, X), float)
    assert 0.5 < codec((X[:, 0] + X[:, 1]) % 1, X) <= 1.0
