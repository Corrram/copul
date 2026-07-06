"""Tests for the diverse random checkerboard generator on ``BivCheckPi``."""

import numpy as np
import pytest

from copul import BivCheckPi


@pytest.mark.parametrize("strategy", BivCheckPi.DIVERSE_STRATEGIES)
def test_random_bistochastic_matrix_is_doubly_stochastic(strategy):
    rng = np.random.default_rng(0)
    for n in (2, 3, 7, 20):
        M = BivCheckPi.random_bistochastic_matrix(n, rng=rng, strategy=strategy)
        assert M.shape == (n, n)
        assert np.all(M >= -1e-12)
        # all row sums equal and all column sums equal (doubly stochastic)
        assert np.allclose(M.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(M.sum(axis=0), 1.0, atol=1e-6)


def test_generate_diverse_yields_valid_copulas():
    rng = np.random.default_rng(42)
    cops = BivCheckPi.generate_diverse(200, grid_size=(2, 40), rng=rng)
    assert len(cops) == 200
    for C in cops:
        # uniform margins == genuine copula
        assert np.allclose(C.matr.sum(axis=1), 1.0 / C.n, atol=1e-6)
        assert np.allclose(C.matr.sum(axis=0), 1.0 / C.m, atol=1e-6)


def test_generate_diverse_single_returns_instance():
    C = BivCheckPi.generate_diverse(1, grid_size=5, rng=0)
    assert isinstance(C, BivCheckPi)
    assert C.n == 5


def test_diverse_sample_respects_xi_beta_region():
    """Every checkerboard is a copula, so it must satisfy |beta|^3 <= 2 xi."""
    rng = np.random.default_rng(7)
    cops = BivCheckPi.generate_diverse(500, grid_size=(2, 50), rng=rng)
    worst = max(abs(C.blomqvists_beta()) ** 3 - 2.0 * C.chatterjees_xi() for C in cops)
    assert worst <= 1e-9, f"region inequality violated by {worst:.2e}"
