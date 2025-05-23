"""
Unit tests for the *block–diagonal* mixed checkerboard copula.

                       BivBlockDiagMixed(block_sizes, sign)

1) If the sign‐matrix is constant (–1 / 0 / +1) the copula must coincide
   with BivCheckW / BivCheckPi / BivCheckMin based on the *same* Δ.

2) For truly mixed sign patterns it must delegate per block and return
   plausible dependence measures.

3) All plotting helpers should run without raising.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt

from copul.checkerboard.biv_block_diag_mixed import BivBlockDiagMixed
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_w import BivCheckW

matplotlib.use("Agg")  # suppress GUI back-ends


# -------------------------------------------------------------------- #
# helpers                                                              #
# -------------------------------------------------------------------- #
def make_block_diag_delta(block_sizes: list[int]) -> np.ndarray:
    """Return the canonical block-diagonal Δ with masses 1/(d·n_r)."""
    d = sum(block_sizes)
    Δ = np.zeros((d, d), dtype=float)
    offs = np.concatenate(([0], np.cumsum(block_sizes)))
    for r, n_r in enumerate(block_sizes):
        sl = slice(offs[r], offs[r + 1])
        Δ[sl, sl] = 1.0 / (d * n_r)
    return Δ


# -------------------------------------------------------------------- #
# 1)  Constant sign matrix  →  should match the specialised classes    #
# -------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sign_val, base_cls",
    [
        (0, BivCheckPi),  # independence
        (+1, BivCheckMin),  # perfect + dependence
        (-1, BivCheckW),  # perfect – dependence
    ],
)
def test_constant_sign_matches_base(sign_val, base_cls):
    sizes = [2, 2]  # d = 4
    Δ = make_block_diag_delta(sizes)
    S = np.full_like(Δ, sign_val, dtype=int)

    cop_mixed = BivBlockDiagMixed(sizes, sign=S)
    cop_base = base_cls(Δ)

    pts = [(0.2, 0.2), (0.8, 0.8), (0.25, 0.6), (1.0, 1.0)]
    for u, v in pts:
        assert np.isclose(cop_mixed.cdf(u, v), cop_base.cdf(u, v))

    tau = cop_mixed.tau()
    tau_actual = cop_base.tau()
    rho = cop_mixed.rho()
    rho_actual = cop_base.rho()
    xi = cop_mixed.xi()
    xi_actual = cop_base.xi()
    assert np.isclose(tau, tau_actual)
    assert np.isclose(rho, rho_actual)
    assert np.isclose(xi, xi_actual)


# -------------------------------------------------------------------- #
# 2)  Mixed pattern: check per-block delegation and measures           #
# -------------------------------------------------------------------- #
def test_mixed_block_pattern():
    sizes = [1, 2, 1]  # blocks of size 1,2,1  → d = 4
    Δ = make_block_diag_delta(sizes)

    # blocks: 0 → Π, 1 → Min, 2 → W
    S = np.zeros_like(Δ, dtype=int)
    S[0, 0] = 0  # block 0  (size 1)
    S[1:3, 1:3] = +1  # block 1  (size 2)
    S[3, 3] = -1  # block 2  (size 1)

    cop = BivBlockDiagMixed(sizes, sign=S)

    # build reference copulas for each pure regime
    pi = BivCheckPi(Δ)
    cm = BivCheckMin(Δ)
    cw = BivCheckW(Δ)

    tests = [
        ((0.1, 0.1), pi),  # block 0
        ((0.3, 0.3), cm),  # block 1
        ((0.3, 0.7), cm),  # block 1
        ((0.9, 0.9), cw),  # block 2
    ]
    for (u, v), ref in tests:
        assert np.isclose(cop.cdf(u, v), ref.cdf(u, v))

    # quick sanity on measures
    tau, rho, xi = cop.tau(), cop.rho(), cop.xi()
    assert -1 <= tau <= 1
    assert -1 <= rho <= 1
    assert 0 <= xi <= 1
    # central (+) block dominates  → expect positive tau
    assert tau > 0


# -------------------------------------------------------------------- #
# 3)  Block-diagonal class should agree with general mixed class       #
# -------------------------------------------------------------------- #
def test_agrees_with_general_class():
    sizes = [3, 1]
    Δ = make_block_diag_delta(sizes)

    S = np.zeros_like(Δ, dtype=int)
    S[:3, :3] = +1  # first block = Min
    # last block remains 0 (= Π)

    from copul.checkerboard.biv_check_mixed import BivCheckMixed

    cop_general = BivCheckMixed(Δ, sign=S)
    cop_block = BivBlockDiagMixed(sizes, sign=S)
    assert hasattr(cop_block, "scatter_plot")

    xi = cop_general.xi()
    tau = cop_general.tau()
    rho = cop_general.rho()
    assert np.isclose(tau, cop_block.tau())
    assert np.isclose(rho, cop_block.rho())
    assert np.isclose(xi, cop_block.xi())
    assert xi == tau <= rho


# -------------------------------------------------------------------- #
# 4)  Plotting helpers (smoke – should not crash)                      #
# -------------------------------------------------------------------- #
@pytest.fixture
def small_block_mixed():
    make_block_diag_delta([1, 1])
    S = np.array([[0, 0], [0, 1]])
    return BivBlockDiagMixed([1, 1], sign=S)


@pytest.mark.parametrize(
    "plot_fun",
    [
        lambda c: c.plot_cdf(),
        lambda c: c.plot_cond_distr_1(),
        lambda c: c.plot_cond_distr_2(),
    ],
)
def test_plotting_runs(plot_fun, small_block_mixed):
    try:
        plot_fun(small_block_mixed)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"plot raised {exc}")
    finally:
        plt.close("all")
