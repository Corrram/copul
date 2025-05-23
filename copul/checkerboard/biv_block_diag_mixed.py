"""
Block–diagonal mixed checkerboard copula.

*   Blocks sizes  n₁,…,n_k  along the diagonal  (∑ n_r = d).
*   Inside a block I_r×I_r every cell mass is  Δ_{ij}=1/(d·n_r).
*   A sign matrix S (−1/0/+1) chooses per cell between
        0  → Π        ↘ independence
        +1 → ↗ (check-min)   perfect +  dependence
        −1 → ↘ (check-w)     perfect −  dependence
*   All piece-wise numerics are inherited from
    :class:`copul.checkerboard.biv_check_mixed.BivCheckMixed`.
  ------------------------------------------------------------------
  Closed-form measures (Prop. block_diag_mixed)
      ξ = 1 − B₂/d² + d⁻² Σ_r P_r/n_r²
      τ = 1 − B₂/d² + d⁻² Σ_r S_r/n_r²
      ρ = 3 d⁻³ Σ_r n_r(2d−2a_r−n_r)² − 3 + d⁻¹ Σ_r S_r/n_r
  with
      B₂ = Σ_r n_r² ,
      S_r = Σ_{i,j∈I_r} S_{ij} ,
      P_r = Σ_{i,j∈I_r} |S_{ij}| ,
      a_r = starting index of block r  (0-based).
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np

from copul.checkerboard.biv_check_mixed import BivCheckMixed


class BivBlockDiagMixed(BivCheckMixed):
    # ------------------------------------------------------------------ #
    #                              init                                  #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        sizes: Sequence[int],
        *,
        sign: Union[np.ndarray, Sequence[Sequence[int]], None] = None,
        **kwargs,
    ):
        # ---- sanity for the block-size vector ------------------------ #
        if not sizes:
            raise ValueError("`sizes` must contain at least one positive integer")
        if any(s <= 0 for s in sizes):
            raise ValueError("all block sizes must be positive integers")

        self.sizes: List[int] = list(map(int, sizes))
        self.k: int = len(self.sizes)  # number of blocks
        self.d: int = sum(self.sizes)  # total dimension
        self._offsets = np.concatenate(([0], np.cumsum(self.sizes)))[:-1]

        # ---- build the canonical block-diagonal Δ -------------------- #
        Δ = self.make_block_diag_delta(self.sizes)

        # ---- delegate everything else to the mixed checkerboard ------ #
        super().__init__(Δ, sign=sign, **kwargs)

    # ------------------------------------------------------------------ #
    # readable repr / str                                                #
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:  # pragma: no cover
        return f"BivBlockDiagMixed(sizes={self.sizes})"

    __repr__ = __str__

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def make_block_diag_delta(block_sizes: list[int]) -> np.ndarray:
        """
        Canonical block-diagonal checkerboard matrix Δ.

        Each block I_r×I_r of size n_r×n_r receives mass 1/(d·n_r),
        where d = Σ n_r.
        """
        d = sum(block_sizes)
        Δ = np.zeros((d, d), dtype=float)

        offs = np.concatenate(([0], np.cumsum(block_sizes)))
        for r, n_r in enumerate(block_sizes):
            sl = slice(offs[r], offs[r + 1])
            Δ[sl, sl] = 1.0 / (d * n_r)

        return Δ

    # ------------------------------------------------------------------ #
    #  CLOSED-FORM DEPENDENCE MEASURES (override default formulas)       #
    # ------------------------------------------------------------------ #
    def _block_sums(self):
        """
        Return per-block (S_r, P_r) lists and B₂ = Σ n_r².
        """
        S_r, P_r = [], []
        B2 = 0
        for r, n_r in enumerate(self.sizes):
            a, b = self._offsets[r], self._offsets[r] + n_r
            blk = self.sign[a:b, a:b]
            S_r.append(int(blk.sum()))
            P_r.append(int(np.abs(blk).sum()))
            B2 += n_r * n_r
        return S_r, P_r, B2

    # --------------  Chatterjee’s ξ  ---------------------------------- #
    def xi(self, *, condition_on_y: bool = False) -> float:  # noqa: D401
        S_r, P_r, B2 = self._block_sums()
        d = self.d
        term_blocks = sum(P / n_r**2 for P, n_r in zip(P_r, self.sizes))
        return 1.0 - B2 / d**2 + term_blocks / d**2

    # --------------  Kendall’s τ  ------------------------------------- #
    def tau(self) -> float:
        S_r, _, B2 = self._block_sums()
        d = self.d
        term_blocks = sum(S / n_r**2 for S, n_r in zip(S_r, self.sizes))
        return 1.0 - B2 / d**2 + term_blocks / d**2

    # --------------  Spearman’s ρ  ------------------------------------ #
    # --- inside BivBlockDiagMixed -----------------------------------------

    # --------------  Spearman’s ρ  ------------------------------------ #
    def rho(self) -> float:
        """
        Closed-form Spearman’s ρ for the block-diagonal mixed copula
            ρ = 3 d⁻³ Σ_r n_r (2d − 2a_r − n_r)²  − 3
                + d⁻³ Σ_r S_r / n_r .
        The second-line coefficient is 1/d³ ( *not* 1/d ).
        """
        S_r, _, _ = self._block_sums()
        d = self.d

        # --- checkerboard part --------------------------------------- #
        num = 0.0
        for a_r, n_r in zip(self._offsets, self.sizes):
            num += n_r * (2 * d - 2 * a_r - n_r) ** 2  # ❶ no “−2”
        core = (3.0 / d**3) * num - 3.0

        # --- sign-contribution  -------------------------------------- #
        extra = sum(S / n_r for S, n_r in zip(S_r, self.sizes)) / d**3  # ❷ 1/d³

        return core + extra


if __name__ == "__main__":  # pragma: no cover
    sizes = [1, 1, 2, 1]
    Δ = BivBlockDiagMixed.make_block_diag_delta(sizes)
    S = np.zeros_like(Δ, dtype=int)
    S[:2, :2] = +1
    S[0, 0] = 0
    S[2, 3] = 1
    S[3, 2] = 1
    S[4, 4] = 1
    cop_block = BivBlockDiagMixed(sizes, sign=S)
    cop_block.scatter_plot()
    xi = cop_block.xi()
    tau = cop_block.tau()
    rho = cop_block.rho()
    print(f"xi = {xi:.3f}, tau = {tau:.3f}")
