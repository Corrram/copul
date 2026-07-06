"""
Bivariate Checkerboard Copula module.

This module provides a bivariate checkerboard copula implementation
that combines properties of both CheckPi and BivCopula classes.
"""

import numpy as np
from typing import Union, List
import warnings

import sympy

from copul.schur_order.ltd_verifier import LTDVerifier
from copul.checkerboard.check_pi import CheckPi
from copul.family.core.biv_core_copula import BivCoreCopula
from copul.schur_order.cis_verifier import CISVerifier


class BivCheckPi(CheckPi, BivCoreCopula):
    """
    Bivariate Checkerboard Copula class.

    This class implements a bivariate checkerboard copula, which is defined by
    a matrix of values that determine the copula's distribution.
    """

    params: List = []
    intervals: dict = {}

    def __init__(self, matr: Union[List[List[float]], np.ndarray], **kwargs):
        """
        Initialize a bivariate checkerboard copula.

        Args:
            matr: A matrix (2D array) defining the checkerboard distribution.
            **kwargs: Additional parameters passed to BivCopula.

        Raises:
            ValueError: If matrix dimensions are invalid or matrix contains negative values.
        """
        # Convert input to numpy array if it's a list
        if isinstance(matr, list):
            matr = np.array(matr, dtype=float)
        if isinstance(matr, sympy.Matrix):
            matr = np.array(matr).astype(float)

        # Input validation
        if not hasattr(matr, "ndim"):
            raise ValueError("Input matrix must be a 2D array or list")
        if matr.ndim != 2:
            raise ValueError(
                f"Input matrix must be 2-dimensional, got {matr.ndim} dimensions"
            )
        if np.any(matr < 0):
            raise ValueError("All matrix values must be non-negative")

        CheckPi.__init__(self, matr)
        BivCoreCopula.__init__(self, **kwargs)

        self.m = self.matr.shape[0]
        self.n = self.matr.shape[1]

        # Normalize matrix if not already normalized
        if not np.isclose(np.sum(self.matr), 1.0):
            warnings.warn(
                "Matrix not normalized. Normalizing to ensure proper density.",
                UserWarning,
            )
            self.matr = self.matr / np.sum(self.matr)

    def __str__(self) -> str:
        """
        Return a string representation of the copula.

        Returns:
            str: String representation showing dimensions of the checkerboard.
        """
        return f"BivCheckPi(m={self.m}, n={self.n})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        If the matrix is larger than 5x5, only the top-left 5x5 block is shown.

        Returns:
            str: A string representation of the object, including matrix info.
        """
        rows, cols = self.matr.shape
        if rows > 5 and cols > 5:
            matr_preview = np.array2string(
                self.matr[:5, :5], max_line_width=80, suppress_small=True
            ).replace("\n", " ")
            matr_str = f"{matr_preview} (top-left 5x5 block)"
        else:
            matr_str = self.matr.tolist()

        return f"BivCheckPi(matr={matr_str}, m={self.m}, n={self.n})"

    @property
    def is_symmetric(self) -> bool:
        """
        Check if the copula is symmetric (C(u,v) = C(v,u)).

        Returns:
            bool: True if the copula is symmetric, False otherwise.
        """
        if self.matr.shape[0] != self.matr.shape[1]:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        For checkerboard copulas, this property is always True.

        Returns:
            bool: Always True for checkerboard copulas.
        """
        return True

    @classmethod
    def generate_randomly(cls, grid_size: int | list | None = None, n: int = 1):
        if grid_size is None:
            grid_size = [2, 50]
        generated_copulas = []
        for i in range(n):
            if isinstance(grid_size, list):
                grid_size = np.random.randint(*grid_size)
            # 1) draw n permutations all at once via argsort of uniforms
            perms = np.argsort(
                np.random.rand(grid_size, grid_size), axis=1
            )  # shape (n,n)
            # 2) draw n cauchy random variables
            a = np.abs(np.random.standard_cauchy(size=grid_size))  # shape (n,)
            # a**1.5
            # a = a**1.5

            # 3) build weighted sum of permuted identity matrices:
            #    M[j,k] = sum_i a[i] * 1{perms[i,j] == k}
            #    -> we can do this in one np.add.at call
            rows = np.repeat(
                np.arange(grid_size)[None, :], grid_size, axis=0
            )  # shape (n,n)
            cols = perms  # shape (n,n)
            weights = np.broadcast_to(a[:, None], (grid_size, grid_size))  # shape (n,n)
            M = np.zeros((grid_size, grid_size), float)
            np.add.at(M, (rows.ravel(), cols.ravel()), weights.ravel())

            # 4) feed into copul
            generated_copulas.extend([cls(M)])
        if n == 1:
            return generated_copulas[0]
        return generated_copulas

    # ------------------------------------------------------------------
    # Diverse random checkerboard generation
    # ------------------------------------------------------------------
    #: Strategies used by :meth:`generate_diverse` / :meth:`random_bistochastic_matrix`.
    DIVERSE_STRATEGIES = (
        "permutation",
        "birkhoff_sparse",
        "birkhoff_dense",
        "sinkhorn_uniform",
        "sinkhorn_exponential",
        "sinkhorn_lognormal",
        "band",
        "power",
    )

    @staticmethod
    def _sinkhorn(A, iters: int = 2000, tol: float = 1e-13) -> np.ndarray:
        """Sinkhorn--Knopp normalization to a doubly stochastic matrix.

        Alternately rescales rows and columns of the nonnegative matrix ``A``
        until all row and column sums equal one (up to ``tol``).
        """
        A = np.array(A, dtype=float)
        A[A < 0] = 0.0
        # Guarantee a positive diagonal support so the iteration cannot stall
        # on an all-zero row or column.
        if not (A.sum(axis=1).all() and A.sum(axis=0).all()):
            A = A + 1e-12
        for _ in range(iters):
            A = A / A.sum(axis=1, keepdims=True)
            A = A / A.sum(axis=0, keepdims=True)
            if np.max(np.abs(A.sum(axis=1) - 1.0)) < tol:
                break
        return A

    @classmethod
    def random_bistochastic_matrix(cls, n: int, rng=None, strategy: str | None = None):
        """Draw a random ``n x n`` doubly stochastic matrix.

        Every returned matrix has all row and column sums equal, so that the
        induced :class:`BivCheckPi` has uniform margins, i.e. a genuine
        copula. The ``strategy`` controls the qualitative shape and is chosen
        uniformly at random from :attr:`DIVERSE_STRATEGIES` when ``None``:

        * ``"permutation"`` -- a single random permutation matrix (deterministic,
          close to the maximal-functional-dependence regime);
        * ``"birkhoff_sparse"`` -- a sparse convex combination of a few
          permutation matrices with heavy-tailed (Dirichlet, small concentration)
          weights, biased towards near-deterministic copulas;
        * ``"birkhoff_dense"`` -- a convex combination of many permutation
          matrices, biased towards near-independence;
        * ``"sinkhorn_*"`` -- Sinkhorn--Knopp normalization of a nonnegative
          base matrix with uniform, exponential, log-normal or sparsified
          entries (broadly spread interiors);
        * ``"band"`` -- a circulant mixture of cyclic shifts (mass near the
          diagonal), exactly doubly stochastic;
        * ``"power"`` -- ``U**p`` for uniform ``U`` and random ``p`` made doubly
          stochastic, interpolating between near-uniform and very peaked.

        Args:
            n: grid size (number of rows/columns).
            rng: ``numpy`` ``Generator``, integer seed, or ``None``.
            strategy: one of :attr:`DIVERSE_STRATEGIES`, or ``None`` to pick at
                random.

        Returns:
            np.ndarray: a doubly stochastic ``n x n`` matrix.
        """
        rng = np.random.default_rng(rng)
        n = int(n)
        if strategy is None:
            strategy = rng.choice(cls.DIVERSE_STRATEGIES)

        if strategy == "permutation":
            return np.eye(n)[rng.permutation(n)]

        if strategy in ("birkhoff_sparse", "birkhoff_dense"):
            if strategy == "birkhoff_sparse":
                k = int(rng.integers(1, min(4, n) + 1))
                w = rng.dirichlet(np.full(k, 0.3))
            else:
                k = int(rng.integers(n, 3 * n + 1))
                w = rng.dirichlet(np.ones(k))
            M = np.zeros((n, n))
            for wk in w:
                M += wk * np.eye(n)[rng.permutation(n)]
            return M

        if strategy.startswith("sinkhorn"):
            if strategy == "sinkhorn_uniform":
                A = rng.random((n, n))
            elif strategy == "sinkhorn_exponential":
                A = rng.exponential(1.0, (n, n))
            elif strategy == "sinkhorn_lognormal":
                A = np.exp(rng.normal(0.0, rng.uniform(0.5, 2.0), (n, n)))
            else:
                raise ValueError(f"Unknown strategy {strategy!r}")
            A = A + 1e-4 * A.max()  # full support => Sinkhorn converges exactly
            return cls._sinkhorn(A)

        if strategy == "band":
            # Circulant mixture of cyclic-shift permutations with offsets in
            # [-w, w]: exactly doubly stochastic by construction (each shift is
            # a permutation) and concentrated near the diagonal.
            w = int(rng.integers(0, n))
            offsets = np.arange(-w, w + 1)
            weights = rng.random(offsets.size) + 1e-3
            weights /= weights.sum()
            idx = np.arange(n)
            M = np.zeros((n, n))
            for off, wt in zip(offsets, weights):
                M[idx, (idx + off) % n] += wt
            return M

        if strategy == "power":
            A = rng.random((n, n)) ** rng.uniform(1.0, 4.0)
            A = A + 1e-4 * A.max()
            return cls._sinkhorn(A)

        raise ValueError(f"Unknown strategy {strategy!r}")

    @classmethod
    def generate_diverse(
        cls,
        n_samples: int = 1,
        grid_size=(2, 60),
        rng=None,
        strategy: str | None = None,
    ):
        """Generate diverse random checkerboard copulas.

        Each sample draws a (possibly random) grid size and a doubly stochastic
        matrix via :meth:`random_bistochastic_matrix`, yielding genuine copulas
        spread across the attainable set of dependence measures -- useful for
        stress-testing inequalities and attainable regions.

        Args:
            n_samples: number of copulas to generate.
            grid_size: fixed ``int`` grid size, or an inclusive ``(low, high)``
                range from which the grid size is drawn uniformly per sample.
            rng: ``numpy`` ``Generator``, integer seed, or ``None``.
            strategy: fixed strategy name, or ``None`` to randomize per sample.

        Returns:
            A single :class:`BivCheckPi` if ``n_samples == 1``, else a list.
        """
        rng = np.random.default_rng(rng)
        out = []
        for _ in range(int(n_samples)):
            if isinstance(grid_size, (tuple, list)):
                n = int(rng.integers(int(grid_size[0]), int(grid_size[1]) + 1))
            else:
                n = int(grid_size)
            M = cls.random_bistochastic_matrix(n, rng=rng, strategy=strategy)
            M = M / M.sum()  # pre-normalize to avoid the not-normalized warning
            out.append(cls(M))
        return out[0] if int(n_samples) == 1 else out

    def is_cis(self, i=1) -> bool:
        """
        Check if the copula is cis.
        """
        return CISVerifier(i).is_cis(self)

    def is_si(self, i=1) -> bool:
        """
        Check if the copula is stochastically increasing.
        """
        return self.is_cis(i)

    def is_ltd(self):
        return LTDVerifier().is_ltd(self)

    def is_lti(self):
        return LTDVerifier().is_lti(self)

    def is_rti(self):
        return LTDVerifier().is_rti(self)

    def is_rtd(self):
        return LTDVerifier().is_rtd(self)

    def transpose(self):
        """
        Transpose the checkerboard matrix.
        """
        return BivCheckPi(self.matr.T)

    def cond_distr_1(self, *args):
        return self.cond_distr(1, *args)

    def cond_distr_2(self, *args):
        return self.cond_distr(2, *args)

    def spearmans_rho(self) -> float:
        """
        Compute Spearman's rho for a bivariate checkerboard copula.
        """
        p = np.asarray(self.matr, dtype=float)
        m, n = p.shape
        # Compute the factors (2*(i+1)-1)=2i+1 for rows and columns:
        i = np.arange(m).reshape(-1, 1)  # Column vector (i from 0 to m-1)
        j = np.arange(n).reshape(1, -1)  # Row vector (j from 0 to n-1)

        numerator = (2 * m - 2 * i - 1) * (2 * n - 2 * j - 1)
        denominator = m * n
        omega = numerator / denominator
        trace = np.trace(omega.T @ p)
        return 3 * trace - 3

    def kendalls_tau(self) -> float:
        """
        Calculate the tau coefficient more efficiently using numpy's vectorized operations.

        Returns:
            float: The calculated tau coefficient.
        """
        Xi_m = 2 * np.tri(self.m) - np.eye(self.m)
        Xi_n = 2 * np.tri(self.n) - np.eye(self.n)
        return 1 - np.trace(Xi_m @ self.matr @ Xi_n @ self.matr.T)

    def chatterjees_xi(self, condition_on_y: bool = False) -> float:
        if condition_on_y:
            delta = self.matr.T
            m = self.n
            n = self.m
        else:
            delta = self.matr
            m = self.m
            n = self.n
        T = np.ones(n) - np.tri(n)
        M = T @ T.T + T.T + 1 / 3 * np.eye(n)
        trace = np.trace(delta.T @ delta @ M)
        xi = 6 * m / n * trace - 2
        return xi

    def blests_nu(self) -> float:
        """
        Blest's measure of rank association (nu) for a checkerboard copula.

        Closed-form matrix formula::

            nu(C^Δ_Π) = (24 / (m^2 n)) * tr(Δ^T K) - 2,
            where
            K = L_m^T U L_n
                + 1/2 L_m^T U
                + 1/2 U L_n
                + 1/4 U
                - 1/2 L_m^T E L_n
                - 1/4 L_m^T E
                - 1/3 E L_n
                - 1/6 E.

        Here::

            Δ is the m×n checkerboard matrix,
            L_m (resp. L_n) are strictly lower-triangular "ones" matrices,
            E is the m×n all-ones matrix,
            U = w e_n^T with w_i = m - i + 1.

        Returns:
            float: Blest's nu.
        """
        P = np.asarray(self.matr, dtype=float)
        m, n = P.shape

        # Strictly lower-triangular ones matrices L_m, L_n
        Lm = np.tri(m, m, k=-1, dtype=float)
        Ln = np.tri(n, n, k=-1, dtype=float)

        # E = ones(m,n), U = w e_n^T with w_i = m - i + 1
        E = np.ones((m, n), dtype=float)
        w = np.arange(m, 0, -1, dtype=float)  # [m, m-1, ..., 1]
        U = w[:, None] * np.ones((1, n), dtype=float)

        # Assemble K per the closed-form
        K = (
            Lm.T @ U @ Ln
            + 0.5 * (Lm.T @ U)
            + 0.5 * (U @ Ln)
            + 0.25 * U
            - 0.5 * (Lm.T @ E @ Ln)
            - 0.25 * (Lm.T @ E)
            - (1.0 / 3.0) * (E @ Ln)
            - (1.0 / 6.0) * E
        )

        # nu = (24 / (m^2 n)) * sum_{i,j} Δ_{ij} K_{ij} - 2
        nu = (24.0 / (m * m * n)) * np.sum(P * K) - 2.0
        return float(nu)

    @staticmethod
    def _W_diag(n: int):
        J = np.fliplr(np.eye(n))
        L = np.tri(n)
        H = J @ (L @ L.T) @ J
        return (H - 0.5 * np.ones((n, n)) - (1 / 6) * np.eye(n)) / n

    def spearmans_footrule(self) -> float:
        if self.m != self.n:
            warnings.warn("Footrule is implemented for square matrices only.")
            return np.nan
        n = self.n
        Wd = self._W_diag(n)
        return 6.0 * np.sum(Wd * self.matr) - 2.0

    def ginis_gamma(self) -> float:
        if self.m != self.n:
            warnings.warn("Gini's Gamma is implemented for square matrices only.")
            return np.nan

        n = self.n
        P = self.matr

        # main-diagonal weight
        J = np.fliplr(np.eye(n))
        L = np.tri(n)
        H = J @ (L @ L.T) @ J
        Wd = (H - 0.5 * np.ones((n, n)) - (1 / 6) * np.eye(n)) / n

        # anti-diagonal weight
        i = np.arange(n)[:, None]
        j = np.arange(n)[None, :]
        K = np.maximum(0, n - 1 - (i + j))  # Hankel ramp towards the anti-diagonal
        # The anti-diagonal cells contribute int_0^1 s(1-s) ds = 1/6 to the
        # integral of C along the secondary diagonal (NOT 1/3): on that diagonal
        # the cell-local coordinates satisfy t = 1 - s, so the bilinear term
        # Delta_{ij} * s * t integrates to Delta_{ij}/6.  Using 1/3 overstates
        # Gini's gamma -- e.g. the 2x2 countermonotone checkerboard would give
        # -1/3 instead of the correct -2/3.
        Wa = (K + (1 / 6) * J) / n

        return 4.0 * (np.sum(Wd * P) + np.sum(Wa * P)) - 2.0

    def blomqvists_beta(self):
        """Blomqvist’s beta."""
        return 4.0 * self.cdf(0.5, 0.5) - 1.0


if __name__ == "__main__":
    matr = [
        [3, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 2, 1, 0],
        [0, 0, 0, 3],
    ]
    check = BivCheckPi(matr)

    print(
        f"xi: {check.chatterjees_xi()}, tai: {check.kendalls_tau()}, rho: {check.spearmans_rho()}, SI: {check.is_si()}, LTD: {check.is_ltd()}, RTI: {check.is_rti()}, LTI: {check.is_lti()}, RTD: {check.is_rtd()}"
    )
    for alpha in range(0, 11):
        matr = [
            [3, 0, 0],
            [0, 1 + alpha * 0.1, 2 - alpha * 0.1],
            [0, 2 - alpha * 0.1, 1 + alpha * 0.1],
        ]
        # matr = [
        #     [3, 0, 0],
        #     [0, 0.5 * alpha, 0.5 * (6 - alpha)],
        #     [0, 0.5 * (6 - alpha), 0.5 * alpha],
        # ]
        # matr = [[1,0], [0, 1]]
        ccop = BivCheckPi(matr)
        result = ccop.is_ltd()
        # ccop.plot_c_over_u()
        # ccop.plot_cond_distr_1()
        xi = ccop.chatterjees_xi()
        rho = ccop.spearmans_footrule()
        print(f"Alpha={alpha}, 6-Alpha= {6 - alpha}, LTD={result}, xi={xi}, rho={rho}")
    # footrule = ccop.spearmans_footrule()
    # gini = ccop.ginis_gamma()
    # beta = ccop.blomqvists_beta()
    # # ccop.plot_cdf()
    # # ccop.plot_pdf()
    # print(
    #     f"xi = {xi:.3f}, rho = {rho:.3f}, footrule = {footrule:.3f}, gini = {gini:.3f}, beta = {beta:.3f}"
    # )
