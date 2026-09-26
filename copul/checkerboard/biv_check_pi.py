"""
Bivariate Checkerboard Copula module.

This module provides a bivariate checkerboard copula implementation
that combines properties of both CheckPi and BivCopula classes.  All
evaluations and dependence measures are exact and vectorised, see
:mod:`copul.checkerboard._biv_engine`.
"""

import numpy as np
import sympy

from copul.checkerboard import _biv_engine as eng
from copul.checkerboard._biv_mixin import BivCheckerboardMixin
from copul.checkerboard.check_pi import CheckPi
from copul.family.core.biv_core_copula import BivCoreCopula


class BivCheckPi(BivCheckerboardMixin, CheckPi, BivCoreCopula):
    """
    Bivariate Checkerboard Copula class.

    This class implements a bivariate checkerboard copula, which is defined by
    a matrix of values that determine the copula's distribution.  Inside each
    cell the mass is spread uniformly (independence kernel).
    """

    params: list = []
    intervals: dict = {}

    def __init__(self, matr: list[list[float]] | np.ndarray, **kwargs):
        """
        Initialize a bivariate checkerboard copula.

        Args:
            matr: A matrix (2D array) defining the checkerboard distribution,
                or another checkerboard copula (its ``matr`` is used).  The
                input is never modified; it is normalised to total mass one.
            **kwargs: Additional parameters (ignored).

        Raises:
            ValueError: If matrix dimensions are invalid or matrix contains negative values.
        """
        if hasattr(matr, "matr") and not isinstance(matr, np.ndarray):
            matr = matr.matr
        if isinstance(matr, sympy.Matrix):
            matr = np.array(matr.tolist(), dtype=float)
        matr = np.array(matr, dtype=float)

        # Input validation
        if matr.ndim != 2:
            raise ValueError(f"Input matrix must be 2-dimensional, got {matr.ndim} dimensions")
        if np.any(matr < 0):
            raise ValueError("All matrix values must be non-negative")

        CheckPi.__init__(self, matr)
        BivCoreCopula.__init__(self)

        self.m = self.matr.shape[0]
        self.n = self.matr.shape[1]

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
    def generate_randomly(
        cls,
        grid_size: int | list | tuple | None = None,
        n: int = 1,
        *,
        rng=None,
        random_state=None,
    ):
        """Generate random checkerboard copulas (sums of weighted permutations).

        Args:
            grid_size: fixed ``int`` grid size, or an inclusive ``[low, high]``
                (list or tuple) range from which the grid size is drawn
                uniformly *per sample*.  Defaults to ``[2, 50]``.
            n: number of copulas to generate.
            rng: ``numpy`` ``Generator``, integer seed or ``None`` (fresh
                entropy).  The global NumPy seed is never touched.
            random_state: alias of ``rng``.

        Returns:
            A single instance of ``cls`` if ``n == 1``, else a list.
        """
        if rng is None:
            rng = random_state
        rng = np.random.default_rng(rng)
        if grid_size is None:
            grid_size = (2, 50)
        generated_copulas = []
        for _ in range(int(n)):
            if isinstance(grid_size, (list, tuple, np.ndarray)):
                low, high = int(grid_size[0]), int(grid_size[1])
                size = int(rng.integers(low, high + 1))
            else:
                size = int(grid_size)
            # 1) draw `size` permutations via argsort of uniforms
            perms = np.argsort(rng.random((size, size)), axis=1)
            # 2) heavy-tailed (Cauchy) weights
            a = np.abs(rng.standard_cauchy(size=size))
            # 3) M[j, k] = sum_i a[i] * 1{perms[i, j] == k}
            rows = np.repeat(np.arange(size)[None, :], size, axis=0)
            weights = np.broadcast_to(a[:, None], (size, size))
            M = np.zeros((size, size), float)
            np.add.at(M, (rows.ravel(), perms.ravel()), weights.ravel())
            generated_copulas.append(cls(M / M.sum()))
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

    def transpose(self):
        """
        Transpose the checkerboard matrix (i.e. swap the roles of U and V).
        """
        return type(self)(self.matr.T)

    def rearrange_cis(self):
        """Stochastically increasing rearrangement (Strothmann, Dette, Siburg 2022).

        Returns:
            BivCheckPi: the rearranged checkerboard copula, which is SI and
            has the same margins.
        """
        from copul.schur_order.cis_rearranger import CISRearranger

        return BivCheckPi(CISRearranger.rearrange_checkerboard(self.matr))

    def pdf(self, *args, **kwargs):
        """Density; same call conventions as :meth:`cdf`."""
        u, v, scalar = eng.parse_uv(args, kwargs)
        return eng.finish(eng.pdf_pi(self.matr, u, v), scalar)

    @staticmethod
    def _W_diag(n: int):
        J = np.fliplr(np.eye(n))
        L = np.tri(n)
        H = J @ (L @ L.T) @ J
        return (H - 0.5 * np.ones((n, n)) - (1 / 6) * np.eye(n)) / n


if __name__ == "__main__":
    matr = [
        [3, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 2, 1, 0],
        [0, 0, 0, 3],
    ]
    check = BivCheckPi(matr)
    print(
        f"xi: {check.chatterjees_xi()}, tau: {check.kendalls_tau()}, "
        f"rho: {check.spearmans_rho()}, SI: {check.is_si()}, LTD: {check.is_ltd()}"
    )
