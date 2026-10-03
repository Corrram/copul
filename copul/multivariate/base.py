r"""
Numerical :math:`d`-dimensional copulas.

:class:`CopulaND` is the common base class of the :math:`d`-dimensional
copulas of :mod:`copul.multivariate`.  It provides a uniform, vectorized
numerical API:

* ``cdf(u)``, ``pdf(u)``, ``logpdf(u)`` and ``survival_function(u)``
  :math:`=P(U_1>u_1,\dots,U_d>u_d)` accept an ``(N, d)`` array (returning
  an array of length ``N``), a single point of length ``d`` (returning a
  ``float``), any ``(..., d)`` array, or ``d`` separate broadcastable
  coordinates ``cdf(u1, ..., ud)``;
* ``rvs(n, random_state)`` returns an ``(n, d)`` sample;
* ``h_volume(a, b)`` is the :math:`C`-volume of the box :math:`[a,b]`;
* ``margin(i, j, ...)`` returns the copula of :math:`(U_i, U_j, \dots)`
  (0-based indices; a copul *bivariate* copula object for two indices);
* ``survival_copula()``, ``is_copula()``;
* multivariate dependence measures (``spearmans_rho(kind=...)``,
  ``kendalls_tau()``, ``blomqvists_beta()``, see
  :mod:`copul.multivariate.measures`) and matrices of pairwise measures.

A :math:`d`-copula is the restriction to :math:`[0,1]^d` of a distribution
function with standard uniform univariate margins; equivalently a function
:math:`C:[0,1]^d\to[0,1]` that is grounded, has uniform margins and is
:math:`d`-increasing (Nelsen, 2006, §2.10).  Every copula lies between
the Fréchet--Hoeffding bounds

.. math::

   W_d(u) = \max\Bigl(\sum_{i=1}^d u_i - d + 1,\ 0\Bigr) \le C(u)
   \le M_d(u) = \min_i u_i

(Nelsen, 2006, §2.10); :math:`M_d` is a copula for every :math:`d`,
:math:`W_d` only for :math:`d=2`.  Subclasses implement the vectorized hooks
``_cdf(U)`` (``U`` of shape ``(N, d)`` in :math:`[0,1]^d`), optionally
``_logpdf(U)`` / ``_pdf(U)`` (absolutely continuous copulas) and
``_rvs(n, rng)``; values returned by ``_cdf`` are clipped to the
Fréchet--Hoeffding bounds, which also enforces the boundary conditions
exactly.

References
----------
* Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press, ch. 1--3.
* Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer,
  §2.10.
* Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs
  marges. *Publ. Inst. Statist. Univ. Paris* 8, 229--231.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from copul.exceptions import PropertyUnavailableException
from copul.family.constructions._base import NumericBivCopula, as_rng

__all__ = [
    "BivariateCopulaND",
    "BivariateMarginCopula",
    "CopulaND",
    "FunctionalCopulaND",
    "MarginalCopulaND",
    "SurvivalCopulaND",
    "as_rng",
    "frechet_hoeffding_bounds",
    "parse_points",
]


# ---------------------------------------------------------------------------
# argument handling
# ---------------------------------------------------------------------------


def parse_points(args: Sequence[Any], dim: int, name: str = "cdf"):
    """Parse the call conventions of the evaluation methods.

    Returns ``(U, shape, scalar)`` with ``U`` a float array of shape
    ``(N, dim)``, the output shape and whether a ``float`` is returned.
    """
    if len(args) == 1:
        a = np.asarray(args[0], dtype=float)
        if a.ndim == 0 or a.shape[-1] != dim:
            raise ValueError(
                f"{name}(): expected an array whose last axis has length {dim} "
                f"(or {dim} separate coordinates), got shape {a.shape}."
            )
        return a.reshape(-1, dim), a.shape[:-1], a.ndim == 1
    if len(args) == dim:
        try:
            arrs = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in args])
        except ValueError as e:
            raise ValueError(f"{name}(): coordinates cannot be broadcast together: {e}") from None
        shape = arrs[0].shape
        return np.stack([x.ravel() for x in arrs], axis=1), shape, len(shape) == 0
    raise TypeError(
        f"{name}() takes one (..., {dim}) array or {dim} coordinates, got {len(args)} arguments."
    )


def finish(out, shape, scalar: bool):
    """Reshape a flat result to ``shape`` (``float`` for scalar input)."""
    out = np.asarray(out, dtype=float)
    if scalar:
        return float(out.reshape(-1)[0])
    return out.reshape(shape)


def normalize_index(idx: Sequence[Any], dim: int) -> tuple[int, ...]:
    """Validate 0-based margin indices (``margin(0, 2)`` or ``margin([0, 2])``)."""
    if len(idx) == 1 and np.ndim(idx[0]) == 1:
        idx = tuple(idx[0])
    out = tuple(int(i) for i in idx)
    if len(out) < 1:
        raise ValueError("margin(): at least one index is required.")
    if len(set(out)) != len(out):
        raise ValueError(f"margin(): indices must be distinct, got {out}.")
    for i in out:
        if not 0 <= i < dim:
            raise IndexError(f"margin(): index {i} out of range 0..{dim - 1}.")
    return out


def frechet_hoeffding_bounds(U: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""Fréchet--Hoeffding bounds :math:`W_d(u)` and :math:`M_d(u)`.

    .. math::

       W_d(u) = \max\Bigl(\sum_i u_i - d + 1, 0\Bigr),\qquad M_d(u) = \min_i u_i,

    the pointwise sharp bounds of every :math:`d`-copula (Nelsen, 2006,
    §2.10).

    Parameters
    ----------
    U : array_like of shape (N, d) or (d,)

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Lower and upper bound at each point.
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    d = U.shape[-1]
    return np.maximum(U.sum(axis=-1) - d + 1.0, 0.0), U.min(axis=-1)


def _subsets(d: int):
    """All subsets of ``range(d)`` as boolean masks, with their sizes."""
    masks = np.array(list(itertools.product([False, True], repeat=d)), dtype=bool)
    return masks, masks.sum(axis=1)


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------


class CopulaND:
    r"""Base class of numerical :math:`d`-dimensional copulas (see module docstring).

    Parameters
    ----------
    dim : int
        Dimension :math:`d\ge 2`.

    Notes
    -----
    Class attributes describing structural properties used by the measure
    routines:

    ``radially_symmetric``
        :math:`U \overset{d}{=} 1-U` (e.g. elliptical copulas); then
        :math:`\bar C(u) = C(1-u)`.
    ``exchangeable``
        :math:`C` is invariant under permutations of its arguments.
    ``_cheap_cdf``
        whether ``cdf`` is cheap enough for quasi-Monte Carlo integration.
    """

    radially_symmetric: bool = False
    exchangeable: bool = False
    _cheap_cdf: bool = True

    def __init__(self, dim: int) -> None:
        dim = int(dim)
        if dim < 2:
            raise ValueError(f"the dimension must be at least 2, got {dim}.")
        self.dim = dim

    # -- hooks ---------------------------------------------------------------
    def _cdf(self, U: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _logpdf(self, U: np.ndarray) -> np.ndarray:
        if type(self)._pdf is CopulaND._pdf:
            raise PropertyUnavailableException(f"{type(self).__name__} has no density.")
        with np.errstate(divide="ignore"):
            return np.log(self._pdf(U))

    def _pdf(self, U: np.ndarray) -> np.ndarray:
        if type(self)._logpdf is CopulaND._logpdf:
            raise PropertyUnavailableException(f"{type(self).__name__} has no density.")
        return np.exp(self._logpdf(U))

    def _rvs(self, n: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(f"{type(self).__name__} has no sampler.")

    def _exact_measure(self, key: str) -> float | None:
        """Closed-form value of a multivariate measure (``"rho1"``, ``"rho2"``,
        ``"rho3"``, ``"tau"``, ``"beta"``) or ``None`` (hook for subclasses)."""
        return None

    def _survival(self, U: np.ndarray) -> np.ndarray:
        r"""Inclusion--exclusion :math:`\bar C(u)=\sum_J(-1)^{|J|}C(u^J)`."""
        if self.radially_symmetric:
            return self._cdf_clean(1.0 - U)
        masks, sizes = _subsets(self.dim)
        n = U.shape[0]
        pts = np.where(masks[:, None, :], U[None, :, :], 1.0).reshape(-1, self.dim)
        vals = self._cdf_clean(pts).reshape(len(masks), n)
        signs = np.where(sizes % 2 == 0, 1.0, -1.0)
        return signs @ vals

    # -- properties -----------------------------------------------------------
    @property
    def is_absolutely_continuous(self) -> bool:
        """Whether the copula has a density (default ``False``)."""
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"

    def __str__(self) -> str:
        return self.__repr__()

    # -- evaluation -------------------------------------------------------------
    def _cdf_clean(self, U: np.ndarray) -> np.ndarray:
        U = np.clip(np.asarray(U, dtype=float), 0.0, 1.0)
        if U.shape[0] == 0:
            return np.empty(0)
        with np.errstate(all="ignore"):
            out = np.asarray(self._cdf(U), dtype=float).reshape(-1)
        lo, hi = frechet_hoeffding_bounds(U)
        out = np.where(np.isfinite(out), out, 0.5 * (lo + hi))
        return np.clip(out, lo, hi)

    def cdf(self, *args) -> float | np.ndarray:
        r"""Copula :math:`C(u_1,\dots,u_d)` (vectorized, see module docstring)."""
        U, shape, scalar = parse_points(args, self.dim, "cdf")
        return finish(self._cdf_clean(U), shape, scalar)

    __call__ = cdf

    def _require_density(self, what: str) -> None:
        if not self.is_absolutely_continuous:
            raise PropertyUnavailableException(
                f"{type(self).__name__} is not absolutely continuous; {what} is unavailable."
            )

    def _logpdf_clean(self, U: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        out = np.full(U.shape[0], -np.inf)
        inside = np.all((U >= 0.0) & (U <= 1.0), axis=1)
        if np.any(inside):
            with np.errstate(all="ignore"):
                val = np.asarray(self._logpdf(U[inside]), dtype=float).reshape(-1)
            out[inside] = np.where(np.isnan(val), -np.inf, val)
        return out

    def pdf(self, *args) -> float | np.ndarray:
        r"""Density :math:`c(u)=\partial^d C(u)/\partial u_1\cdots\partial u_d`."""
        self._require_density("pdf")
        U, shape, scalar = parse_points(args, self.dim, "pdf")
        with np.errstate(over="ignore"):
            return finish(np.exp(self._logpdf_clean(U)), shape, scalar)

    def logpdf(self, *args) -> float | np.ndarray:
        r"""Log-density :math:`\log c(u)` (``-inf`` outside the unit cube)."""
        self._require_density("logpdf")
        U, shape, scalar = parse_points(args, self.dim, "logpdf")
        return finish(self._logpdf_clean(U), shape, scalar)

    def survival_function(self, *args) -> float | np.ndarray:
        r"""Joint survival function :math:`\bar C(u) = P(U_1>u_1,\dots,U_d>u_d)`.

        .. math::

           \bar C(u) = \sum_{J\subseteq\{1,\dots,d\}} (-1)^{|J|}\,
           C\bigl(u^{J}\bigr),\qquad u^J_j = u_j\ (j\in J),\ u^J_j = 1\ (j\notin J)

        (inclusion--exclusion; :math:`2^d` evaluations of :math:`C`), or
        :math:`\bar C(u) = C(1-u)` for radially symmetric copulas.
        """
        U, shape, scalar = parse_points(args, self.dim, "survival_function")
        U = np.clip(U, 0.0, 1.0)
        out = np.clip(np.asarray(self._survival(U), dtype=float), 0.0, 1.0)
        return finish(out, shape, scalar)

    def rvs(self, n: int = 1, random_state: Any = None) -> np.ndarray:
        """``n`` samples of the copula as an ``(n, d)`` array.

        Parameters
        ----------
        n : int
            Number of samples.
        random_state : int, numpy.random.Generator or None
            Seed or generator (the global NumPy state is never used).
        """
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return np.empty((0, self.dim))
        out = np.asarray(self._rvs(n, as_rng(random_state)), dtype=float)
        return np.clip(out.reshape(n, self.dim), 0.0, 1.0)

    def h_volume(self, a: Any, b: Any) -> float | np.ndarray:
        r""":math:`C`-volume of the box :math:`[a, b] = \prod_i [a_i, b_i]`.

        .. math::

           V_C([a,b]) = \sum_{\varepsilon\in\{0,1\}^d} (-1)^{d-|\varepsilon|}
           C\bigl(c^\varepsilon\bigr),\qquad c^\varepsilon_i = b_i\ (\varepsilon_i=1),
           \ c^\varepsilon_i = a_i\ (\varepsilon_i = 0),

        i.e. :math:`P(U\in(a,b])` (Nelsen, 2006, §2.10).  ``a`` and ``b``
        are points of length ``d`` or ``(N, d)`` arrays with ``a <= b``.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        scalar = a.ndim == 1 and b.ndim == 1
        a2, b2 = np.broadcast_arrays(np.atleast_2d(a), np.atleast_2d(b))
        if a2.shape[-1] != self.dim:
            raise ValueError(f"h_volume(): corners must have length {self.dim}.")
        if np.any(a2 > b2):
            raise ValueError("h_volume(): need a <= b componentwise.")
        out = box_volume(self._cdf_clean, np.clip(a2, 0, 1), np.clip(b2, 0, 1))
        return float(out[0]) if scalar else out

    # -- derived copulas ---------------------------------------------------------
    def margin(self, *idx) -> Any:
        r"""Copula of the sub-vector :math:`(U_{i_1},\dots,U_{i_k})` (0-based indices).

        Obtained by setting the other arguments to one,
        :math:`C_{I}(u_I) = C(u_I, \mathbf 1)` (Nelsen, 2006, §2.10).  Two
        indices give a copul *bivariate* copula object (with the full
        bivariate API and all measures), more indices a :class:`CopulaND`.
        Subclasses return closed-form margins (e.g. a Gaussian copula with the
        sub-correlation matrix).
        """
        idx = normalize_index(idx, self.dim)
        if len(idx) == 1:
            raise ValueError("one-dimensional margins of a copula are uniform; use >= 2 indices.")
        if len(idx) > 2 and idx == tuple(range(self.dim)):
            return self
        return self._margin(idx)

    def _margin(self, idx: tuple[int, ...]) -> Any:
        if len(idx) == 2:
            return BivariateMarginCopula(self, idx)
        return MarginalCopulaND(self, idx)

    def survival_copula(self) -> CopulaND:
        r"""Survival copula :math:`\hat C`, the copula of :math:`1-U`.

        :math:`\hat C(u) = \bar C(1-u)` (Nelsen, 2006, §2.6 and §2.10).
        """
        return SurvivalCopulaND(self)

    def is_copula(self, grid: int | Sequence[float] = 11, tol: float = 1e-8, return_details=False):
        """Numerically check the copula axioms on a grid (see
        :func:`copul.multivariate.is_copula_nd`)."""
        from copul.multivariate.validation import is_copula_nd

        return is_copula_nd(self, grid=grid, tol=tol, return_details=return_details)

    # -- measures --------------------------------------------------------------------
    def spearmans_rho(self, kind: int = 1, method: str = "auto", **kwargs) -> float:
        r"""Multivariate Spearman's :math:`\rho_1`, :math:`\rho_2` or :math:`\rho_3`
        (Schmid & Schmidt, 2007); see :func:`copul.multivariate.spearmans_rho_nd`."""
        from copul.multivariate.measures import spearmans_rho_nd

        return spearmans_rho_nd(self, kind=kind, method=method, **kwargs)

    def kendalls_tau(self, method: str = "auto", **kwargs) -> float:
        r"""Multivariate Kendall's :math:`\tau_d` (Nelsen, 1996); see
        :func:`copul.multivariate.kendalls_tau_nd`."""
        from copul.multivariate.measures import kendalls_tau_nd

        return kendalls_tau_nd(self, method=method, **kwargs)

    def blomqvists_beta(self) -> float:
        r"""Multivariate Blomqvist's :math:`\beta_d` (Úbeda-Flores, 2005;
        Schmid & Schmidt, 2007); see :func:`copul.multivariate.blomqvists_beta_nd`."""
        from copul.multivariate.measures import blomqvists_beta_nd

        return blomqvists_beta_nd(self)

    def pairwise_matrix(self, measure: str = "tau", **kwargs) -> np.ndarray:
        r"""Matrix of a bivariate measure of all bivariate margins.

        Parameters
        ----------
        measure : str
            Name of a measure method of copul's bivariate copulas
            (``"tau"``/``"kendalls_tau"``, ``"rho"``/``"spearmans_rho"``,
            ``"beta"``/``"blomqvists_beta"``, ``"xi"``/``"chatterjees_xi"``,
            ``"lambda_L"``, ``"lambda_U"``, ...).
        **kwargs
            Passed to the measure method.

        Returns
        -------
        numpy.ndarray of shape (d, d)
            Entry :math:`(i,j)` is the measure of the margin :math:`(U_i,U_j)`;
            the diagonal holds the value at :math:`M` (one) for concordance
            measures.
        """
        name = _MEASURE_ALIASES.get(measure, measure)
        d = self.dim
        out = np.eye(d)
        if self.exchangeable:
            val = float(getattr(self.margin(0, 1), name)(**kwargs))
            return np.where(np.eye(d, dtype=bool), 1.0, val)
        symmetric = name not in _ASYMMETRIC_MEASURES
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if j < i and symmetric:
                    out[i, j] = out[j, i]
                else:
                    out[i, j] = float(getattr(self.margin(i, j), name)(**kwargs))
        return out

    def kendalls_tau_matrix(self) -> np.ndarray:
        r"""Matrix of the bivariate Kendall's :math:`\tau` of all pairs."""
        return self.pairwise_matrix("kendalls_tau")

    def spearmans_rho_matrix(self) -> np.ndarray:
        r"""Matrix of the bivariate Spearman's :math:`\rho` of all pairs."""
        return self.pairwise_matrix("spearmans_rho")


_MEASURE_ALIASES = {
    "tau": "kendalls_tau",
    "rho": "spearmans_rho",
    "beta": "blomqvists_beta",
    "xi": "chatterjees_xi",
    "footrule": "spearmans_footrule",
    "gamma": "ginis_gamma",
    "nu": "blests_nu",
}
_ASYMMETRIC_MEASURES = {"chatterjees_xi", "blests_nu"}


def box_volume(cdf: Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray):
    r"""Vectorized :math:`H`-volumes of boxes :math:`(a_k, b_k]` for a ``cdf``
    taking an ``(N, d)`` array (inclusion--exclusion over the :math:`2^d`
    vertices)."""
    n, d = a.shape
    masks, sizes = _subsets(d)
    verts = np.where(masks[:, None, :], b[None, :, :], a[None, :, :]).reshape(-1, d)
    vals = np.asarray(cdf(verts), dtype=float).reshape(len(masks), n)
    signs = np.where((d - sizes) % 2 == 0, 1.0, -1.0)
    return signs @ vals


# ---------------------------------------------------------------------------
# generic derived copulas
# ---------------------------------------------------------------------------


def _embed(U: np.ndarray, idx: tuple[int, ...], dim: int) -> np.ndarray:
    full = np.ones((U.shape[0], dim))
    full[:, list(idx)] = U
    return full


class MarginalCopulaND(CopulaND):
    r"""Copula :math:`C_I(u_I) = C(u_I,\mathbf 1)` of a sub-vector (generic).

    Parameters
    ----------
    parent : CopulaND
    idx : tuple of int
        0-based indices (at least three; two indices give a
        :class:`BivariateMarginCopula`).
    """

    def __init__(self, parent: CopulaND, idx: Sequence[int]) -> None:
        self.parent = parent
        self.idx = tuple(int(i) for i in idx)
        super().__init__(len(self.idx))
        self.exchangeable = parent.exchangeable
        self.radially_symmetric = parent.radially_symmetric
        self._cheap_cdf = parent._cheap_cdf

    def _cdf(self, U):
        return self.parent._cdf_clean(_embed(U, self.idx, self.parent.dim))

    def _rvs(self, n, rng):
        return self.parent._rvs(n, rng)[:, list(self.idx)]

    @property
    def is_absolutely_continuous(self) -> bool:
        return False  # a marginal density would need numerical integration

    def _margin(self, idx):
        return self.parent.margin(*[self.idx[i] for i in idx])

    def __repr__(self) -> str:
        return f"{self.parent!r}.margin{self.idx}"


class BivariateMarginCopula(NumericBivCopula):
    r"""Bivariate margin :math:`C_{ij}(u,v)` of a :class:`CopulaND` as a copul
    bivariate copula (cdf from the parent, conditional distributions by finite
    differences, sampling from the parent's sampler)."""

    def __init__(self, parent: CopulaND, idx: Sequence[int]) -> None:
        self.parent = parent
        self.idx = tuple(int(i) for i in idx)
        super().__init__()

    def _cdf(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        pts = _embed(np.column_stack([u.ravel(), v.ravel()]), self.idx, self.parent.dim)
        return self.parent._cdf_clean(pts).reshape(u.shape)

    def _rvs(self, n, rng):
        return self.parent._rvs(n, rng)[:, list(self.idx)]

    def __repr__(self) -> str:
        return f"{self.parent!r}.margin{self.idx}"

    __str__ = __repr__


class SurvivalCopulaND(CopulaND):
    r"""Survival copula :math:`\hat C(u) = \bar C(1-u)` of a :class:`CopulaND`
    (the copula of :math:`1-U`)."""

    def __init__(self, base: CopulaND) -> None:
        self.base = base
        super().__init__(base.dim)
        self.exchangeable = base.exchangeable
        self.radially_symmetric = base.radially_symmetric
        self._cheap_cdf = base._cheap_cdf

    def _cdf(self, U):
        return self.base._survival(1.0 - U)

    def _survival(self, U):
        return self.base._cdf_clean(1.0 - U)

    def _logpdf(self, U):
        return self.base._logpdf(1.0 - U)

    def _rvs(self, n, rng):
        return 1.0 - self.base._rvs(n, rng)

    @property
    def is_absolutely_continuous(self) -> bool:
        return self.base.is_absolutely_continuous

    def survival_copula(self) -> CopulaND:
        return self.base

    def _margin(self, idx):
        m = self.base.margin(*idx)
        if len(idx) == 2:
            from copul.family.constructions import survival

            return survival(m)
        return m.survival_copula()

    def __repr__(self) -> str:
        return f"SurvivalCopulaND({self.base!r})"


class FunctionalCopulaND(CopulaND):
    r""":math:`d`-copula given by vectorized callables.

    Parameters
    ----------
    cdf : callable
        ``cdf(U)`` with ``U`` of shape ``(N, d)`` returning ``N`` values.
    dim : int
        Dimension.
    pdf, logpdf : callable, optional
        Density or log-density with the same signature (makes the copula
        absolutely continuous).
    rvs : callable, optional
        ``rvs(n, rng)`` returning an ``(n, d)`` sample.
    name : str, optional
        Name used in ``repr``.

    Examples
    --------
    >>> import numpy as np
    >>> from copul.multivariate import FunctionalCopulaND
    >>> C = FunctionalCopulaND(lambda U: np.prod(U, axis=1), dim=3,
    ...                        pdf=lambda U: np.ones(len(U)))
    >>> C.cdf([0.5, 0.5, 0.5])
    0.125
    """

    def __init__(
        self,
        cdf: Callable[[np.ndarray], np.ndarray],
        dim: int,
        pdf: Callable[[np.ndarray], np.ndarray] | None = None,
        logpdf: Callable[[np.ndarray], np.ndarray] | None = None,
        rvs: Callable[[int, np.random.Generator], np.ndarray] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(dim)
        self._cdf_fn = cdf
        self._pdf_fn = pdf
        self._logpdf_fn = logpdf
        self._rvs_fn = rvs
        self.name = name or "FunctionalCopulaND"

    def _cdf(self, U):
        return self._cdf_fn(U)

    def _logpdf(self, U):
        if self._logpdf_fn is not None:
            return self._logpdf_fn(U)
        if self._pdf_fn is not None:
            with np.errstate(divide="ignore"):
                return np.log(np.asarray(self._pdf_fn(U), dtype=float))
        raise PropertyUnavailableException(f"{self.name} has no density.")

    def _rvs(self, n, rng):
        if self._rvs_fn is None:
            raise NotImplementedError(f"{self.name}: no sampler was given.")
        return self._rvs_fn(n, rng)

    @property
    def is_absolutely_continuous(self) -> bool:
        return self._pdf_fn is not None or self._logpdf_fn is not None

    def __repr__(self) -> str:
        return f"{self.name}(dim={self.dim})"


class BivariateCopulaND(CopulaND):
    r"""View of a copul *bivariate* copula as a :class:`CopulaND` with ``dim=2``.

    All evaluations are delegated to the vectorized numerical API of the
    bivariate copula; ``margin(0, 1)`` returns the original object and
    the measures use its (closed-form or numerical) bivariate measures.
    """

    def __init__(self, copula: Any) -> None:
        super().__init__(2)
        self.copula = copula
        try:
            self.exchangeable = bool(copula.is_symmetric)
        except Exception:
            self.exchangeable = False

    def _cdf(self, U):
        return np.asarray(self.copula.cdf(U[:, 0], U[:, 1]), dtype=float)

    def _logpdf(self, U):
        return np.asarray(self.copula.logpdf(U[:, 0], U[:, 1]), dtype=float)

    def _rvs(self, n, rng):
        return np.asarray(self.copula.rvs(n, random_state=rng), dtype=float)

    def _survival(self, U):
        return np.asarray(self.copula.survival_function(U[:, 0], U[:, 1]), dtype=float)

    @property
    def is_absolutely_continuous(self) -> bool:
        try:
            return bool(self.copula.is_absolutely_continuous)
        except Exception:
            return False

    def _margin(self, idx):
        from copul.family.constructions import transpose

        return transpose(self.copula)  # idx == (1, 0)

    def margin(self, *idx):
        idx = normalize_index(idx, 2)
        if idx == (0, 1):
            return self.copula
        return super().margin(*idx)

    def __repr__(self) -> str:
        return f"BivariateCopulaND({self.copula!r})"
