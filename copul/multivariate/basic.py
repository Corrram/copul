r"""
Basic :math:`d`-dimensional copulas and conversions.

* :class:`IndependenceND` -- the product copula
  :math:`\Pi_d(u)=\prod_i u_i` (independent components);
* :class:`UpperFrechetND` -- the upper Fréchet--Hoeffding bound
  :math:`M_d(u)=\min_i u_i` (comonotone components, Nelsen, 2006,
  §2.10);
* :class:`MixtureND` -- convex combinations :math:`\sum_k w_k C_k` (a convex
  combination of copulas is a copula, Nelsen, 2006, §2.2);
* :func:`as_copula_nd` -- view any copula object of :mod:`copul` (bivariate
  families, the legacy symbolic multivariate classes, :math:`d`-dimensional
  checkerboards, callables) as a :class:`~copul.multivariate.CopulaND`;
* :func:`margin` -- margins of any :math:`d`-copula with a cdf.

The lower bound :math:`W_d` is not a copula for :math:`d\ge 3` (Nelsen, 2006,
§2.10); it is available pointwise through
:func:`~copul.multivariate.frechet_hoeffding_bounds`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from copul.multivariate.base import (
    BivariateCopulaND,
    CopulaND,
    FunctionalCopulaND,
    normalize_index,
)

__all__ = ["IndependenceND", "MixtureND", "UpperFrechetND", "as_copula_nd", "margin"]


class IndependenceND(CopulaND):
    r"""Independence copula :math:`\Pi_d(u)=\prod_{i=1}^d u_i`.

    Parameters
    ----------
    dim : int
        Dimension :math:`d\ge 2`.

    Examples
    --------
    >>> from copul.multivariate import IndependenceND
    >>> IndependenceND(3).cdf([0.5, 0.5, 0.5])
    0.125
    >>> IndependenceND(4).kendalls_tau()
    0.0
    """

    radially_symmetric = True
    exchangeable = True

    def _cdf(self, U):
        return np.prod(U, axis=1)

    def _logpdf(self, U):
        return np.zeros(U.shape[0])

    def _survival(self, U):
        return np.prod(1.0 - U, axis=1)

    def _rvs(self, n, rng):
        return rng.random((n, self.dim))

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def _margin(self, idx):
        if len(idx) == 2:
            from copul.family.frechet.biv_independence_copula import BivIndependenceCopula

            return BivIndependenceCopula()
        return IndependenceND(len(idx))

    # exact measure values (all measures vanish at independence)
    def _exact_measure(self, key: str):
        return 0.0


class UpperFrechetND(CopulaND):
    r"""Upper Fréchet--Hoeffding bound :math:`M_d(u)=\min_i u_i`.

    The distribution of :math:`(V,\dots,V)` with :math:`V\sim U(0,1)`, i.e. of
    comonotone components (Nelsen, 2006, §2.10).  It is singular (all
    mass on the main diagonal), so ``pdf`` is unavailable.

    Parameters
    ----------
    dim : int
        Dimension :math:`d\ge 2`.
    """

    radially_symmetric = True
    exchangeable = True

    def _cdf(self, U):
        return np.min(U, axis=1)

    def _survival(self, U):
        return np.maximum(1.0 - np.max(U, axis=1), 0.0)

    def _rvs(self, n, rng):
        return np.repeat(rng.random((n, 1)), self.dim, axis=1)

    def _margin(self, idx):
        if len(idx) == 2:
            from copul.family.frechet.upper_frechet import UpperFrechet

            return UpperFrechet()
        return UpperFrechetND(len(idx))

    # all concordance measures equal one at M_d
    def _exact_measure(self, key: str):
        return 1.0


class MixtureND(CopulaND):
    r"""Convex combination :math:`C=\sum_k w_k C_k` of :math:`d`-copulas.

    Sampling draws the component of each observation from the weights; the
    density exists if all components are absolutely continuous.

    Parameters
    ----------
    copulas : sequence of CopulaND (or objects accepted by :func:`as_copula_nd`)
    weights : sequence of float, optional
        Non-negative weights summing to one (default: equal weights).

    Examples
    --------
    >>> from copul.multivariate import IndependenceND, MixtureND, UpperFrechetND
    >>> C = MixtureND([UpperFrechetND(3), IndependenceND(3)], [0.4, 0.6])
    >>> round(C.cdf([0.5, 0.5, 0.5]), 10)
    0.275
    """

    def __init__(self, copulas: Sequence[Any], weights: Sequence[float] | None = None) -> None:
        comps = [as_copula_nd(c) for c in copulas]
        if not comps:
            raise ValueError("MixtureND needs at least one component.")
        dims = {c.dim for c in comps}
        if len(dims) != 1:
            raise ValueError(f"all components must have the same dimension, got {sorted(dims)}.")
        w = np.full(len(comps), 1.0 / len(comps)) if weights is None else np.asarray(weights, float)
        if w.shape != (len(comps),) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
            raise ValueError("weights must be non-negative, one per component, and sum to one.")
        super().__init__(dims.pop())
        self.components = comps
        self.weights = w / w.sum()
        self.exchangeable = all(c.exchangeable for c in comps)
        self.radially_symmetric = all(c.radially_symmetric for c in comps)
        self._cheap_cdf = all(c._cheap_cdf for c in comps)

    def _cdf(self, U):
        return sum(w * c._cdf_clean(U) for w, c in zip(self.weights, self.components))

    def _survival(self, U):
        return sum(w * c._survival(U) for w, c in zip(self.weights, self.components))

    def _pdf(self, U):
        return sum(w * np.exp(c._logpdf_clean(U)) for w, c in zip(self.weights, self.components))

    def _rvs(self, n, rng):
        counts = rng.multinomial(n, self.weights)
        parts = [c._rvs(int(k), rng) for c, k in zip(self.components, counts) if k > 0]
        out = np.concatenate(parts, axis=0)
        return out[rng.permutation(n)]

    @property
    def is_absolutely_continuous(self) -> bool:
        return all(c.is_absolutely_continuous for c in self.components)

    def _margin(self, idx):
        ms = [c.margin(*idx) for c in self.components]
        if len(idx) == 2:
            from copul.family.constructions import mixture

            return mixture(ms, list(self.weights))
        return MixtureND(ms, self.weights)

    def __repr__(self) -> str:
        parts = ", ".join(f"{w:.3g}*{c!r}" for w, c in zip(self.weights, self.components))
        return f"MixtureND({parts})"


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------


def _cdf_from_object(obj: Any, dim: int):
    """A vectorized ``cdf(U)`` for legacy multivariate objects."""
    vec = getattr(obj, "cdf_vectorized", None)
    if callable(vec):

        def cdf(U):
            try:
                return np.asarray(vec(*[U[:, j] for j in range(dim)]), dtype=float)
            except (TypeError, ValueError):
                return np.asarray(vec(U), dtype=float)

        return cdf

    def cdf(U):
        try:
            out = np.asarray(obj.cdf(U), dtype=float)
            if out.shape == (U.shape[0],):
                return out
        except Exception:
            pass
        return np.array([float(obj.cdf(*row)) for row in U])

    return cdf


def _rvs_from_object(obj: Any):
    rvs = getattr(obj, "rvs", None)
    if not callable(rvs):
        return None

    def sample(n, rng):
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            return np.asarray(rvs(n, random_state=seed), dtype=float)
        except TypeError:
            return np.asarray(rvs(n), dtype=float)

    return sample


def as_copula_nd(obj: Any, dim: int | None = None) -> CopulaND:
    r"""View a copula object of :mod:`copul` as a :class:`CopulaND`.

    Supported inputs:

    * a :class:`CopulaND` (returned unchanged);
    * a bivariate copul copula (``cp.Clayton(2)``, constructions,
      checkerboards, ...) -- a :class:`~copul.multivariate.BivariateCopulaND`
      view with ``dim=2``;
    * the legacy symbolic multivariate classes: ``MultivariateGaussian``
      (:class:`~copul.multivariate.GaussianND`), ``MultivariateClayton`` and
      ``MultivariateGumbelHougaard`` (:class:`~copul.multivariate.ArchimedeanCopulaND`),
      ``IndependenceCopula`` (:class:`IndependenceND`), ``MVFrechet``
      (:class:`MixtureND` of :math:`M_d`, :math:`\Pi_d` and, for
      :math:`d=2`, :math:`W`);
    * any other object with a ``dim`` attribute and a ``cdf`` accepting an
      ``(N, d)`` array or ``cdf_vectorized(u1, ..., ud)`` (e.g.
      :math:`d`-dimensional ``CheckPi``) -- wrapped generically;
    * a callable ``cdf(U)`` (``dim`` required).

    Parameters
    ----------
    obj : object
    dim : int, optional
        Dimension (needed for plain callables).

    Returns
    -------
    CopulaND
    """
    if isinstance(obj, CopulaND):
        return obj
    from copul.family.core.biv_core_copula import BivCoreCopula

    if isinstance(obj, BivCoreCopula) or (
        getattr(obj, "dim", None) == 2 and hasattr(obj, "cond_distr_1_inv")
    ):
        return BivariateCopulaND(obj)
    name = type(obj).__name__
    d = getattr(obj, "dim", dim)
    if name in ("MultivariateGaussian",) and getattr(obj, "corr_matrix", None) is not None:
        from copul.multivariate.elliptical import GaussianND

        return GaussianND(np.asarray(obj.corr_matrix, dtype=float))
    if name == "MultivariateClayton":
        from copul.multivariate.archimedean import ArchimedeanCopulaND

        return ArchimedeanCopulaND("clayton", d, theta=float(obj.theta))
    if name in ("MultivariateGumbelHougaard", "MultivariateGumbelHougaardEV"):
        from copul.multivariate.archimedean import ArchimedeanCopulaND

        return ArchimedeanCopulaND("gumbel", d, theta=float(obj.theta))
    if name in ("IndependenceCopula", "MultivariateArchimedeanIndependence"):
        return IndependenceND(d)
    if name == "MVFrechet":
        a, b = float(obj.alpha), float(obj.beta)
        comps, w = [UpperFrechetND(d), IndependenceND(d)], [a, 1.0 - a - b]
        if d == 2 and b > 0:
            from copul.family.frechet.lower_frechet import LowerFrechet

            comps.append(BivariateCopulaND(LowerFrechet()))
            w.append(b)
        keep = [i for i, wi in enumerate(w) if wi > 0]
        return MixtureND([comps[i] for i in keep], [w[i] for i in keep])
    if callable(obj) and not hasattr(obj, "cdf"):
        if dim is None:
            raise ValueError("as_copula_nd(callable) needs dim=.")
        return FunctionalCopulaND(obj, dim)
    if d is None or not hasattr(obj, "cdf"):
        raise TypeError(f"cannot interpret {name} as a d-dimensional copula.")
    return FunctionalCopulaND(
        _cdf_from_object(obj, int(d)),
        int(d),
        rvs=_rvs_from_object(obj),
        name=f"as_copula_nd({name})",
    )


def margin(C: Any, idx: Sequence[int]) -> Any:
    r"""Margin :math:`C_I(u_I) = C(u_I,\mathbf 1)` of a :math:`d`-copula.

    Parameters
    ----------
    C : CopulaND or any object accepted by :func:`as_copula_nd`
    idx : sequence of int
        0-based indices of the retained components (in the desired order).

    Returns
    -------
    copul bivariate copula (two indices) or CopulaND
    """
    Cn = as_copula_nd(C)
    return Cn.margin(*normalize_index((idx,), Cn.dim))
