r"""
Numeric-first base class for copulas built from other copulas.

:class:`NumericBivCopula` is a :class:`~copul.family.core.biv_copula.BivCopula`
whose distribution function, conditional distributions and density are
*vectorised NumPy functions* instead of SymPy expressions.  Subclasses
implement

* ``_cdf(u, v)`` -- :math:`C(u,v)` on arrays of equal shape in :math:`[0,1]^2`;
* ``_h1(u, v)``  -- :math:`\partial_1 C(u,v) = P(V\le v\mid U=u)`;
* ``_h2(u, v)``  -- :math:`\partial_2 C(u,v) = P(U\le u\mid V=v)`;
* ``_pdf(u, v)`` -- the density (only if ``is_absolutely_continuous``);
* ``_rvs(n, rng)`` -- exact sampling with a :class:`numpy.random.Generator`,

and get the public API of every copul copula:

* ``cdf(u, v)``, ``cond_distr_1(u, v)``, ``cond_distr_2(u, v)``,
  ``pdf(u, v)`` accepting scalars or arrays (keyword form ``u=, v=`` too);
* the vectorised hooks ``cdf_vectorized``, ``pdf_vectorized`` and
  ``_numeric_callables`` used by :func:`copul.measures.backend.numeric_backend`;
* ``rvs(n, random_state)`` with ``random_state`` an ``int``, ``None`` or a
  :class:`numpy.random.Generator`;
* all dependence measures (``spearmans_rho(method=...)`` ...) through the
  package's measure dispatch; Blomqvist's :math:`\beta = 4C(\tfrac12,\tfrac12)-1`
  is always exact.
"""

from __future__ import annotations

import numpy as np

from copul.exceptions import PropertyUnavailableException
from copul.family.core.biv_copula import BivCopula

__all__ = [
    "NumericBivCopula",
    "as_rng",
    "component_callables",
    "component_is_ac",
    "component_rvs",
    "ensure_numeric_copula",
]


# ---------------------------------------------------------------------------
# helpers shared by the constructions
# ---------------------------------------------------------------------------


def as_rng(random_state=None) -> np.random.Generator:
    """Turn ``None``/``int``/``Generator``/``RandomState`` into a ``Generator``."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.randint(0, 2**31 - 1))
    return np.random.default_rng(random_state)


def ensure_numeric_copula(C, name: str = "copula"):
    """Validate that ``C`` is a fully specified bivariate copula object."""
    from copul.measures.backend import free_parameters

    if not hasattr(C, "cdf"):
        raise TypeError(f"{name} must be a bivariate copula object, got {type(C).__name__}.")
    dim = getattr(C, "dim", 2)
    if dim != 2:
        raise ValueError(f"{name} must be bivariate (dim={dim}).")
    free = free_parameters(C)
    if free:
        raise ValueError(
            f"{name} ({type(C).__name__}) has free parameters {free}; "
            "constructions need fully specified components."
        )
    return C


def component_callables(C):
    """Vectorised ``(cdf, h1, h2)`` of a component copula.

    Uses the component's own numeric hooks for :class:`NumericBivCopula`
    instances and :func:`copul.measures.backend.numeric_backend` otherwise.
    """
    if isinstance(C, NumericBivCopula):
        return C.cdf_vectorized, C._h1_clean, C._h2_clean
    from copul.measures.backend import numeric_backend

    be = numeric_backend(C)
    return be.cdf, be.h1, be.h2


def component_pdf(C):
    """Vectorised density of a component copula."""
    if isinstance(C, NumericBivCopula):
        return C.pdf_vectorized
    from copul.measures.backend import numeric_backend

    return numeric_backend(C).pdf


def component_is_ac(C) -> bool:
    """Whether a component is known to be absolutely continuous.

    Unknown (property missing or raising) counts as *not* absolutely
    continuous, so that no density is claimed for the construction.
    """
    try:
        return bool(C.is_absolutely_continuous)
    except Exception:
        return False


def component_rvs(C, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of a component copula as an ``(n, 2)`` array."""
    n = int(n)
    if n <= 0:
        return np.empty((0, 2))
    if isinstance(C, NumericBivCopula):
        return C.rvs(n, random_state=rng)
    fast = _fast_rvs(C, n, rng)
    if fast is not None:
        return fast
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        x = C.rvs(n, random_state=seed)
    except TypeError:
        np.random.seed(seed)
        x = C.rvs(n)
    return np.asarray(x, dtype=float).reshape(n, 2)


def _fast_rvs(C, n, rng):
    """Exact samplers for the Fréchet bounds and independence (their generic
    samplers are slow), ``None`` for other copulas."""
    from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
    from copul.family.frechet.lower_frechet import LowerFrechet
    from copul.family.frechet.upper_frechet import UpperFrechet

    if type(C) is UpperFrechet:
        t = rng.random(n)
        return np.column_stack([t, t])
    if type(C) is LowerFrechet:
        t = rng.random(n)
        return np.column_stack([t, 1.0 - t])
    if type(C) is BivIndependenceCopula:
        return rng.random((n, 2))
    return None


def _parse_uv(args, kwargs, fname):
    """Extract ``(u, v)`` from the call conventions of the copul API."""
    kwargs = dict(kwargs)
    u = kwargs.pop("u", None)
    v = kwargs.pop("v", None)
    if kwargs:
        raise TypeError(f"{fname}() got unexpected keyword arguments {sorted(kwargs)}")
    if len(args) == 2:
        u, v = args
    elif len(args) == 1:
        pts = np.asarray(args[0], dtype=float)
        if pts.ndim == 1 and pts.size == 2:
            u, v = pts[0], pts[1]
        elif pts.ndim == 2 and pts.shape[1] == 2:
            u, v = pts[:, 0], pts[:, 1]
        else:
            raise ValueError(f"{fname}(): expected a point (u, v) or an (n, 2) array.")
    elif len(args) > 2:
        raise TypeError(f"{fname}() takes at most two coordinates.")
    if u is None or v is None:
        raise TypeError(
            f"{fname}() of a numeric copula needs both coordinates, e.g. {fname}(0.3, 0.7)."
        )
    return u, v


def _finish(val, u, v):
    """Return a float for scalar input, an ndarray otherwise."""
    if np.ndim(u) == 0 and np.ndim(v) == 0:
        return float(np.asarray(val).reshape(-1)[0])
    return val


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------


class NumericBivCopula(BivCopula):
    """Bivariate copula given by vectorised NumPy functions (see module docstring)."""

    params: list = []
    intervals: dict = {}

    def __init__(self):
        super().__init__()

    # -- to be implemented by subclasses -----------------------------------
    def _cdf(self, u, v):  # pragma: no cover - abstract
        raise NotImplementedError

    def _h1(self, u, v):
        return self._fd(self._cdf, u, v, 0)

    def _h2(self, u, v):
        return self._fd(self._cdf, u, v, 1)

    def _pdf(self, u, v):
        return self._fd(self._h1_clean, u, v, 1, clip=False)

    def _rvs(self, n, rng):  # pragma: no cover - abstract
        raise NotImplementedError

    # -- properties --------------------------------------------------------
    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def is_symmetric(self) -> bool:
        g = np.linspace(0.05, 0.95, 13)
        U, V = np.meshgrid(g, g, indexing="ij")
        return bool(np.allclose(self.cdf_vectorized(U, V), self.cdf_vectorized(V, U), atol=1e-10))

    # -- finite differences (fallbacks) --------------------------------------
    @staticmethod
    def _fd(f, u, v, axis, clip=True):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        x = u if axis == 0 else v
        h = np.minimum(1e-6, 0.5 * np.minimum(x, 1 - x))
        h = np.where(h > 0, h, 1e-9)
        lo = np.clip(x - h, 0.0, 1.0)
        hi = np.clip(x + h, 0.0, 1.0)
        if axis == 0:
            val = (f(hi, v) - f(lo, v)) / (hi - lo)
        else:
            val = (f(u, hi) - f(u, lo)) / (hi - lo)
        return np.clip(val, 0.0, 1.0) if clip else np.maximum(val, 0.0)

    # -- vectorised hooks ------------------------------------------------------
    @staticmethod
    def _prep(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)

    def cdf_vectorized(self, u, v):
        """Vectorised :math:`C(u,v)` (clipped to the Fréchet–Hoeffding bounds)."""
        u, v = self._prep(u, v)
        with np.errstate(all="ignore"):
            c = np.asarray(self._cdf(u, v), dtype=float)
        c = np.broadcast_to(c, u.shape)
        return np.clip(c, np.maximum(u + v - 1.0, 0.0), np.minimum(u, v))

    def _h1_clean(self, u, v):
        u, v = self._prep(u, v)
        with np.errstate(all="ignore"):
            h = np.broadcast_to(np.asarray(self._h1(u, v), dtype=float), u.shape)
        return np.clip(np.nan_to_num(h, nan=0.0), 0.0, 1.0)

    def _h2_clean(self, u, v):
        u, v = self._prep(u, v)
        with np.errstate(all="ignore"):
            h = np.broadcast_to(np.asarray(self._h2(u, v), dtype=float), u.shape)
        return np.clip(np.nan_to_num(h, nan=0.0), 0.0, 1.0)

    def cond_distr_1_vectorized(self, u, v):
        r"""Vectorised :math:`\partial_1 C(u,v) = P(V\le v\mid U=u)`."""
        return self._h1_clean(u, v)

    def cond_distr_2_vectorized(self, u, v):
        r"""Vectorised :math:`\partial_2 C(u,v) = P(U\le u\mid V=v)`."""
        return self._h2_clean(u, v)

    def pdf_vectorized(self, u, v):
        """Vectorised density :math:`c(u,v)` (absolutely continuous copulas only)."""
        if not self.is_absolutely_continuous:
            raise PropertyUnavailableException(
                f"{type(self).__name__} is not (known to be) absolutely continuous."
            )
        u, v = self._prep(u, v)
        with np.errstate(all="ignore"):
            d = np.broadcast_to(np.asarray(self._pdf(u, v), dtype=float), u.shape)
        return np.maximum(np.nan_to_num(d, nan=0.0, posinf=np.inf), 0.0)

    def _numeric_callables(self):
        """Hook for :func:`copul.measures.backend.numeric_backend`."""
        d = {"cdf": self.cdf_vectorized, "h1": self._h1_clean, "h2": self._h2_clean}
        if self.is_absolutely_continuous:
            d["pdf"] = self.pdf_vectorized
        return d

    # -- public evaluation API --------------------------------------------
    def cdf(self, *args, **kwargs):
        """:math:`C(u,v)`; scalars give a ``float``, arrays an ``ndarray``."""
        u, v = _parse_uv(args, kwargs, "cdf")
        return _finish(self.cdf_vectorized(u, v), u, v)

    def cond_distr_1(self, *args, **kwargs):
        r""":math:`\partial_1 C(u,v) = P(V\le v\mid U=u)`."""
        u, v = _parse_uv(args, kwargs, "cond_distr_1")
        return _finish(self._h1_clean(u, v), u, v)

    def cond_distr_2(self, *args, **kwargs):
        r""":math:`\partial_2 C(u,v) = P(U\le u\mid V=v)`."""
        u, v = _parse_uv(args, kwargs, "cond_distr_2")
        return _finish(self._h2_clean(u, v), u, v)

    def cond_distr(self, i, *args, **kwargs):
        """``cond_distr_1`` (``i=1``) or ``cond_distr_2`` (``i=2``)."""
        if i == 1:
            return self.cond_distr_1(*args, **kwargs)
        if i == 2:
            return self.cond_distr_2(*args, **kwargs)
        raise ValueError(f"i must be 1 or 2, got {i}")

    def pdf(self, *args, **kwargs):
        """Density :math:`c(u,v)` (raises for copulas that are not absolutely continuous)."""
        u, v = _parse_uv(args, kwargs, "pdf")
        return _finish(self.pdf_vectorized(u, v), u, v)

    def rvs(self, n=1, random_state=None, approximate=False):
        """Exact samples of the copula.

        Parameters
        ----------
        n : int
            Number of samples.
        random_state : int, numpy.random.Generator or None
            Seed or generator.
        approximate : bool
            Ignored (sampling is exact).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n, 2)``.
        """
        n = int(n)
        if n <= 0:
            return np.empty((0, 2))
        out = np.asarray(self._rvs(n, as_rng(random_state)), dtype=float)
        return np.clip(out.reshape(n, 2), 0.0, 1.0)

    # -- measures -------------------------------------------------------------
    def blomqvists_beta(self, *args, **kwargs):
        r"""Blomqvist's :math:`\beta = 4C(\tfrac12,\tfrac12) - 1` (exact)."""
        return 4.0 * float(self.cdf_vectorized(0.5, 0.5)) - 1.0

    # -- misc ---------------------------------------------------------------
    def validate_copula(self, m: int = 21, tol: float = 1e-8, return_details: bool = False):
        """Numerically check the copula axioms on an ``(m+1) x (m+1)`` grid."""
        g = np.linspace(0.0, 1.0, m + 1)
        U, V = np.meshgrid(g, g, indexing="ij")
        C = self.cdf_vectorized(U, V)
        grounded = bool(np.all(np.abs(C[0, :]) <= tol) and np.all(np.abs(C[:, 0]) <= tol))
        margins = bool(
            np.allclose(C[-1, :], g, atol=5 * tol) and np.allclose(C[:, -1], g, atol=5 * tol)
        )
        mass = np.diff(np.diff(C, axis=0), axis=1)
        increasing = bool(mass.min() >= -1e-10 and abs(mass.sum() - 1.0) <= 1e-6)
        ok = grounded and margins and increasing
        if not return_details:
            return ok
        return ok, {
            "grounded_ok": grounded,
            "margins_ok": margins,
            "increasing_ok": increasing,
            "min_cell_mass": float(mass.min()),
        }

    def is_copula(self, m: int = 21, tol: float = 1e-8, return_details: bool = False):
        return self.validate_copula(m=m, tol=tol, return_details=return_details)

    def __call__(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError(f"{type(self).__name__} has no free parameters to set.")
        return self

    def __repr__(self):
        return f"{type(self).__name__}()"

    __str__ = __repr__
