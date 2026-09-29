r"""
Two-parameter Archimedean copulas given by a Laplace transform.

An Archimedean copula

.. math::

   C(u,v) = \psi\bigl(\varphi(u) + \varphi(v)\bigr),\qquad \varphi = \psi^{-1},

whose generator inverse :math:`\psi` is the Laplace transform of a positive
random variable :math:`V` (the *frailty*) is a valid copula in every
dimension.  Its conditional distributions and density are

.. math::

   \partial_1 C(u,v) = \psi'(s)\,\varphi'(u),\qquad
   c(u,v) = \psi''(s)\,\varphi'(u)\,\varphi'(v),\qquad
   s = \varphi(u)+\varphi(v),

and it is sampled exactly by the Marshall–Olkin algorithm
(:mod:`copul.family.bb._frailty`).

:class:`LTArchimedeanCopula` implements all of this generically on the
*logarithmic scale*: subclasses provide :math:`\log\varphi`,
:math:`\log(-\varphi')`, :math:`\psi`, :math:`\log(-\psi')` and
:math:`\log\psi''` as functions of :math:`\log s` (the generators of the BB
families overflow double precision in the interior of the unit square), the
symbolic :math:`\varphi` and :math:`\psi` (for the SymPy API with free
parameters) and a frailty sampler.  The classes behave like every other
copul family: free parameters stay symbolic (``BB1().cdf()`` is a SymPy
expression), fully specified instances evaluate numerically and are
understood by the measures engine through the ``_numeric_callables`` hook.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from copul.family.constructions._base import _finish, _parse_uv, as_rng
from copul.family.core.biv_copula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

__all__ = ["LTArchimedeanCopula", "log_expm1", "softplus"]


def softplus(x):
    r""":math:`\log(1+e^x)` without overflow."""
    return np.logaddexp(0.0, x)


def log_expm1(x):
    r""":math:`\log(e^x - 1)` for :math:`x>0`, accurate for small and large :math:`x`."""
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        big = x > 30.0
        return np.where(big, x + np.log1p(-np.exp(-np.where(big, x, 30.0))), np.log(np.expm1(x)))


def _is_numeric(x) -> bool:
    if isinstance(x, sp.Basic):
        return bool(x.is_number)
    return True


class LTArchimedeanCopula(BivCopula):
    """Base class of Archimedean copulas with a Laplace-transform generator inverse.

    Subclasses define ``params``/``intervals`` (SymPy symbols) and the
    numeric building blocks documented in the module docstring.
    """

    t = sp.symbols("t", positive=True)
    y = sp.symbols("y", nonnegative=True)

    # -- construction -----------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_params()

    def __call__(self, *args, **kwargs):
        new = super().__call__(*args, **kwargs)
        if isinstance(new, LTArchimedeanCopula):
            new._check_params()
        return new

    def _check_params(self):
        for p in type(self).params:
            name = str(p)
            val = getattr(self, name, p)
            if isinstance(val, sp.Basic) and not val.is_number:
                continue
            iv = type(self).intervals.get(name)
            if iv is not None and not bool(iv.contains(sp.Float(float(val)))):
                raise ValueError(f"{type(self).__name__}: {name}={val} outside {iv}")

    @property
    def _pv(self) -> tuple:
        return tuple(float(getattr(self, str(p))) for p in type(self).params)

    def _fully_specified(self) -> bool:
        for p in type(self).params:
            val = getattr(self, str(p), p)
            if isinstance(val, sp.Basic) and not val.is_number:
                return False
        return True

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    def __str__(self):
        return self.__repr__()

    # -- symbolic API ---------------------------------------------------------------
    def _sym_params(self):
        return tuple(getattr(self, str(p)) for p in type(self).params)

    def _phi_sym(self, t, *p):  # pragma: no cover - abstract
        raise NotImplementedError

    def _psi_sym(self, s, *p):  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def generator(self):
        r"""Generator :math:`\varphi(t)` (SymPy, variable ``t``)."""
        return SymPyFuncWrapper(self._phi_sym(self.t, *self._sym_params()))

    @property
    def inv_generator(self):
        r"""Generator inverse :math:`\psi(y)` = Laplace transform of the frailty."""
        return SymPyFuncWrapper(self._psi_sym(self.y, *self._sym_params()))

    @property
    def _cdf_expr(self):
        p = self._sym_params()
        return self._psi_sym(self._phi_sym(self.u, *p) + self._phi_sym(self.v, *p), *p)

    # -- numeric building blocks (log scale) -------------------------------------------
    def _log_phi(self, t, *p):  # pragma: no cover - abstract
        raise NotImplementedError

    def _log_mdphi(self, t, *p):  # pragma: no cover - abstract
        raise NotImplementedError

    def _psi_ls(self, ls, *p):  # pragma: no cover - abstract
        raise NotImplementedError

    def _log_mdpsi_ls(self, ls, *p):  # pragma: no cover - abstract
        raise NotImplementedError

    def _log_d2psi_ls(self, ls, *p):  # pragma: no cover - abstract
        raise NotImplementedError

    def _log_frailty(self, n, rng, *p):  # pragma: no cover - abstract
        raise NotImplementedError

    # -- vectorised evaluation ------------------------------------------------------------
    def _ls(self, u, v, p):
        return np.logaddexp(self._log_phi(u, *p), self._log_phi(v, *p))

    @staticmethod
    def _prep(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)

    def cdf_vectorized(self, u, v):
        r"""Vectorised :math:`C(u,v)`."""
        u, v = self._prep(u, v)
        p = self._pv
        with np.errstate(all="ignore"):
            c = self._psi_ls(self._ls(u, v, p), *p)
        lo, hi = np.maximum(u + v - 1.0, 0.0), np.minimum(u, v)
        c = np.where(np.isfinite(c), c, 0.5 * (lo + hi))
        c = np.where((u <= 0) | (v <= 0), 0.0, c)
        c = np.where(u >= 1, v, np.where(v >= 1, u, c))
        return np.clip(c, lo, hi)

    def _h(self, a, b):
        p = self._pv
        with np.errstate(all="ignore"):
            ls = self._ls(a, b, p)
            h = np.exp(self._log_mdpsi_ls(ls, *p) + self._log_mdphi(a, *p))
        h = np.where(np.isfinite(h), h, np.where(b >= a, 1.0, 0.0))
        h = np.where(b >= 1, 1.0, np.where(b <= 0, 0.0, h))
        return np.clip(h, 0.0, 1.0)

    def cond_distr_1_vectorized(self, u, v):
        r"""Vectorised :math:`\partial_1 C(u,v) = \psi'(s)\varphi'(u)`."""
        u, v = self._prep(u, v)
        return self._h(u, v)

    def cond_distr_2_vectorized(self, u, v):
        r"""Vectorised :math:`\partial_2 C(u,v) = \psi'(s)\varphi'(v)`."""
        u, v = self._prep(u, v)
        return self._h(v, u)

    def pdf_vectorized(self, u, v):
        r"""Vectorised density :math:`\psi''(s)\varphi'(u)\varphi'(v)`."""
        u, v = self._prep(u, v)
        p = self._pv
        with np.errstate(all="ignore"):
            ls = self._ls(u, v, p)
            d = np.exp(self._log_d2psi_ls(ls, *p) + self._log_mdphi(u, *p) + self._log_mdphi(v, *p))
        return np.maximum(np.nan_to_num(d, nan=0.0, posinf=np.inf), 0.0)

    def _numeric_callables(self):
        """Hook for :func:`copul.measures.backend.numeric_backend`."""
        return {
            "cdf": self.cdf_vectorized,
            "h1": self.cond_distr_1_vectorized,
            "h2": self.cond_distr_2_vectorized,
            "pdf": self.pdf_vectorized,
        }

    # -- public evaluation API (numeric fast path, symbolic otherwise) ------------------------
    def _numeric_call(self, args, kwargs) -> bool:
        if not self._fully_specified():
            return False
        vals = list(args) + [kwargs[k] for k in ("u", "v") if k in kwargs]
        if set(kwargs) - {"u", "v"}:
            return False
        if len(args) == 1:
            return True
        if len(vals) != 2:
            return False
        return all(_is_numeric(x) for x in vals)

    def cdf(self, *args, **kwargs):
        """:math:`C(u,v)`: numeric for numeric arguments, SymPy wrapper otherwise."""
        if self._numeric_call(args, kwargs):
            u, v = _parse_uv(args, kwargs, "cdf")
            return _finish(self.cdf_vectorized(u, v), u, v)
        return super().cdf(*args, **kwargs)

    def cond_distr_1(self, *args, **kwargs):
        r""":math:`\partial_1 C(u,v)`: numeric for numeric arguments, SymPy otherwise."""
        if self._numeric_call(args, kwargs):
            u, v = _parse_uv(args, kwargs, "cond_distr_1")
            return _finish(self.cond_distr_1_vectorized(u, v), u, v)
        return super().cond_distr_1(*args, **kwargs)

    def cond_distr_2(self, *args, **kwargs):
        r""":math:`\partial_2 C(u,v)`: numeric for numeric arguments, SymPy otherwise."""
        if self._numeric_call(args, kwargs):
            u, v = _parse_uv(args, kwargs, "cond_distr_2")
            return _finish(self.cond_distr_2_vectorized(u, v), u, v)
        return super().cond_distr_2(*args, **kwargs)

    def cond_distr(self, i, *args, **kwargs):
        if i in (1, 2) and self._numeric_call(args, kwargs):
            return (
                self.cond_distr_1(*args, **kwargs) if i == 1 else self.cond_distr_2(*args, **kwargs)
            )
        return super().cond_distr(i, *args, **kwargs)

    def pdf(self, *args, **kwargs):
        """Density: numeric for numeric arguments, SymPy wrapper otherwise."""
        if self._numeric_call(args, kwargs):
            u, v = _parse_uv(args, kwargs, "pdf")
            return _finish(self.pdf_vectorized(u, v), u, v)
        return super().pdf(*args, **kwargs)

    def rvs(self, n=1, random_state=None, approximate=False):
        r"""Exact samples by the Marshall–Olkin frailty algorithm.

        :math:`U_i = \psi(E_i/V)` with :math:`E_1,E_2\sim\mathrm{Exp}(1)`
        and the frailty :math:`V` of the family.

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
        numpy.ndarray of shape ``(n, 2)``
        """
        if not self._fully_specified():
            raise ValueError("rvs needs a fully specified copula.")
        n = int(n)
        if n <= 0:
            return np.empty((0, 2))
        rng = as_rng(random_state)
        p = self._pv
        lv = np.asarray(self._log_frailty(n, rng, *p), dtype=float).reshape(n)
        le = np.log(rng.exponential(size=(n, 2)))
        with np.errstate(all="ignore"):
            out = self._psi_ls(le - lv[:, None], *p)
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

    # -- measures valid for every member -------------------------------------------------
    def blomqvists_beta(self, *args, **kwargs):
        r"""Blomqvist's :math:`\beta = 4C(\tfrac12,\tfrac12) - 1 = 4\psi(2\varphi(\tfrac12)) - 1`."""
        if self._fully_specified():
            return 4.0 * float(self.cdf_vectorized(0.5, 0.5)) - 1.0
        p = self._sym_params()
        half = sp.Rational(1, 2)
        return 4 * self._psi_sym(2 * self._phi_sym(half, *p), *p) - 1
