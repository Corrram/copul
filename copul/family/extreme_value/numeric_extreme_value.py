r"""
Bivariate extreme-value copulas given by a numerical Pickands function.

A bivariate extreme-value copula is

.. math::

   C_A(u,v) = \exp\Bigl(\log(uv)\,A\Bigl(\frac{\log v}{\log(uv)}\Bigr)\Bigr)
            = \exp\bigl(-\ell(-\log u, -\log v)\bigr),

with a convex Pickands dependence function :math:`A:[0,1]\to[1/2,1]`,
:math:`\max(t,1-t)\le A(t)\le 1`, and the stable tail dependence function
:math:`\ell(x,y)=(x+y)A(y/(x+y))` (Pickands 1981; Gudendorf & Segers 2010).

:class:`NumericExtremeValueCopula` turns any vectorised Pickands function
(e.g. an estimate, the attractor of a copula, or a function without a
SymPy expression) into a fully usable
:class:`~copul.family.extreme_value.biv_extreme_value_copula.BivExtremeValueCopula`:
the numerical backend evaluates :math:`C`, :math:`\partial_1C`,
:math:`\partial_2C` and the density from :math:`A, A', A''`; Spearman's
:math:`\rho` and Kendall's :math:`\tau` use the one-dimensional Pickands
formulas of the measures engine; :math:`\lambda_U=2-2A(1/2)`, Blomqvist's
:math:`\beta` and the footrule are exact; sampling is by conditional
inversion.

References
----------
Pickands, J. (1981). Multivariate extreme value distributions. *Bull. Int.
Statist. Inst.* 49, 859--878.
Gudendorf, G. & Segers, J. (2010). Extreme-value copulas. In *Copula Theory
and Its Applications*, Lecture Notes in Statistics 198, Springer, 127--145.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from copul.family.archimedean.numeric_archimedean import _call, fd_derivative
from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula

__all__ = ["NumericExtremeValueCopula", "NumericPickands"]


class NumericPickands:
    r"""Vectorised Pickands function :math:`A` with its derivatives.

    Calling it returns a ``float`` for scalar input and an ``ndarray``
    otherwise (also accepted: ``A(t=0.5)``).  ``deriv(t, k)`` gives
    :math:`A^{(k)}(t)`, :math:`k=1,2`.
    """

    def __init__(self, A: Callable, dA: Callable | None = None, d2A: Callable | None = None):
        self._A = A
        self._dA = dA
        self._d2A = d2A

    def value(self, t) -> np.ndarray:
        r""":math:`A(t)`, with :math:`A(0)=A(1)=1` and clipped to
        :math:`[\max(t,1-t), 1]` (removes rounding noise)."""
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        a = _call(self._A, t)
        a = np.where((t <= 0.0) | (t >= 1.0), 1.0, a)
        return np.clip(a, np.maximum(t, 1.0 - t), 1.0)

    def first(self, t) -> np.ndarray:
        r""":math:`A'(t)` (given or fourth-order finite differences)."""
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        if self._dA is not None:
            return _call(self._dA, t)
        return np.clip(fd_derivative(self.value, t, 1, 0.0, 1.0, hmin=1e-6), -1.0, 1.0)

    def second(self, t) -> np.ndarray:
        r""":math:`A''(t)` (given or finite differences), non-negative."""
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        if self._d2A is not None:
            return _call(self._d2A, t)
        if self._dA is not None:
            d2 = fd_derivative(self._dA, t, 1, 0.0, 1.0, hmin=1e-6)
        else:
            d2 = fd_derivative(self.value, t, 2, 0.0, 1.0, rel=2e-2, hmin=1e-4)
        return np.maximum(d2, 0.0)

    def deriv(self, t, k: int = 1):
        """:math:`A^{(k)}(t)` for ``k`` in ``(1, 2)``."""
        out = self.first(t) if k == 1 else self.second(t)
        return float(out) if np.ndim(t) == 0 else out

    def __call__(self, t=None, **kwargs):
        if t is None:
            t = kwargs.pop("t", None)
        if t is None:
            raise TypeError("NumericPickands needs numerical t")
        out = self.value(t)
        return float(out) if np.ndim(t) == 0 else out

    def __repr__(self):  # pragma: no cover - cosmetic
        return "NumericPickands()"


class NumericExtremeValueCopula(BivExtremeValueCopula):
    r"""Bivariate extreme-value copula with a numerical Pickands function.

    Parameters
    ----------
    A : callable
        Vectorised Pickands dependence function on :math:`[0,1]`.
    dA, d2A : callable, optional
        :math:`A'` and :math:`A''` (finite differences otherwise).
    name : str, optional
        Name used in ``repr``.
    absolutely_continuous : bool, optional
        Whether the copula has a density everywhere.  Detected from kinks of
        :math:`A` (jumps of :math:`A'`, which put singular mass on curves) if
        omitted.

    Examples
    --------
    >>> import numpy as np
    >>> from copul.family.extreme_value.numeric_extreme_value import NumericExtremeValueCopula
    >>> C = NumericExtremeValueCopula(lambda t: (t**2 + (1 - t) ** 2) ** 0.5)  # Gumbel(2)
    >>> round(C.lambda_U(), 12) == round(2 - 2**0.5, 12)
    True
    >>> round(C.kendalls_tau(), 8)
    0.5
    """

    params: list = []
    intervals: dict = {}
    _free_symbols: dict = {}

    def __init__(
        self,
        A: Callable,
        dA: Callable | None = None,
        d2A: Callable | None = None,
        *,
        name: str | None = None,
        absolutely_continuous: bool | None = None,
    ):
        self._pk = NumericPickands(A, dA, d2A)
        self._name = name
        self._ac = absolutely_continuous
        super().__init__()

    # -- Pickands function -------------------------------------------------------
    @property
    def pickands(self) -> NumericPickands:
        r"""The numerical Pickands function (callable, vectorised)."""
        return self._pk

    @pickands.setter
    def pickands(self, value):  # pragma: no cover - immutable
        raise AttributeError("the Pickands function of a numeric EV copula is fixed")

    def _pickands_numpy(self):
        """Hook for :func:`copul.measures.backend.numeric_backend` (``A, A', A''``)."""
        return self._pk.value, self._pk.first, self._pk.second

    def deriv_pickand_at_0(self):
        r""":math:`A'(0^+)`."""
        return float(self._pk.first(np.array([0.0]))[0])

    # -- evaluation ---------------------------------------------------------------------
    def cdf_vectorized(self, u, v):
        r"""Vectorised :math:`C(u,v)=\exp(-(x+y)A(y/(x+y)))`, :math:`x=-\log u`, :math:`y=-\log v`."""
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        with np.errstate(all="ignore"):
            x, y = -np.log(u), -np.log(v)
            s = x + y
            t = np.where(s > 0, y / s, 0.5)
            c = np.exp(-s * self._pk.value(t))
        c = np.where((u <= 0) | (v <= 0), 0.0, c)
        c = np.where(u >= 1, v, np.where(v >= 1, u, c))
        return np.clip(c, np.maximum(u + v - 1.0, 0.0), np.minimum(u, v))

    def cdf(self, *args, **kwargs):
        r""":math:`C(u,v)` for numerical arguments (there is no symbolic form)."""
        if not args and not kwargs:
            raise TypeError(
                "NumericExtremeValueCopula has no symbolic cdf; call cdf(u, v) with numbers."
            )
        u, v = _uv(args, kwargs)
        out = self.cdf_vectorized(u, v)
        return float(out) if np.ndim(out) == 0 else out

    def pdf(self, *args, **kwargs):
        r"""Density for numerical arguments (there is no symbolic form)."""
        raise TypeError("NumericExtremeValueCopula.pdf needs numerical arguments (u, v).")

    # -- properties --------------------------------------------------------------------
    @property
    def is_absolutely_continuous(self) -> bool:
        if self._ac is None:
            t = np.linspace(0.002, 0.998, 499)
            d1 = self._pk.first(t)
            jump = np.max(np.abs(np.diff(d1)))
            smooth = np.median(np.abs(np.diff(d1))) + 1e-12
            self._ac = bool(jump < max(50.0 * smooth, 1e-3))
        return self._ac

    @property
    def is_symmetric(self) -> bool:
        t = np.linspace(0.0, 1.0, 101)
        return bool(np.allclose(self._pk.value(t), self._pk.value(1.0 - t), atol=1e-12))

    def __call__(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError("NumericExtremeValueCopula has no free parameters to set.")
        return self

    def __repr__(self):
        return (
            f"NumericExtremeValueCopula({self._name})"
            if self._name
            else "NumericExtremeValueCopula()"
        )

    __str__ = __repr__


def _uv(args, kwargs):
    kwargs = dict(kwargs)
    u = kwargs.pop("u", None)
    v = kwargs.pop("v", None)
    if len(args) == 2:
        u, v = args
    elif len(args) == 1:
        pts = np.asarray(args[0], dtype=float)
        u, v = (pts[0], pts[1]) if pts.ndim == 1 else (pts[:, 0], pts[:, 1])
    if u is None or v is None:
        raise TypeError("cdf needs both coordinates u and v")
    return np.asarray(u, float), np.asarray(v, float)
