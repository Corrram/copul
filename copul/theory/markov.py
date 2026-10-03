r"""
Markov-operator theory of bivariate copulas.

The *Markov product* (Darsow, Nguyen & Olsen 1992)

.. math::

   (A * B)(u, v) = \int_0^1 \partial_2 A(u,t)\,\partial_1 B(t,v)\,dt

turns the bivariate copulas into a monoid with neutral element :math:`M`
and null element :math:`\Pi` (:math:`\Pi*C = C*\Pi = \Pi`,
:math:`M*C = C*M = C`); :math:`W*C(u,v) = v - C(1-u, v)` and
:math:`C*W(u,v) = u - C(u, 1-v)`, in particular :math:`W*W = M`.  The
product is associative, bilinear and satisfies
:math:`(A*B)^\top = B^\top*A^\top`.  If :math:`C` is the copula of
:math:`(X_0, X_1)` and :math:`D` that of :math:`(X_1, X_2)` for a Markov
chain, :math:`C*D` is the copula of :math:`(X_0, X_2)` (Chapman--Kolmogorov).

Each copula defines a *Markov operator* on :math:`L^1([0,1])`
(Olsen, Darsow & Nguyen 1996; Durante & Sempi 2016)

.. math::

   (T_C f)(x) = \frac{d}{dx}\int_0^1 \partial_2 C(x,t)\,f(t)\,dt
              = \int_0^1 f(y)\,K_C(x, dy)
              = \mathbb E\bigl[f(V)\mid U = x\bigr]

for a.e. :math:`x`, with the Markov kernel
:math:`K_C(x,[0,y]) = \partial_1 C(x,y)`.  :math:`T_C` is linear, positive,
:math:`T_C 1 = 1`, preserves the Lebesgue integral, its adjoint is
:math:`T_{C^\top}` and :math:`T_{A*B} = T_A\,T_B`.

Invertibility (Darsow, Nguyen & Olsen 1992): :math:`C^\top` is a left
inverse of :math:`C`, :math:`C^\top * C = M`, iff for every :math:`v`,
:math:`\partial_1 C(u,v)\in\{0,1\}` for a.e. :math:`u`, i.e. iff :math:`V`
is a measurable function of :math:`U` (*complete dependence*); a copula has
a left inverse iff this holds, and the left inverse is then :math:`C^\top`.
Proof sketch: :math:`(C^\top*C)(u,u)=\int_0^1(\partial_1C(t,u))^2\,dt` equals
:math:`u = M(u,u)` iff :math:`\int_0^1\partial_1C(1-\partial_1C)\,dt = 0`.
Integrating once more,
:math:`\int\!\!\int\partial_1C(1-\partial_1C) = (1-\xi(C))/6` with
Chatterjee's :math:`\xi`, so :math:`C` is left invertible iff
:math:`\xi(C)=1` (cf. Dette, Siburg & Stoimenov 2013; Chatterjee 2021).
Right invertibility (:math:`C*C^\top=M`) is the same for :math:`\partial_2C`
(:math:`U` a function of :math:`V`); copulas with both are the
*mutually completely dependent* ones, e.g. all shuffles of :math:`M`.

References
----------
* Darsow, W. F., Nguyen, B. & Olsen, E. T. (1992). Copulas and Markov
  processes. *Illinois J. Math.* 36, 600--642.
* Olsen, E. T., Darsow, W. F. & Nguyen, B. (1996). Copulas and Markov
  operators. In *Distributions with Fixed Marginals and Related Topics*,
  IMS Lecture Notes 28, 244--259.
* Darsow, W. F. & Olsen, E. T. (2010). Characterization of idempotent
  2-copulas. *Note Mat.* 30, 147--177.
* Durante, F. & Sempi, C. (2016). *Principles of Copula Theory*. CRC Press.
* Mikusiński, P., Sherwood, H. & Taylor, M. D. (1992). Shuffles of Min.
  *Stochastica* 13, 61--74.
* Dette, H., Siburg, K. F. & Stoimenov, P. A. (2013). A copula-based
  non-parametric measure of regression dependence. *Scand. J. Stat.* 40,
  21--41.
* Trutschnig, W. (2011). On a strong metric on the space of copulas and its
  induced dependence measure. *J. Math. Anal. Appl.* 384, 690--705.

Examples
--------
>>> import copul as cp
>>> from copul.theory.markov import is_left_invertible, markov_operator
>>> bool(is_left_invertible(cp.ShuffleOfMin([2, 3, 1])))
True
>>> bool(is_left_invertible(cp.Clayton(2)))
False
>>> T = markov_operator(cp.FarlieGumbelMorgenstern(0.6))
>>> round(float(T(lambda y: y)(0.0)), 10)  # E[V | U = 0] = 1/2 - theta/6
0.4
"""

from __future__ import annotations

import numpy as np

from copul.star_product import markov_product
from copul.theory.dependence import PropertyResult

__all__ = [
    "MarkovOperator",
    "conditional_expectation",
    "conditional_quantile",
    "is_completely_dependent",
    "is_idempotent",
    "is_invertible",
    "is_left_invertible",
    "is_mutually_completely_dependent",
    "is_right_invertible",
    "markov_kernel",
    "markov_operator",
    "markov_power",
    "markov_product",
    "markov_product_cdf",
    "regression_function",
    "transpose",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _backend(C):
    from copul.measures.backend import numeric_backend

    return numeric_backend(C)


def _pi_matrix(C):
    from copul.theory.distances import _pi_checkerboard_matrix

    return _pi_checkerboard_matrix(C)


def transpose(C):
    r"""Transposed copula :math:`C^\top(u,v) = C(v,u)` (the law of :math:`(V,U)`).

    Checkerboards (transposed mass and sign matrices), straight shuffles of
    :math:`M` (inverse permutation) and the symmetric :math:`M`, :math:`W`,
    :math:`\Pi` are transposed exactly within their class; any other copula
    is wrapped by :func:`copul.family.constructions.transpose`.
    """
    from copul.checkerboard._biv_mixin import BivCheckerboardMixin
    from copul.checkerboard.biv_check_min import BivCheckMin
    from copul.checkerboard.biv_check_mixed import BivCheckMixed
    from copul.checkerboard.biv_check_pi import BivCheckPi
    from copul.checkerboard.biv_check_w import BivCheckW
    from copul.checkerboard.shuffle_min import ShuffleOfMin
    from copul.family.constructions import transpose as _transpose
    from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
    from copul.family.frechet.lower_frechet import LowerFrechet
    from copul.family.frechet.upper_frechet import UpperFrechet
    from copul.family.other.independence_copula import IndependenceCopula

    if isinstance(C, (UpperFrechet, LowerFrechet, BivIndependenceCopula, IndependenceCopula)):
        return C
    if isinstance(C, ShuffleOfMin):
        return ShuffleOfMin(np.argsort(C.pi0) + 1)
    if isinstance(C, BivCheckerboardMixin):
        P = np.asarray(C.matr, dtype=float).T
        if isinstance(C, BivCheckMixed):
            return BivCheckMixed(P, sign=np.asarray(C.sign).T)
        for cls in (BivCheckMin, BivCheckW):
            if type(C) is cls:
                return cls(P)
        if type(C) is BivCheckPi:
            return BivCheckPi(P)
    return _transpose(C)


def conditional_quantile(C):
    r"""Conditional quantile function :math:`(u, w)\mapsto\partial_1C(u,\cdot)^{-1}(w)`.

    The :math:`w`-quantile of :math:`V` given :math:`U=u` (vectorized), i.e.
    ``cond_distr_1_inv``; exact for :math:`M` (:math:`u`), :math:`W`
    (:math:`1-u`) and :math:`\Pi` (:math:`w`).
    """
    from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
    from copul.family.frechet.lower_frechet import LowerFrechet
    from copul.family.frechet.upper_frechet import UpperFrechet
    from copul.family.other.independence_copula import IndependenceCopula

    if isinstance(C, UpperFrechet):
        return lambda u, w: np.asarray(u, float) + 0.0 * np.asarray(w, float)
    if isinstance(C, LowerFrechet):
        return lambda u, w: 1.0 - np.asarray(u, float) + 0.0 * np.asarray(w, float)
    if isinstance(C, (BivIndependenceCopula, IndependenceCopula)):
        return lambda u, w: np.asarray(w, float) + 0.0 * np.asarray(u, float)
    return _backend(C).h1_inv


def markov_kernel(C):
    r"""The Markov kernel :math:`(x, y)\mapsto K_C(x,[0,y]) = \partial_1 C(x,y)`.

    Returns a vectorized callable (``P(V <= y | U = x)``).
    """
    h1 = _backend(C).get("h1")

    def kernel(x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        with np.errstate(all="ignore"):
            out = np.asarray(h1(np.clip(x, 0, 1), np.clip(y, 0, 1)), dtype=float)
        out = np.where(y <= 0, 0.0, np.where(y >= 1, 1.0, out))
        return np.clip(out, 0.0, 1.0)

    return kernel


def markov_product_cdf(A, B, u, v, *, rtol: float = 1e-9, atol: float = 1e-12):
    r"""Evaluate :math:`(A*B)(u,v)=\int_0^1\partial_2A(u,t)\,\partial_1B(t,v)\,dt`.

    Unlike :func:`copul.star_product.markov_product` (which returns a
    checkerboard approximation on a grid), this evaluates the product at
    arbitrary points by vectorized adaptive Gauss--Kronrod quadrature with
    jump detection (the integrand may be discontinuous for copulas with
    singular components).  Exact for two independence-kernel checkerboards.

    Parameters
    ----------
    A, B : bivariate copulas
    u, v : array_like
        Broadcastable evaluation points.
    rtol, atol : float
        Quadrature tolerances.

    Returns
    -------
    numpy.ndarray or float
    """
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    scalar = u.ndim == 0
    PA, PB = _pi_matrix(A), _pi_matrix(B)
    if PA is not None and PB is not None:
        prod = markov_product(_as_checkpi(PA), _as_checkpi(PB))
        out = np.asarray(prod.cdf(u.ravel(), v.ravel()), float).reshape(u.shape)
        return float(out) if scalar else out
    from copul.measures.quadrature import integrate_1d_batch
    from copul.theory.distances import _breaks

    bea, beb = _backend(A), _backend(B)
    h2a, h1b = bea.get("h2"), beb.get("h1")
    uf, vf = np.clip(u.ravel(), 0, 1), np.clip(v.ravel(), 0, 1)
    _, ta = _breaks(bea)
    tb, _ = _breaks(beb)
    tbrk = [b for b in (ta, tb) if b is not None]
    brk = np.unique(np.concatenate(tbrk)) if tbrk else None

    def f(t, rows):
        with np.errstate(all="ignore"):
            a = np.asarray(h2a(uf[rows], t), float)
            b = np.asarray(h1b(t, vf[rows]), float)
        return np.clip(a, 0, 1) * np.clip(b, 0, 1)

    val, _ = integrate_1d_batch(
        f, np.zeros(uf.size), np.ones(uf.size), atol=atol, rtol=rtol, jump_search=True, breaks=brk
    )
    out = np.asarray(val, float)
    # exact boundary values of a copula
    out = np.where((uf <= 0) | (vf <= 0), 0.0, out)
    out = np.where(uf >= 1, vf, np.where(vf >= 1, uf, out))
    out = np.clip(out, np.maximum(uf + vf - 1, 0), np.minimum(uf, vf)).reshape(u.shape)
    return float(out) if scalar else out


def _as_checkpi(P):
    from copul.checkerboard.biv_check_pi import BivCheckPi

    return BivCheckPi(P)


def markov_power(C, n: int, **kwargs):
    r"""The :math:`n`-fold Markov product :math:`C^{*n} = C * \cdots * C`.

    :math:`C^{*0} = M`, :math:`C^{*1} = C`; higher powers by repeated
    squaring with :func:`copul.star_product.markov_product` (exact for
    independence-kernel checkerboards, a checkerboard approximation
    otherwise; ``kwargs`` are passed on).  :math:`C^{*n}` is the copula of
    :math:`(X_0, X_n)` of a stationary Markov chain with transition copula
    :math:`C`.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        from copul.family.frechet.upper_frechet import UpperFrechet

        return UpperFrechet()
    result = None
    base = C
    while n:
        if n & 1:
            result = base if result is None else markov_product(result, base, **kwargs)
        n >>= 1
        if n:
            base = markov_product(base, base, **kwargs)
    return result


# ---------------------------------------------------------------------------
# idempotents and invertibility
# ---------------------------------------------------------------------------


def is_idempotent(C, *, tol: float | None = None, n_grid: int = 17) -> PropertyResult:
    r"""Whether :math:`C*C = C`.

    Exact for independence-kernel checkerboards (matrix identity
    :math:`P\,M\,P = P` of :func:`copul.star_product.markov_product`);
    otherwise :math:`\sup|C*C - C|` is evaluated on an ``n_grid x n_grid``
    grid with :func:`markov_product_cdf` and compared with ``tol`` (default
    :math:`10^{-6}`).  Examples of idempotents are :math:`M`, :math:`\Pi`
    and ordinal sums of copies of :math:`\Pi` and :math:`M` (Darsow & Olsen
    2010); :math:`W` is not (:math:`W*W=M`).
    """
    P = _pi_matrix(C)
    if P is not None:
        Q = np.asarray(markov_product(_as_checkpi(P), _as_checkpi(P)).matr, float)
        Q = Q / Q.sum()
        d = float(np.max(np.abs(Q - P)))
        t = 1e-12 if tol is None else tol
        return PropertyResult(
            "idempotent",
            d <= t,
            "exact",
            d,
            None,
            "independence-kernel checkerboard: C*C has mass matrix P M P",
        )
    t = 1e-6 if tol is None else tol
    x = (np.arange(n_grid) + 0.5) / n_grid
    U, V = np.meshgrid(x, x, indexing="ij")
    cc = markov_product_cdf(C, C, U, V)
    c = np.asarray(_backend(C).cdf(U, V), float)
    diff = np.abs(cc - c)
    k = int(np.argmax(diff))
    d = float(diff.flat[k])
    return PropertyResult(
        "idempotent",
        d <= t,
        "numeric",
        d,
        {"u": float(U.flat[k]), "v": float(V.flat[k])},
        "sup |C*C - C| on a grid (adaptive quadrature of the Markov product)",
        {"tol": t, "n_grid": n_grid},
    )


def _xi_result(C, key, name, tol, reason):
    from copul.checkerboard.shuffle_min import ShuffleOfMin
    from copul.family.frechet.lower_frechet import LowerFrechet
    from copul.family.frechet.upper_frechet import UpperFrechet
    from copul.measures.engine import compute

    if isinstance(C, (UpperFrechet, LowerFrechet, ShuffleOfMin)):
        return PropertyResult(
            name,
            True,
            "exact",
            0.0,
            None,
            "M, W and shuffles of M are mutually completely dependent "
            "(Mikusinski, Sherwood & Taylor 1992)",
            {"xi": 1.0, "tol": tol},
        )
    res = compute(C, key, full_output=True)
    xi = float(res.value)
    deficiency = max(1.0 - xi, 0.0)
    method = "exact" if res.method == "closed" or res.error == 0.0 else "numeric"
    return PropertyResult(
        name,
        deficiency <= tol,
        method,
        deficiency,
        None,
        reason,
        {"xi": xi, "tol": tol, "xi_method": res.method, "xi_error": res.error},
    )


def _product_result(C, name, left, tol, n_grid):
    T = transpose(C)
    A, B = (T, C) if left else (C, T)
    x = (np.arange(n_grid) + 0.5) / n_grid
    d = np.abs(markov_product_cdf(A, B, x, x) - x)
    k = int(np.argmax(d))
    prod = "C^T * C" if left else "C * C^T"
    return PropertyResult(
        name,
        float(d[k]) <= tol,
        "numeric",
        float(d[k]),
        {"u": float(x[k]), "v": float(x[k])},
        f"sup_u |({prod})(u,u) - u| on a grid; ({prod}) = M iff its diagonal is the identity",
        {"tol": tol, "n_grid": n_grid},
    )


def is_left_invertible(
    C, *, tol: float | None = None, method: str = "xi", n_grid: int = 33
) -> PropertyResult:
    r"""Whether :math:`C^\top * C = M` (:math:`C` has a left inverse).

    Equivalent to :math:`\partial_1 C(u,v)\in\{0,1\}` a.e., i.e. to complete
    dependence of :math:`V` on :math:`U` (Darsow, Nguyen & Olsen 1992), and
    to :math:`\xi(C) = 1` (see the module docstring).

    Parameters
    ----------
    method : {"xi", "product"}
        ``"xi"`` (default) tests :math:`1-\xi(C)\le` ``tol`` (default
        :math:`10^{-6}`; exact when :math:`\xi` has a closed form, e.g.
        checkerboards and shuffles of :math:`M`); ``"product"`` evaluates
        :math:`\sup_u|(C^\top*C)(u,u)-u|` by quadrature (default ``tol``
        :math:`10^{-6}`).

    Returns
    -------
    PropertyResult
        ``worst_violation`` is :math:`1-\xi(C) = 6\int\!\!\int\partial_1C\,
        (1-\partial_1C)` (``"xi"``) or the diagonal defect (``"product"``).
    """
    t = 1e-6 if tol is None else tol
    if method == "product":
        return _product_result(C, "left_invertible", True, t, n_grid)
    return _xi_result(
        C,
        "xi",
        "left_invertible",
        t,
        "C^T * C = M iff d1 C in {0,1} a.e. iff xi(C) = 1 (Darsow, Nguyen & Olsen 1992)",
    )


def is_right_invertible(
    C, *, tol: float | None = None, method: str = "xi", n_grid: int = 33
) -> PropertyResult:
    r"""Whether :math:`C * C^\top = M`, i.e. :math:`\partial_2C\in\{0,1\}` a.e.

    Equivalent to :math:`U` being a measurable function of :math:`V` and to
    :math:`\xi_2(C) = \xi(C^\top) = 1`; see :func:`is_left_invertible`.
    """
    t = 1e-6 if tol is None else tol
    if method == "product":
        return _product_result(C, "right_invertible", False, t, n_grid)
    return _xi_result(
        C,
        "xi_2",
        "right_invertible",
        t,
        "C * C^T = M iff d2 C in {0,1} a.e. iff xi(C^T) = 1 (Darsow, Nguyen & Olsen 1992)",
    )


def is_invertible(C, **kwargs) -> PropertyResult:
    r"""Whether :math:`C` is invertible (left and right) for the Markov product.

    Equivalent to mutual complete dependence (e.g. shuffles of :math:`M`).
    """
    left = is_left_invertible(C, **kwargs)
    right = is_right_invertible(C, **kwargs)
    worst = max(left.worst_violation, right.worst_violation)
    method = "exact" if left.method == right.method == "exact" else "numeric"
    return PropertyResult(
        "invertible",
        left.holds and right.holds,
        method,
        worst,
        None,
        "left and right invertible (Darsow, Nguyen & Olsen 1992)",
        {"left": left, "right": right},
    )


def is_completely_dependent(C, i: int = 1, **kwargs) -> PropertyResult:
    r"""Complete dependence: :math:`V=f(U)` a.s. (``i=1``) or :math:`U=g(V)` (``i=2``).

    Equivalent to left (``i=1``) resp. right (``i=2``) invertibility, to
    :math:`\xi = 1` and to :math:`\zeta_1 = 1` (Trutschnig 2011).
    """
    if i == 1:
        r = is_left_invertible(C, **kwargs)
    elif i == 2:
        r = is_right_invertible(C, **kwargs)
    else:
        raise ValueError("i must be 1 or 2")
    r.property = f"completely_dependent({'V|U' if i == 1 else 'U|V'})"
    return r


def is_mutually_completely_dependent(C, **kwargs) -> PropertyResult:
    """Mutual complete dependence (a bijection links ``U`` and ``V``); see :func:`is_invertible`."""
    r = is_invertible(C, **kwargs)
    r.property = "mutually_completely_dependent"
    return r


# ---------------------------------------------------------------------------
# Markov operator
# ---------------------------------------------------------------------------


def _gauss_legendre_unit(n_panels: int, order: int):
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, 1.0, n_panels + 1)
    a, b = edges[:-1, None], edges[1:, None]
    nodes = (0.5 * (b - a) * x[None, :] + 0.5 * (a + b)).ravel()
    weights = (0.5 * (b - a) * w[None, :]).ravel()
    return nodes, weights


class MarkovOperator:
    r"""The Markov operator :math:`T_C f(x) = \mathbb E[f(V)\mid U=x]`.

    Evaluated through the conditional quantile function
    :math:`Q_x = \partial_1C(x,\cdot)^{-1}`:
    :math:`T_Cf(x) = \int_0^1 f(Q_x(w))\,dw` (composite Gauss--Legendre
    rule in :math:`w`), which is valid for every copula, including those
    with singular components.

    Parameters
    ----------
    C : bivariate copula
        Fully specified copula.
    n_panels, order : int
        Composite Gauss--Legendre rule (``n_panels`` panels of ``order``
        nodes) on :math:`(0, 1)`.

    Examples
    --------
    >>> import copul as cp
    >>> T = MarkovOperator(cp.UpperFrechet())
    >>> float(T(lambda y: y**2)(0.5))
    0.25
    """

    def __init__(self, C, n_panels: int = 16, order: int = 16):
        self.copula = C
        self._q = conditional_quantile(C)
        self._w, self._wt = _gauss_legendre_unit(int(n_panels), int(order))
        self._n_panels, self._order = int(n_panels), int(order)

    def apply(self, f, x):
        r""":math:`(T_C f)(x)` for a vectorized function ``f`` at points ``x``."""
        x = np.asarray(x, dtype=float)
        shape = x.shape
        xf = np.clip(x.ravel(), 0.0, 1.0)
        X = np.repeat(xf[:, None], self._w.size, axis=1)
        W = np.broadcast_to(self._w[None, :], X.shape)
        with np.errstate(all="ignore"):
            q = np.asarray(self._q(X.ravel(), W.ravel()), dtype=float).reshape(X.shape)
            fq = np.asarray(f(np.clip(q, 0.0, 1.0)), dtype=float)
        out = np.broadcast_to(fq, X.shape) @ self._wt
        return float(out[0]) if shape == () else out.reshape(shape)

    def __call__(self, f):
        """The function :math:`T_C f` as a vectorized callable."""

        def Tf(x):
            return self.apply(f, x)

        return Tf

    @property
    def adjoint(self) -> MarkovOperator:
        r"""The adjoint operator :math:`T_C^* = T_{C^\top}`."""
        return MarkovOperator(transpose(self.copula), self._n_panels, self._order)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MarkovOperator({self.copula})"


def markov_operator(C, **kwargs) -> MarkovOperator:
    """The Markov operator of ``C`` (see :class:`MarkovOperator`)."""
    return MarkovOperator(C, **kwargs)


def conditional_expectation(C, f, x, given: int = 1, **kwargs):
    r"""Conditional expectation :math:`\mathbb E[f(V)\mid U=x]` (``given=1``).

    With ``given=2`` returns :math:`\mathbb E[f(U)\mid V=x]` (the Markov
    operator of :math:`C^\top`).  ``f`` must be vectorized.
    """
    if given == 1:
        return MarkovOperator(C, **kwargs).apply(f, x)
    if given == 2:
        return MarkovOperator(transpose(C), **kwargs).apply(f, x)
    raise ValueError("given must be 1 or 2")


def regression_function(C, x, given: int = 1, **kwargs):
    r"""Regression function :math:`x\mapsto\mathbb E[V\mid U=x]` (``given=1``).

    The conditional mean in copula scale; :math:`\mathbb E[U\mid V=x]` for
    ``given=2``.
    """
    return conditional_expectation(C, lambda y: y, x, given=given, **kwargs)
