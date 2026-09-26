r"""
Exact, vectorised numerics shared by all bivariate checkerboard copulas.

A bivariate checkerboard copula on an :math:`m\times n` grid is described by a
nonnegative mass matrix :math:`\Delta` (normalised to total mass one) and a
*kernel sign matrix* :math:`S\in\{-1,0,1\}^{m\times n}` choosing the copula
that distributes the mass inside every cell:

* ``0``  -- independence :math:`\Pi` (uniform mass on the rectangle),
* ``+1`` -- comonotone :math:`M` (mass on the rising cell diagonal),
* ``-1`` -- countermonotone :math:`W` (mass on the falling cell diagonal).

``BivCheckPi`` is ``S = 0``, ``BivCheckMin`` is ``S = 1``, ``BivCheckW`` is
``S = -1`` and ``BivCheckMixed`` allows an arbitrary ``S``.

With cell-local coordinates :math:`a = mu - i`, :math:`b = nv - j` (clipped to
``[0, 1]``) the distribution function is

.. math::

   C(u,v) = \sum_{i,j} \Delta_{ij}\, K_{S_{ij}}(a_i(u), b_j(v)),
   \qquad K_0 = ab,\; K_1=\min(a,b),\; K_{-1} = (a+b-1)^+ .

Since :math:`D_s := K_s - K_0` vanishes on the boundary of the unit square,
:math:`C = C_\Pi + \Delta_{i^*j^*} D_{s^*}(a^*, b^*)` where :math:`C_\Pi` is the
bilinear interpolation of the cumulated mass table and :math:`(i^*, j^*)` is
the cell containing :math:`(u,v)`.  All evaluations below are therefore
:math:`O(1)` per point.

Closed forms (0-based indices, :math:`\Delta` normalised to total mass 1)::

    rho  = rho_Pi  + sum_ij S_ij Delta_ij / (m n)
    tau  = tau_Pi  + sum_ij S_ij Delta_ij^2
    xi   = xi_Pi   + (m / n) sum_ij |S_ij| Delta_ij^2        (condition on U)
    nu   = nu_Pi   + sum_ij S_ij Delta_ij (2m - 2i - 1) / (m^2 n)

The Blest add-on follows from
:math:`\int\!\!\int_{[0,1]^2} (1-a)\,D_s(a,b)\,da\,db = s/24` (both :math:`D_1`
and :math:`D_{-1}` are invariant under :math:`(a,b)\mapsto(1-a,1-b)` and
:math:`\int\!\!\int D_s = s/12`), weighted by
:math:`1-u = (m-i-a)/m` and the cell area :math:`1/(mn)`.

Spearman's footrule and Gini's gamma need :math:`\int_0^1 C(t,t)\,dt` and
:math:`\int_0^1 C(t,1-t)\,dt`.  For square grids only the (anti-)diagonal
cells contribute an add-on (``int_0^1 D_1(s,s) ds = 1/6``,
``int_0^1 D_{-1}(s,s) ds = -1/12``, ``int_0^1 D_1(s,1-s) ds = 1/12``,
``int_0^1 D_{-1}(s,1-s) ds = -1/6``).  For general :math:`m\times n` grids the
integrands are piecewise quadratic polynomials in :math:`t` with breakpoints
contained in :math:`\{k/m\}\cup\{l/n\}\cup\{k/(m+n)\}\cup\{k/|m-n|\}`
(cell edges and the kinks of ``min``/``max`` inside a cell); Gauss--Legendre
quadrature with three nodes per piece is therefore *exact*.
"""

from __future__ import annotations

import warnings

import numpy as np

_TOL = 1e-12


# ---------------------------------------------------------------------------
# random number generation
# ---------------------------------------------------------------------------
def resolve_rng(random_state=None):
    """Return a random generator without ever touching the global seed.

    * ``None`` -> NumPy's global legacy ``RandomState`` (so that
      ``np.random.seed`` keeps working for reproducibility), used *without*
      reseeding;
    * an existing ``Generator`` / ``RandomState`` is returned unchanged;
    * anything else (int, ``SeedSequence``, ...) -> ``np.random.default_rng``.

    Only methods shared by both generator types (``random``, ``choice``) are
    used by the callers.
    """
    if random_state is None:
        rs = getattr(np.random, "mtrand", None)
        rs = getattr(rs, "_rand", None)
        return rs if rs is not None else np.random.default_rng()
    if isinstance(random_state, (np.random.Generator, np.random.RandomState)):
        return random_state
    return np.random.default_rng(random_state)


# ---------------------------------------------------------------------------
# argument handling
# ---------------------------------------------------------------------------
def parse_uv(args, kwargs):
    """Normalise the call conventions of bivariate evaluation methods.

    Accepted forms: ``f(u, v)`` (scalars or broadcastable arrays),
    ``f(u=..., v=...)``, ``f([u, v])`` (a single point) and ``f(P)`` with
    ``P`` of shape ``(N, 2)``.

    Returns
    -------
    u, v : ndarray
        Broadcast float arrays.
    scalar : bool
        Whether the result should be returned as a Python float.
    """
    kwargs = dict(kwargs)
    ku = kwargs.pop("u", None)
    kv = kwargs.pop("v", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
    if ku is not None or kv is not None:
        if args or ku is None or kv is None:
            raise ValueError("Provide both u and v (either positionally or as keywords).")
        args = (ku, kv)
    if len(args) == 0:
        raise ValueError("No arguments provided")
    if len(args) == 2:
        u = np.asarray(args[0], dtype=float)
        v = np.asarray(args[1], dtype=float)
        scalar = u.ndim == 0 and v.ndim == 0
        u, v = np.broadcast_arrays(u, v)
        return u, v, scalar
    if len(args) == 1:
        arr = np.asarray(args[0], dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(f"Expected point array of length 2, got {arr.shape[0]}")
            return arr[0], arr[1], True
        if arr.ndim == 2:
            if arr.shape[1] != 2:
                raise ValueError(f"Expected points with 2 dimensions, got {arr.shape[1]}")
            return arr[:, 0], arr[:, 1], False
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array")
    raise ValueError(f"Expected 2 coordinates, got {len(args)}")


def finish(out, scalar):
    out = np.asarray(out, dtype=float)
    return float(out) if scalar else out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _signs(P, S):
    if S is None:
        return None
    S = np.asarray(S)
    if S.ndim == 0:
        if int(S) == 0:
            return None
        return np.full(P.shape, int(S), dtype=int)
    if not np.any(S):
        return None
    return S.astype(int)


def _cum_table(P):
    m, n = P.shape
    G = np.zeros((m + 1, n + 1))
    G[1:, 1:] = P.cumsum(axis=0).cumsum(axis=1)
    return G


def _locate(x, k):
    """Cell index and local coordinate in [0,1] of ``x`` on a k-grid."""
    y = np.clip(x, 0.0, 1.0) * k
    idx = np.minimum(np.floor(y).astype(np.intp), k - 1)
    return idx, y - idx


def _D(s, a, b):
    """Deviation K_s(a, b) - a*b of the cell kernel from independence."""
    ab = a * b
    return np.where(
        s == 1,
        np.minimum(a, b) - ab,
        np.where(s == -1, np.maximum(a + b - 1.0, 0.0) - ab, 0.0),
    )


# ---------------------------------------------------------------------------
# cdf / conditional distributions / density / sampling
# ---------------------------------------------------------------------------
def cdf(P, S, u, v):
    """Distribution function at broadcastable ``u, v`` (exact)."""
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    m, n = P.shape
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    G = _cum_table(P)
    i, a = _locate(u, m)
    j, b = _locate(v, n)
    out = (
        (1 - a) * (1 - b) * G[i, j]
        + a * (1 - b) * G[i + 1, j]
        + (1 - a) * b * G[i, j + 1]
        + a * b * G[i + 1, j + 1]
    )
    if S is not None:
        out = out + P[i, j] * _D(S[i, j], a, b)
    return out


def cond_distr(P, S, which, u, v):
    r"""Conditional distribution function (exact, vectorised).

    ``which=1``: :math:`P(V\le v\mid U=u)=\partial_1 C(u,v)`;
    ``which=2``: :math:`P(U\le u\mid V=v)=\partial_2 C(u,v)`.

    The conditional law inside the row (column) slice is normalised by the
    slice mass, so that for non-copula mass matrices the result is still a
    proper conditional distribution (for genuine copulas this coincides with
    the partial derivative).  On the singular lines of ``Min``/``W`` cells the
    right-continuous version is returned.
    """
    if which not in (1, 2):
        raise ValueError(f"Dimension {which} out of range 1..2")
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    if which == 2:
        P = P.T
        S = None if S is None else S.T
        u, v = v, u
    m, n = P.shape
    i, a = _locate(u, m)
    j, b = _locate(v, n)
    Rc = np.zeros((m, n + 1))
    Rc[:, 1:] = np.cumsum(P, axis=1)
    denom = Rc[i, n]
    if S is None:
        k = b
    else:
        s = S[i, j]
        pos = b > 0
        k = np.where(
            s == 0,
            b,
            np.where(
                s == 1,
                (pos & (b >= a - _TOL)).astype(float),
                (pos & (a + b >= 1.0 - _TOL)).astype(float),
            ),
        )
    num = Rc[i, j] + P[i, j] * k
    out = np.divide(num, denom, out=np.zeros(np.shape(num)), where=denom > 0)
    return np.where(u < 0, 0.0, out)


def pdf_pi(P, u, v):
    """Density of the independence-kernel checkerboard."""
    P = np.asarray(P, dtype=float)
    m, n = P.shape
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    i, _ = _locate(u, m)
    j, _ = _locate(v, n)
    inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    return np.where(inside, P[i, j] * (m * n), 0.0)


def rvs(P, S, n, random_state=None):
    """Draw ``n`` samples (cell by mass, then location according to kernel)."""
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    rng = resolve_rng(random_state)
    m, k = P.shape
    flat = P.ravel()
    total = flat.sum()
    if total <= 0:
        raise ValueError("Matrix contains no positive values, cannot sample")
    idx = rng.choice(flat.size, size=int(n), p=flat / total)
    i, j = np.divmod(idx, k)
    t1 = rng.random(int(n))
    t2 = rng.random(int(n))
    if S is not None:
        s = S[i, j]
        t2 = np.where(s == 1, t1, np.where(s == -1, 1.0 - t1, t2))
    return np.column_stack(((i + t1) / m, (j + t2) / k))


# ---------------------------------------------------------------------------
# closed-form dependence measures
# ---------------------------------------------------------------------------
def spearmans_rho(P, S=None):
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    m, n = P.shape
    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    omega = (2 * m - 2 * i - 1) * (2 * n - 2 * j - 1) / (m * n)
    rho = 3.0 * np.sum(omega * P) - 3.0
    if S is not None:
        rho += np.sum(S * P) / (m * n)
    return float(rho)


def kendalls_tau(P, S=None):
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    m, n = P.shape
    Xi_m = 2 * np.tri(m) - np.eye(m)
    Xi_n = 2 * np.tri(n) - np.eye(n)
    tau = 1.0 - np.trace(Xi_m @ P @ Xi_n @ P.T)
    if S is not None:
        tau += np.sum(S * P**2)
    return float(tau)


def chatterjees_xi(P, S=None, condition_on_y=False):
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    delta = P.T if condition_on_y else P
    m, n = delta.shape
    T = np.ones((n, n)) - np.tri(n)
    M = T @ T.T + T.T + np.eye(n) / 3.0
    xi = 6.0 * m / n * np.trace(delta.T @ delta @ M) - 2.0
    if S is not None:
        xi += (m / n) * np.sum(np.abs(S) * P**2)
    return float(xi)


def blests_nu(P, S=None):
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    m, n = P.shape
    Lm = np.tri(m, m, k=-1)
    Ln = np.tri(n, n, k=-1)
    E = np.ones((m, n))
    w = np.arange(m, 0, -1, dtype=float)
    U = w[:, None] * np.ones((1, n))
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
    nu = (24.0 / (m * m * n)) * np.sum(P * K) - 2.0
    if S is not None:
        i = np.arange(m)[:, None]
        nu += np.sum(S * P * (2 * m - 2 * i - 1)) / (m * m * n)
    return float(nu)


def _diag_integrals(P, S):
    """Exact (int_0^1 C(t,t) dt, int_0^1 C(t,1-t) dt)."""
    P = np.asarray(P, dtype=float)
    m, n = P.shape
    parts = [np.arange(m + 1) / m, np.arange(n + 1) / n]
    parts.append(np.arange(m + n + 1) / (m + n))
    if m != n:
        d = abs(m - n)
        parts.append(np.arange(d + 1) / d)
    bps = np.unique(np.clip(np.concatenate(parts), 0.0, 1.0))
    x, w = np.polynomial.legendre.leggauss(3)
    lo, hi = bps[:-1], bps[1:]
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    t = mid[:, None] + half[:, None] * x[None, :]
    W = half[:, None] * w[None, :]
    i_diag = float(np.sum(W * cdf(P, S, t, t)))
    i_anti = float(np.sum(W * cdf(P, S, t, 1.0 - t)))
    return i_diag, i_anti


def _square_diag_integrals(P, S):
    """Closed form of the two diagonal integrals for square grids."""
    n = P.shape[0]
    J = np.fliplr(np.eye(n))
    L = np.tri(n)
    H = J @ (L @ L.T) @ J
    Wd = (H - 0.5 * np.ones((n, n)) - np.eye(n) / 6.0) / n
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    K = np.maximum(0, n - 1 - (i + j))
    Wa = (K + J / 6.0) / n
    i_diag = float(np.sum(Wd * P))
    i_anti = float(np.sum(Wa * P))
    if S is not None:
        dS = np.diag(S)
        dP = np.diag(P)
        c_d = np.where(dS == 1, 1.0 / 6.0, np.where(dS == -1, -1.0 / 12.0, 0.0))
        i_diag += float(np.sum(c_d * dP)) / n
        aS = np.diag(np.fliplr(S))
        aP = np.diag(np.fliplr(P))
        c_a = np.where(aS == 1, 1.0 / 12.0, np.where(aS == -1, -1.0 / 6.0, 0.0))
        i_anti += float(np.sum(c_a * aP)) / n
    return i_diag, i_anti


def diag_integrals(P, S=None):
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    if P.shape[0] == P.shape[1]:
        return _square_diag_integrals(P, S)
    return _diag_integrals(P, S)


def spearmans_footrule(P, S=None):
    i_diag, _ = diag_integrals(P, S)
    return 6.0 * i_diag - 2.0


def ginis_gamma(P, S=None):
    i_diag, i_anti = diag_integrals(P, S)
    return 4.0 * (i_diag + i_anti) - 2.0


def blomqvists_beta(P, S=None):
    return float(4.0 * cdf(P, S, 0.5, 0.5) - 1.0)


def lambda_L(P, S=None):
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    if S is None or S[0, 0] != 1:
        return 0.0
    return float(P[0, 0] * min(P.shape))


def lambda_U(P, S=None):
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    if S is None or S[-1, -1] != 1:
        return 0.0
    return float(P[-1, -1] * min(P.shape))


# ---------------------------------------------------------------------------
# exact dependence-property checks
# ---------------------------------------------------------------------------
def cis_direction(P, S=None, which=1, tol=1e-12):
    r"""Exact stochastic monotonicity check ``(is_SI, is_SD)``.

    ``which=1`` checks :math:`u\mapsto h(u,v)=P(V\le v\mid U=u)` for every
    ``v`` (SI: nonincreasing, SD: nondecreasing); ``which=2`` the same with
    the roles of the coordinates exchanged.

    Within row ``i`` and column ``j`` (``b = nv - j`` in ``(0,1)``) the
    conditional distribution is ``F_i(j) + p_ij k(a, b)`` with the row
    cumulative ``F_i`` and ``p_ij = Delta_ij / rowsum_i``; ``k = b`` for
    ``Pi`` cells and a 0/1 step in ``a`` for ``Min`` (nonincreasing) and
    ``W`` cells (nondecreasing).  Hence the property holds iff

    1. no cell with positive mass has the wrong within-row monotonicity
       (``W`` cells for SI, ``Min`` cells for SD), and
    2. for adjacent rows (with positive mass), the infimum over row ``i`` of
       ``h(., v)`` dominates the supremum over row ``i+1`` (SI), resp. the
       other way round (SD).  Both bounds are affine in ``b``, so checking
       ``b -> 0`` and ``b -> 1`` suffices.

    For ``BivCheckPi`` this reduces to: the row-normalised cumulative sums
    are nonincreasing down the rows at every column knot.
    """
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    if S is None:
        S = np.zeros(P.shape, dtype=int)
    if which == 2:
        P, S = P.T, S.T
    rs = P.sum(axis=1)
    keep = rs > 0
    P, S, rs = P[keep], S[keep], rs[keep]
    if P.shape[0] <= 1:
        within_ci = not np.any((S == -1) & (P > 0))
        within_cd = not np.any((S == 1) & (P > 0))
        return bool(within_ci), bool(within_cd)
    p = P / rs[:, None]
    F = np.zeros((P.shape[0], P.shape[1] + 1))
    F[:, 1:] = np.cumsum(p, axis=1)
    Fl = F[:, :-1]
    res = []
    for b in (0.0, 1.0):
        pi_part = np.where(S == 0, p * b, 0.0)
        lower = Fl + pi_part  # inf over the row
        upper = Fl + np.where(S == 0, p * b, p)  # sup over the row
        res.append((lower, upper))
    ci = not np.any((S == -1) & (p > tol))
    cd = not np.any((S == 1) & (p > tol))
    for lower, upper in res:
        if ci and np.any(lower[:-1] < upper[1:] - tol):
            ci = False
        if cd and np.any(upper[:-1] > lower[1:] + tol):
            cd = False
    return bool(ci), bool(cd)


def _tail_intercepts(P, S):
    r"""Intercepts of the linear pieces of ``u -> C(u, v)`` (see below).

    For ``v`` in column ``j`` (``b = nv - j``) and ``u`` in row ``i`` the
    map ``u -> C(u,v)`` is piecewise linear: one piece for ``Pi`` cells, two
    pieces (kink at ``a = b`` resp. ``a = 1 - b``) for ``Min``/``W`` cells.
    Each piece is anchored at a grid row ``k`` (``u = k/m``) with value
    ``G_k(b) = C(k/m, v)`` and has slope ``m * sigma(b)`` in ``u``, where
    ``G_k`` and ``sigma`` are affine in ``b``.

    Returns arrays ``(alpha_L, alpha_R)`` evaluated at ``b in {0, 1}`` with

    * ``alpha_L = G_k - k * sigma``: value of the piece's line at ``u = 0``
      (``C(u,v)/u`` is nonincreasing on the piece iff ``alpha_L >= 0``);
    * ``alpha_R = G_k + (m - k) * sigma - v``: value of the line of
      ``1 - u - v + C(u, v)`` at ``u = 1`` (the ratio
      ``(1-u-v+C)/(1-u)`` is nondecreasing iff ``alpha_R >= 0``).
    """
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    if S is None:
        S = np.zeros(P.shape, dtype=int)
    m, n = P.shape
    G = _cum_table(P)
    Rc = np.zeros((m, n + 1))
    Rc[:, 1:] = np.cumsum(P, axis=1)
    cR = Rc[:, :-1]  # (m, n): mass of row i left of column j
    rows = np.arange(m)[:, None] * np.ones((1, n))
    cols = np.ones((m, 1)) * np.arange(n)[None, :]
    aL, aR = [], []
    for b in (0.0, 1.0):
        v = (cols + b) / n
        # value of C at the grid rows, affine in b
        Gk0 = G[:-1, :-1] + b * (G[:-1, 1:] - G[:-1, :-1])  # row k = i
        Gk1 = G[1:, :-1] + b * (G[1:, 1:] - G[1:, :-1])  # row k = i + 1
        sig_pi = cR + P * b
        pieces = []
        # piece anchored at a = 0 (row k = i)
        sig0 = np.where(S == 1, cR + P, np.where(S == -1, cR, sig_pi))
        pieces.append((Gk0, rows, sig0))
        # piece anchored at a = 1 (row k = i + 1)
        sig1 = np.where(S == 1, cR, np.where(S == -1, cR + P, sig_pi))
        pieces.append((Gk1, rows + 1, sig1))
        for Gk, k, sig in pieces:
            aL.append(Gk - k * sig)
            aR.append(Gk + (m - k) * sig - v)
    return np.concatenate([x.ravel() for x in aL]), np.concatenate([x.ravel() for x in aR])


def tail_monotonicity(P, S=None, kind="ltd", tol=1e-12):
    """Exact LTD/LTI/RTI/RTD(V|U) check for checkerboard copulas."""
    aL, aR = _tail_intercepts(P, S)
    kind = kind.lower()
    if kind == "ltd":
        return bool(np.all(aL >= -tol))
    if kind == "lti":
        return bool(np.all(aL <= tol))
    if kind == "rti":
        return bool(np.all(aR >= -tol))
    if kind == "rtd":
        return bool(np.all(aR <= tol))
    raise ValueError(f"Unknown tail monotonicity property {kind!r}")


def quadrant_dependence(P, S=None, positive=True, tol=1e-12):
    r"""Exact PQD (``C >= uv``) / NQD (``C <= uv``) check.

    On every cell ``C(u,v) - uv`` is bilinear on each of the (at most two)
    triangles where the kernel is linear; its ``ab``-coefficient equals
    ``-1/(mn)`` for ``Min``/``W`` cells, so the extrema over a cell are
    attained at the cell corners, except possibly along the kink line:
    a minimum along the falling diagonal of a ``W`` cell and a maximum
    along the rising diagonal of a ``Min`` cell (both are parabolas whose
    vertex is evaluated explicitly).
    """
    P = np.asarray(P, dtype=float)
    S = _signs(P, S)
    m, n = P.shape
    G = _cum_table(P)
    uu = np.arange(m + 1)[:, None] / m
    vv = np.arange(n + 1)[None, :] / n
    diff = G - uu * vv
    if positive and np.any(diff < -tol):
        return False
    if not positive and np.any(diff > tol):
        return False
    if S is None:
        return True
    target = -1 if positive else 1
    ii, jj = np.nonzero((target == S) & (P > 0))
    if ii.size == 0:
        return True
    t = np.linspace(0.0, 1.0, 3)
    # evaluate the quadratic along the kink line at t = 0, 1/2, 1 and locate
    # its vertex
    if target == -1:
        us = (ii[:, None] + t[None, :]) / m
        vs = (jj[:, None] + 1.0 - t[None, :]) / n
    else:
        us = (ii[:, None] + t[None, :]) / m
        vs = (jj[:, None] + t[None, :]) / n
    f = cdf(P, S, us, vs) - us * vs
    f0, fh, f1 = f[:, 0], f[:, 1], f[:, 2]
    c2 = 2.0 * (f0 + f1 - 2.0 * fh)
    c1 = f1 - f0 - c2
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(np.abs(c2) > 0, -c1 / (2.0 * c2), 0.5)
    ts = np.clip(ts, 0.0, 1.0)
    if target == -1:
        us = (ii + ts) / m
        vs = (jj + 1.0 - ts) / n
    else:
        us = (ii + ts) / m
        vs = (jj + ts) / n
    f = cdf(P, S, us, vs) - us * vs
    if positive:
        return bool(np.all(f >= -tol))
    return bool(np.all(f <= tol))


def warn_deprecated(old, new):
    warnings.warn(f"{old} is deprecated; use {new} instead.", DeprecationWarning, stacklevel=3)
