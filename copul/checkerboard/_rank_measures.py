"""Cellwise rank coefficients for rectangular Pi/Min/W checkerboards.

All formulas use the normalized mass matrix P and kernel signs S (0, +1,
-1). Polynomial moments are integrated over squares, triangles, or singular
diagonals. Integrated R² and the absolute CDF distance additionally use
logarithms; singular Savage scores use the dilogarithm. No grid approximation
or adaptive quadrature is used by the closed formulas in this module.
"""

import math
from itertools import pairwise

import numpy as np
from numpy.polynomial import Polynomial
from scipy.linalg import svdvals
from scipy.special import ndtri, spence, xlogy

from copul.checkerboard import _biv_engine as eng


def _inputs(P, S=None, transpose=False):
    p = np.asarray(P, float)
    p = p / p.sum()
    s = eng._signs(p, S)
    s = np.zeros(p.shape, int) if s is None else s
    return (p.T, s.T) if transpose else (p, s)


def blests_eta(P, S=None):
    p, s = _inputs(P, S)
    return (eng.blests_nu(p, s) + eng.blests_nu(p.T, s.T)) / 2


def siburg_stoimenov(P, S=None):
    p, s = _inputs(P, S)

    def xi(p, s):
        # Center before squaring: 6*int h²-2 loses precision near independence,
        # and the subsequent square root would amplify that cancellation.
        m, n = p.shape
        a = m * (np.cumsum(p, axis=1) - p) - np.arange(n) / n
        b = m * p - 1 / n
        return 6 * np.mean((a + b / 2) ** 2 + b**2 / 12) + np.mean((m * p) ** 2 * np.abs(s))

    return math.sqrt((xi(p, s) + xi(p.T, s.T)) / 2)


def correlation_ratio(P, S=None, *, condition_on_y=False):
    """The conditional mean is affine within each conditioning interval."""
    p, s = _inputs(P, S, condition_on_y)
    m, n = p.shape
    means = m * (p @ ((np.arange(n) + 0.5) / n))
    slopes = (m / n) * np.sum(p * s, axis=1)
    return float(np.mean(12 * (means - 0.5) ** 2 + slopes**2))


def _rational_moments(j):
    """Integrals of b**k/(j+b), k=0,1,2, for positive integer j.

    These are log1p(1/j), 1-j*log1p(1/j), and
    1/2-j+j²*log1p(1/j). A convergent expansion avoids cancellation for j>=2.
    """
    if j == 1:
        l = math.log(2)
        return np.array([l, 1 - l, l - 0.5])
    k = np.arange(3)
    result = np.zeros(3)
    term = 1 / j
    for ell in range(60):
        result += term / (k + ell + 1)
        term /= -j
    return result


def integrated_r2(P, S=None, *, condition_on_y=False):
    """Integrate the quadratic conditional variance with weight 1/[v(1-v)]."""
    p, s = _inputs(P, S, condition_on_y)
    m, n = p.shape
    before = np.cumsum(p, axis=1) - p
    a = m * before - np.arange(n) / n
    b = m * p - 1 / n
    singular = (m * p) ** 2 * np.abs(s)
    q = np.array(
        [
            np.mean(a * a, axis=0),
            np.mean(2 * a * b + singular, axis=0),
            np.mean(b * b - singular, axis=0),
        ]
    )
    reverse = np.array([q.sum(axis=0), -q[1] - 2 * q[2], q[2]])[:, ::-1]
    total = 0.0
    for j in range(n):
        # n/[(j+b)(n-j-b)] = 1/(j+b) + 1/(n-j-b).
        coeff = q[:, j] + reverse[:, j]
        # At j=0 the constant vanishes by the uniform margin identities.
        total += coeff[1] + coeff[2] / 2 if j == 0 else coeff @ _rational_moments(j)
    return float(max(0.0, total))


def _cell_coefficients(p):
    """Coefficients of C-uv before adding the local kernel: x+y*a+z*b-k*a*b."""
    m, n = p.shape
    g = eng._cum_table(p)
    i, j = np.indices(p.shape)
    k = 1 / (m * n)
    x = g[:-1, :-1] - i * j * k
    y = np.cumsum(p, axis=1) - p - j * k
    z = np.cumsum(p, axis=0) - p - i * k
    return x, y, z, k


_POWERS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
_R = _POWERS[:, None, 0] + _POWERS[None, :, 0]
_T = _POWERS[:, None, 1] + _POWERS[None, :, 1]
_SQUARE = 1 / ((_R + 1) * (_T + 1))
_MIN_TRIANGLE = 1 / ((_T + 1) * (_R + _T + 2))  # b <= a
_W_TRIANGLE = np.array(
    [
        [
            math.factorial(int(r)) * math.factorial(int(t)) / math.factorial(int(r + t + 2))
            for r, t in zip(rr, tt, strict=True)
        ]
        for rr, tt in zip(_R, _T, strict=True)
    ]
)  # a+b <= 1
_LINE = 1 / (np.arange(3)[:, None] + np.arange(3)[None, :] + 1)


def _pieces(x, y, z, k, mass, sign):
    """Bilinear coefficients, moment matrix, and affine vertical bounds."""
    if sign == 0:
        yield np.array([x, y, z, mass - k]), _SQUARE, (0, 0), (1, 0)
    elif sign > 0:
        yield np.array([x, y, z + mass, -k]), _MIN_TRIANGLE, (0, 0), (0, 1)
        yield np.array([x, y + mass, z, -k]), _SQUARE - _MIN_TRIANGLE, (0, 1), (1, 0)
    else:
        yield np.array([x, y, z, -k]), _W_TRIANGLE, (0, 0), (1, -1)
        yield np.array([x - mass, y + mass, z + mass, -k]), _SQUARE - _W_TRIANGLE, (1, -1), (1, 0)


def hoeffdings_d(P, S=None):
    """Hoeffding's Lebesgue L² index Phi² = 90 int (C-uv)² du dv."""
    p, s = _inputs(P, S)
    x, y, z, k = _cell_coefficients(p)
    total = 0.0
    for ij in np.ndindex(p.shape):
        for c, gram, _, _ in _pieces(x[ij], y[ij], z[ij], k, p[ij], s[ij]):
            total += c @ gram @ c
    return float(max(0.0, 90 * k * total))


def _diagonal(x, y, z, k, mass, sign):
    return np.array([x, y + z + mass, -k]) if sign > 0 else np.array([x + z, y - z - k, k])


def hoeffding_d(P, S=None):
    """D = 30 int (C-uv)² dC, including the mass on singular diagonals."""
    p, s = _inputs(P, S)
    x, y, z, k = _cell_coefficients(p)
    total = 0.0
    for ij in np.ndindex(p.shape):
        if s[ij] == 0:
            c = np.array([x[ij], y[ij], z[ij], p[ij] - k])
            moment = c @ _SQUARE @ c
        else:
            c = _diagonal(x[ij], y[ij], z[ij], k, p[ij], s[ij])
            moment = c @ _LINE @ c
        total += p[ij] * moment
    return float(max(0.0, 30 * total))


def bergsma_dassios(P, S=None):
    return 0.4 * hoeffding_d(P, S) + (4 / 15) * hoeffdings_d(P, S)


def uniform_distance(P, S=None):
    """Extrema occur at cell corners or stationary points on singular diagonals."""
    p, s = _inputs(P, S)
    m, n = p.shape
    g = eng._cum_table(p) - np.arange(m + 1)[:, None] * np.arange(n + 1) / (m * n)
    maximum = np.max(np.abs(g))
    x, y, z, k = _cell_coefficients(p)
    for i, j in np.argwhere(s != 0):
        c = _diagonal(x[i, j], y[i, j], z[i, j], k, p[i, j], s[i, j])
        t = np.clip(-c[1] / (2 * c[2]), 0, 1)
        maximum = max(maximum, abs(c[0] + t * (c[1] + t * c[2])))
    return float(4 * maximum)


def _integral(poly, a, b):
    primitive = poly.integ()
    return primitive(b) - primitive(a)


def _rational_integral(numerator, denominator, lo, hi):
    """Integrate a polynomial divided by a nonzero affine function on an interval."""
    half = (hi - lo) / 2
    mapping = Polynomial([(lo + hi) / 2, half])
    numer = numerator(mapping).coef
    denom = denominator(mapping)
    d = denom(0)
    z = denom.coef[1] / d if denom.degree() else 0.0
    moments = np.zeros(len(numer))
    if abs(z) < 0.5:
        # Stable evaluation of the log primitive when the slope is small.
        term = 1.0
        k = np.arange(len(numer))
        for ell in range(60):
            moments += term * np.where((k + ell) % 2 == 0, 2 / (k + ell + 1), 0)
            term *= -z
    else:
        # If the affine denominator vanishes at an endpoint, the numerator
        # has a double zero there. Polynomial division removes the 0*log(0).
        if abs(z) >= 1 - 1e-12:
            quotient, remainder = divmod(Polynomial(numer), Polynomial([1, z]))
            scale = max(np.max(np.abs(numer)), np.finfo(float).tiny)
            if np.max(np.abs(remainder.coef)) <= 1e-12 * scale:
                return float(half / abs(d) * _integral(quotient, -1, 1))
        z = np.clip(z, -np.nextafter(1.0, 0.0), np.nextafter(1.0, 0.0))
        moments[0] = (np.log1p(z) - np.log1p(-z)) / z
        for k in range(1, len(numer)):
            moments[k] = ((2 / k if k % 2 else 0) - moments[k - 1]) / z
    return float(half / abs(d) * (numer @ moments))


def _absolute_piece(c, lower, upper):
    """Integrate |A(a)+B(a)b| over affine vertical bounds, using logarithms."""
    a, b = Polynomial(c[:2]), Polynomial(c[2:])
    low, high = Polynomial(lower), Polynomial(upper)
    f_low, f_high = a + b * low, a + b * high
    cuts = [0.0, 1.0]
    for poly in (f_low, f_high, b):
        cuts.extend(float(r.real) for r in poly.roots() if abs(r.imag) < 1e-12 and 0 < r.real < 1)
    cuts = sorted(set(cuts))
    area = a * (high - low) + b * (high**2 - low**2) / 2
    numerator = (f_low**2 + f_high**2) / 2
    result = 0.0
    for lo, hi in pairwise(cuts):
        mid = (lo + hi) / 2
        fl, fh = f_low(mid), f_high(mid)
        if fl * fh >= 0:
            result += np.sign(fl + fh) * _integral(area, lo, hi)
        else:
            result += _rational_integral(numerator, b, lo, hi)
    return max(0.0, result)


def schweizer_wolff_sigma(P, S=None):
    """12 int |C-uv|, splitting at the zero curves of each bilinear piece."""
    p, s = _inputs(P, S)
    x, y, z, k = _cell_coefficients(p)
    total = 0.0
    for ij in np.ndindex(p.shape):
        for c, _, low, high in _pieces(x[ij], y[ij], z[ij], k, p[ij], s[ij]):
            total += _absolute_piece(c, low, high)
    return float(12 * k * total)


def score_means(n, normal):
    t = np.arange(n + 1) / n
    if normal:
        primitive = -np.exp(-(ndtri(t) ** 2) / 2) / np.sqrt(2 * np.pi)
    else:
        primitive = xlogy(1 - t, 1 - t) - (1 - t)
    return n * np.diff(primitive)


def _normal_second_moments(n):
    t = np.arange(n + 1) / n
    q = ndtri(t)
    product = np.zeros_like(q)
    finite = np.isfinite(q)
    product[finite] = q[finite] * np.exp(-(q[finite] ** 2) / 2) / np.sqrt(2 * np.pi)
    return n * np.diff(t - product)


def normal_score_correlation(P, S=None):
    """Closed for Pi cells and singular cells lying on the global (anti-)diagonal."""
    p, s = _inputs(P, S)
    m, n = p.shape
    a, b = score_means(m, True), score_means(n, True)
    value = np.sum(p * a[:, None] * b[None, :] * (s == 0))
    second = _normal_second_moments(m)
    for i, j in np.argwhere((s != 0) & (p > 0)):
        if m != n or (s[i, j] > 0 and i != j) or (s[i, j] < 0 and i + j != n - 1):
            raise NotImplementedError(
                "Normal scores on off-diagonal singular cells require numerical integration"
            )
        value += p[i, j] * s[i, j] * second[i]
    return float(value)


def _log_product_min(a, b):
    """int_0^1 log(a-t) log(b-t) dt, a,b positive integers."""
    a, b = min(a, b), max(a, b)
    d = b - a

    def primitive(x):
        if d == 0:
            return 0.0 if x == 0 else x * (np.log(x) ** 2 - 2 * np.log(x) + 2)
        if x == 0:
            return -d * np.log(d)
        lx, ly = np.log(x), np.log(x + d)
        return (
            x * lx * ly
            - (x + d) * ly
            - x * lx
            + 2 * x
            + d * (lx * np.log1p(x / d) + spence(1 + x / d))
        )

    return primitive(a) - primitive(a - 1)


def _log_product_w(a, b):
    """int_0^1 log(a-t) log(b-1+t) dt, a,b positive integers."""
    length = a + b - 1
    log_length = np.log(length)

    def primitive(x):
        z = x / length
        zlz, omlom = xlogy(z, z), xlogy(1 - z, 1 - z)
        cross = 0.0 if z == 0 or z == 1 else zlz * np.log1p(-z)
        g = zlz - omlom - 2 * z
        h = cross + omlom - zlz + 2 * z + spence(z)
        return length * (z * log_length**2 + log_length * g + h)

    return primitive(a) - primitive(a - 1)


def savage_score_correlation(P, S=None):
    """Exact cell score products, including Min/W diagonals (dilogarithms)."""
    p, s = _inputs(P, S)
    m, n = p.shape
    a, b = score_means(m, False), score_means(n, False)
    value = np.sum(p * a[:, None] * b[None, :] * (s == 0))
    for i, j in np.argwhere((s != 0) & (p > 0)):
        product = _log_product_min(m - i, n - j) if s[i, j] > 0 else _log_product_w(m - i, n - j)
        # int log(m-i-t) = log(m) - mean(J(U) | cell i).
        product += np.log(m) * b[j] + np.log(n) * a[i] - np.log(m) * np.log(n)
        value += p[i, j] * product
    return float(value - 1)


def partition_correlation(p):
    """Norm of the centered conditional-expectation matrix of finite categories."""
    rows, cols = p.sum(1), p.sum(0)
    p = p[np.ix_(rows > 0, cols > 0)]
    rows, cols = rows[rows > 0], cols[cols > 0]
    root = np.sqrt(rows[:, None] * cols[None, :])
    return float(np.clip(svdvals(p / root - root)[0], 0, 1))


def maximal_correlation(P, S=None):
    """Finite operator norms on cell constants and centered local even functions.

    Min and W both act as the identity on local even functions, with matrix
    sqrt(m*n)*P*abs(S). The odd block is bounded by that nonnegative matrix.
    Pi annihilates local centered functions. The two subspaces are orthogonal.
    """
    p, s = _inputs(P, S)
    m, n = p.shape
    local = svdvals(np.sqrt(m * n) * p * np.abs(s))[0]
    return float(np.clip(max(partition_correlation(p), local), 0, 1))


def hellinger_from_affinity(affinity):
    b = float(np.clip(affinity, 0, 1))
    return math.sqrt(4 * (1 - b) * (1 + b) * (1 + b * b) / (2 - b**4 + math.sqrt(4 - 3 * b**4)))


def linfoot_correlation(P, S=None):
    p, s = _inputs(P, S)
    if np.any((s != 0) & (p > 0)):
        return 1.0
    density = p.size * p
    information = float(np.mean(xlogy(density, density)))
    return math.sqrt(-math.expm1(-2 * max(0.0, information)))


def hellinger_correlation(P, S=None):
    p, s = _inputs(P, S)
    return hellinger_from_affinity(np.mean(np.sqrt(p.size * p * (s == 0))))


def silvey_delta(P, S=None):
    p, s = _inputs(P, S)
    return float(np.mean(np.maximum(1 - p.size * p * (s == 0), 0)))


def _int_poly_abs(w0, w1, c0, c1):
    r""":math:`\int_0^1 (w_0 + w_1 b)\,|c_0 + c_1 b|\,db` exactly (vectorized)."""
    w0, w1, c0, c1 = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (w0, w1, c0, c1)))

    def prim(b):  # antiderivative of (w0 + w1 b)(c0 + c1 b)
        return w0 * c0 * b + (w0 * c1 + w1 * c0) * b**2 / 2 + w1 * c1 * b**3 / 3

    with np.errstate(all="ignore"):
        r = np.where(c1 != 0, -c0 / np.where(c1 != 0, c1, 1.0), -1.0)
    r = np.where((r > 0) & (r < 1), r, 1.0)
    s0 = np.sign(c0 + c1 * 0.5 * r)  # sign on (0, r)
    s1 = np.sign(c0 + c1 * 0.5 * (1 + r))  # sign on (r, 1)
    zero = np.zeros_like(r)
    return s0 * (prim(r) - prim(zero)) + s1 * (prim(np.ones_like(r)) - prim(r))


def trutschnig_zeta(P, S=None) -> float:
    r"""Exact :math:`\zeta_1` of a bivariate checkerboard copula.

    ``P`` is the ``m x n`` mass matrix (normalized internally) and ``S`` the
    kernel sign matrix (``0``: :math:`\Pi`, ``1``: :math:`M`, ``-1``:
    :math:`W` cells, see :mod:`copul.checkerboard._biv_engine`).  For
    :math:`u` in row :math:`i` (local coordinate :math:`a`) and :math:`v` in
    column :math:`j` (local :math:`b`, :math:`v=(j+b)/n`)

    .. math::

       \partial_1 C(u,v) = m R_{ij} + m\Delta_{ij}\,k(a,b),\qquad
       R_{ij} = \textstyle\sum_{j'<j}\Delta_{ij'},

    with :math:`k=b` for :math:`\Pi` cells and :math:`k=\mathbf 1\{b>a\}`
    (:math:`M`) resp. :math:`\mathbf 1\{a+b>1\}` (:math:`W`).  For fixed
    :math:`b` the indicator equals one on an :math:`a`-set of measure
    :math:`b` in both singular cases, so with :math:`A = mR_{ij} - j/n` and
    :math:`p = m\Delta_{ij}` the cell contributes

    .. math::

       \frac{1}{mn}\int_0^1 |A + (p - \tfrac1n) b|\,db \;(\Pi),\qquad
       \frac{1}{mn}\int_0^1 \bigl[b\,|A + p - \tfrac bn|
       + (1-b)\,|A - \tfrac bn|\bigr]\,db \;(M, W),

    integrals of piecewise polynomials evaluated exactly.
    """
    P = np.asarray(P, dtype=float)
    P = P / P.sum()
    m, n = P.shape
    if S is None:
        S = np.zeros((m, n), dtype=int)
    S = np.broadcast_to(np.asarray(S, dtype=int), (m, n))
    R = np.zeros((m, n))
    R[:, 1:] = np.cumsum(P, axis=1)[:, :-1]
    j = np.arange(n)[None, :]
    A = m * R - j / n
    p = m * P
    c = 1.0 / n
    pi_part = _int_poly_abs(1.0, 0.0, A, p - c)
    sing_part = _int_poly_abs(0.0, 1.0, A + p, -c) + _int_poly_abs(1.0, -1.0, A, -c)
    cell = np.where(S == 0, pi_part, sing_part)
    return float(3.0 * cell.sum() / (m * n))
