r"""
Catalogue of exact regions with closed-form boundaries.

Only regions whose boundary formulas are (i) sourced from this repository or
the literature and (ii) survive the numerical validation in
``tests/regions/`` (boundary copulas attain the boundary, 2000 random
checkerboards lie inside, optimal checkerboards from :mod:`copul.optim` lie
inside and close to the boundary) are registered.

Registered regions
------------------
``(xi, rho)``
    :math:`|\rho|\le M(\xi)` with the Ansari--Rockel bound
    (:func:`copul.schur_order.bounds_from_xi.rho_max_given_xi`), attained by
    :class:`~copul.family.other.xi_rho_boundary_copula.XiRhoBoundaryCopula`.
``(xi, nu)``
    :math:`|\nu|\le N(b(\xi))` with :math:`\Xi(b)=\xi`
    (:func:`~copul.schur_order.bounds_from_xi.nu_bounds_from_xi`), attained by
    :class:`~copul.family.other.clamped_parabola_copula.XiNuBoundaryCopula`.
``(rho, nu)``
    upper boundary traced by the V-threshold family
    (:class:`~copul.family.other.v_threshold_copula.VThresholdCopula`),
    :math:`\nu_{\max}(\rho)=1-\tfrac34(1-\rho)^{4/3}` for :math:`\rho\ge0`;
    lower boundary :math:`\nu_{\min}(\rho)=-\nu_{\max}(-\rho)`
    (``notes/examples/blest-regions/2025-10-14-nu-rho-region.py``).
``(xi, footrule)`` for SI copulas
    :math:`\xi\le\psi\le\sqrt\xi`
    (:func:`~copul.schur_order.bounds_from_xi.psi_bounds_from_xi` with
    ``cls="SI"``), upper boundary attained by Fréchet mixtures
    :math:`\sqrt\xi\,M+(1-\sqrt\xi)\,\Pi`.
``(xi, beta)``
    :math:`|\beta|^3\le2\xi` (Orenday Lares & Rockel, arXiv:2606.30033).

"""

from __future__ import annotations

import numpy as np

from copul.regions.base import ExactRegion, KeyPoint

__all__ = [
    "build_default_regions",
    "nu_max_given_rho",
    "nu_max_given_xi",
    "rho_max_given_xi",
    "xi_nu_parametric",
]


# ----------------------------------------------------------------------------
# vectorised closed forms
# ----------------------------------------------------------------------------
def _b_of_xi_rho(x: np.ndarray) -> np.ndarray:
    r"""Parameter :math:`b(\xi)` of the :math:`(\xi,\rho)` boundary family :math:`C_b`."""
    x = np.asarray(x, dtype=float)
    b = np.zeros_like(x)
    lo = (x > 0) & (x <= 0.3)
    hi = (x > 0.3) & (x < 1)
    xl = x[lo]
    arg = np.clip(-3.0 * np.sqrt(6.0 * xl) / 5.0, -1.0, 1.0)
    b[lo] = np.sqrt(6.0 * xl) / (2.0 * np.cos(np.arccos(arg) / 3.0))
    xh = x[hi]
    b[hi] = (5.0 + np.sqrt(5.0 * (6.0 * xh - 1.0))) / (10.0 * (1.0 - xh))
    b[x >= 1] = np.inf
    return b


def rho_max_given_xi(x) -> np.ndarray:
    r"""Vectorised :math:`\max\{\rho(C):\xi(C)=x\}` (Ansari & Rockel).

    .. math::

       M(x)=\begin{cases} b-\tfrac{3b^2}{10}, & x\le\tfrac3{10},\\
       1-\tfrac1{2b^2}+\tfrac1{5b^3}, & x>\tfrac3{10},\end{cases}

    with :math:`b=b(x)` as in :func:`copul.schur_order.bounds_from_xi.rho_max_given_xi`.
    """
    x = np.asarray(x, dtype=float)
    b = _b_of_xi_rho(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(x <= 0.3, b - 0.3 * b * b, 0.0)
        big = 1.0 - 1.0 / (2.0 * b * b) + 1.0 / (5.0 * b**3)
    out = np.where(x > 0.3, big, out)
    out = np.where(x >= 1.0, 1.0, out)
    out = np.where(x <= 0.0, 0.0, out)
    return out


def xi_nu_parametric(b) -> tuple[np.ndarray, np.ndarray]:
    r"""Boundary curve :math:`b\mapsto(\Xi(b),N(b))` of the :math:`(\xi,\nu)` region.

    With :math:`s=b^{-1/2}`, :math:`t=\sqrt{1-1/b}` and
    :math:`A=\operatorname{arsinh}(t/s)`,

    .. math::

       \Xi(b)=\begin{cases}\frac{8(7s^2-3)}{105s^6}, & b\le1,\\[1ex]
       \frac{-105s^8A+183s^6t-38s^4t-88s^2t+112s^2+48t-48}{210s^6}, & b>1,
       \end{cases}

    .. math::

       N(b)=\begin{cases}\frac{4(28s^2-9)}{105s^4}, & b\le1,\\[1ex]
       \frac{-105s^8A+87s^6t+250s^4t-376s^2t+448s^2+144t-144}{420s^4}, & b>1,
       \end{cases}

    (see :mod:`copul.schur_order.bounds_from_xi`).  Evaluated in extended
    precision to limit the cancellation for large :math:`b`.
    """
    b = np.asarray(b, dtype=np.longdouble)
    s2 = 1.0 / b
    s = np.sqrt(s2)
    small = b <= 1
    xi = np.empty_like(b)
    nu = np.empty_like(b)
    xi[small] = 8.0 * (7.0 * s2[small] - 3.0) / (105.0 * s2[small] ** 3)
    nu[small] = 4.0 * (28.0 * s2[small] - 9.0) / (105.0 * s2[small] ** 2)
    lg = ~small
    if np.any(lg):
        sl, s2l = s[lg], s2[lg]
        t = np.sqrt(1.0 - s2l)
        A = np.arcsinh(t / sl)
        tm1 = -s2l / (1.0 + t)  # t - 1 without cancellation
        num_x = (
            -105.0 * s2l**4 * A
            + 183.0 * s2l**3 * t
            - 38.0 * s2l**2 * t
            - 88.0 * s2l * t
            + 112.0 * s2l
            + 48.0 * tm1
        )
        num_n = (
            -105.0 * s2l**4 * A
            + 87.0 * s2l**3 * t
            + 250.0 * s2l**2 * t
            - 376.0 * s2l * t
            + 448.0 * s2l
            + 144.0 * tm1
        )
        xi[lg] = num_x / (210.0 * s2l**3)
        nu[lg] = num_n / (420.0 * s2l**2)
    return xi.astype(float), nu.astype(float)


def _b_of_xi_nu(x: np.ndarray, iters: int = 110) -> np.ndarray:
    """Vectorised bisection for :math:`\\Xi(b)=x` in :math:`\\log b`."""
    x = np.asarray(x, dtype=float)
    lo = np.full(x.shape, -40.0)
    hi = np.full(x.shape, 40.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        xm, _ = xi_nu_parametric(np.exp(mid))
        up = xm < x
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return np.exp(0.5 * (lo + hi))


def nu_max_given_xi(x) -> np.ndarray:
    r"""Vectorised :math:`\max\{\nu(C):\xi(C)=x\}=N(b)` with :math:`\Xi(b)=x`."""
    x = np.asarray(x, dtype=float)
    inner = (x > 0) & (x < 1)
    out = np.where(x >= 1, 1.0, 0.0)
    if np.any(inner):
        b = _b_of_xi_nu(x[inner])
        _, nu = xi_nu_parametric(b)
        out = out.astype(float)
        out[inner] = nu
    return out


def nu_max_given_rho(r) -> np.ndarray:
    r"""Upper boundary of the :math:`(\rho,\nu)` region (V-threshold family).

    With :math:`\mu\in[0,2]`: :math:`\rho=1-\mu^3`, :math:`\nu=1-\tfrac34\mu^4`
    for :math:`\mu\le1` and :math:`\rho=(2-\mu)^3-1`,
    :math:`\nu=-\tfrac34\mu^4+4\mu^3-6\mu^2+3` for :math:`\mu\ge1`.
    """
    r = np.clip(np.asarray(r, dtype=float), -1.0, 1.0)
    mu_pos = np.cbrt(1.0 - r)
    mu_neg = 2.0 - np.cbrt(1.0 + r)
    mu = np.where(r >= 0, mu_pos, mu_neg)
    nu_small = 1.0 - 0.75 * mu**4
    nu_big = -0.75 * mu**4 + 4.0 * mu**3 - 6.0 * mu**2 + 3.0
    return np.where(mu <= 1.0, nu_small, nu_big)


# ----------------------------------------------------------------------------
# boundary copula factories (lazy imports)
# ----------------------------------------------------------------------------
def _pi():
    from copul.family.frechet.biv_independence_copula import BivIndependenceCopula

    return BivIndependenceCopula()


def _M():
    from copul.family.frechet.upper_frechet import UpperFrechet

    return UpperFrechet()


def _W():
    from copul.family.frechet.lower_frechet import LowerFrechet

    return LowerFrechet()


def _xi_rho_family(sign: float):
    def factory(x: float):
        if x <= 0:
            return _pi()
        if x >= 1:
            return _M() if sign > 0 else _W()
        from copul.family.other.xi_rho_boundary_copula import XiRhoBoundaryCopula

        b = float(_b_of_xi_rho(np.array([x]))[0])
        return XiRhoBoundaryCopula(b=sign * b)

    return factory


def _xi_nu_family(sign: float):
    def factory(x: float):
        if x <= 0:
            return _pi()
        if x >= 1:
            return _M() if sign > 0 else _W()
        from copul.family.other.clamped_parabola_copula import XiNuBoundaryCopula

        b = float(_b_of_xi_nu(np.array([x]))[0])
        return XiNuBoundaryCopula(b=sign * b)

    return factory


def _rho_nu_upper(r: float):
    if r >= 1:
        return _M()
    if r <= -1:
        return _W()
    from copul.family.other.v_threshold_copula import VThresholdCopula

    return VThresholdCopula.from_rho(float(r))


def _frechet_sqrt(x: float):
    from copul.family.frechet.frechet import Frechet

    a = float(np.sqrt(max(x, 0.0)))
    if a <= 0:
        return _pi()
    if a >= 1:
        return _M()
    return Frechet(alpha=a, beta=0.0)


# ----------------------------------------------------------------------------
# region definitions
# ----------------------------------------------------------------------------
def _kp(label, x, y, factory=None) -> KeyPoint:
    return KeyPoint(label, float(x), float(y), factory)


def xi_rho() -> ExactRegion:
    """Exact :math:`(\\xi,\\rho)` region."""
    return ExactRegion(
        "xi",
        "rho",
        lower=lambda x: -rho_max_given_xi(x),
        upper=rho_max_given_xi,
        x_range=(0.0, 1.0),
        key_points=[
            _kp(r"\Pi", 0, 0, _pi),
            _kp("M", 1, 1, _M),
            _kp("W", 1, -1, _W),
            _kp("C_1", 0.3, 0.7, lambda: _xi_rho_family(1)(0.3)),
            _kp("C_{-1}", 0.3, -0.7, lambda: _xi_rho_family(-1)(0.3)),
        ],
        reference=(
            "J. Ansari and M. Rockel: exact region of Chatterjee's xi and "
            "Spearman's rho, |rho| <= M(xi), attained by the diagonal-band "
            "family C_b."
        ),
        source=(
            "copul/schur_order/bounds_from_xi.py (rho_max_given_xi); "
            "copul/family/other/xi_rho_boundary_copula.py"
        ),
        boundary_family={"upper": _xi_rho_family(1.0), "lower": _xi_rho_family(-1.0)},
        status="published",
    )


def xi_nu() -> ExactRegion:
    """Exact :math:`(\\xi,\\nu)` region."""
    x1, n1 = (float(v[0]) for v in xi_nu_parametric(np.array([1.0])))
    return ExactRegion(
        "xi",
        "nu",
        lower=lambda x: -nu_max_given_xi(x),
        upper=nu_max_given_xi,
        x_range=(0.0, 1.0),
        key_points=[
            _kp(r"\Pi", 0, 0, _pi),
            _kp("M", 1, 1, _M),
            _kp("W", 1, -1, _W),
            _kp(r"C^{\xi,\nu}_{1}", x1, n1, lambda: _xi_nu_family(1)(32 / 105)),
        ],
        reference=(
            "|nu| <= N(b) with Xi(b) = xi, attained by the clamped-parabola "
            "family (XiNuBoundaryCopula)."
        ),
        source=(
            "copul/schur_order/bounds_from_xi.py (nu_bounds_from_xi); "
            "copul/family/other/clamped_parabola_copula.py; "
            "notes/examples/blest-regions/2025-10-18-xi-blest-region.py"
        ),
        boundary_family={"upper": _xi_nu_family(1.0), "lower": _xi_nu_family(-1.0)},
        status="package",
    )


def rho_nu() -> ExactRegion:
    """Exact :math:`(\\rho,\\nu)` region."""
    return ExactRegion(
        "rho",
        "nu",
        lower=lambda r: -nu_max_given_rho(-np.asarray(r, dtype=float)),
        upper=nu_max_given_rho,
        x_range=(-1.0, 1.0),
        key_points=[
            _kp("M", 1, 1, _M),
            _kp("W", -1, -1, _W),
            _kp("C_1", 0.0, 0.25, lambda: _rho_nu_upper(0.0)),
        ],
        reference=(
            "M. Rockel: exact regions for Blest's rank correlation nu, "
            "arXiv:2609.27634; upper boundary by the V-threshold family, "
            "lower boundary by central symmetry."
        ),
        source=(
            "notes/examples/blest-regions/2025-10-14-nu-rho-region.py; "
            "copul/family/other/v_threshold_copula.py"
        ),
        boundary_family={"upper": _rho_nu_upper},
        status="preprint",
        notes=(
            "The lower boundary is attained by the reflections u - C(u, 1 - v) of "
            "the V-threshold copulas; no class is provided (boundary_copula "
            "returns None for side='lower')."
        ),
    )


def xi_footrule_si() -> ExactRegion:
    """Exact :math:`(\\xi,\\psi)` region for stochastically increasing copulas."""
    return ExactRegion(
        "xi",
        "footrule",
        lower=lambda x: np.asarray(x, dtype=float),
        upper=lambda x: np.sqrt(np.clip(np.asarray(x, dtype=float), 0, None)),
        x_range=(0.0, 1.0),
        key_points=[
            _kp(r"\Pi", 0, 0, _pi),
            _kp("M", 1, 1, _M),
            _kp(r"C^{\mathrm{Fr}}_{1/2}", 0.25, 0.5, lambda: _frechet_sqrt(0.25)),
        ],
        reference="xi <= psi <= sqrt(xi) for SI copulas (package bounds).",
        source=(
            "copul/schur_order/bounds_from_xi.py (psi_bounds_from_xi, cls='SI'); "
            "notes/examples/xi-footrule/2025-08-17-xi-footrule-si-region.py"
        ),
        boundary_family={"upper": _frechet_sqrt},
        copula_class="si",
        status="package",
    )


def xi_beta() -> ExactRegion:
    """Exact :math:`(\\xi,\\beta)` region :math:`|\\beta|^3\\le2\\xi`."""

    def upper(x):
        return np.minimum(np.cbrt(2.0 * np.clip(np.asarray(x, dtype=float), 0.0, None)), 1.0)

    return ExactRegion(
        "xi",
        "beta",
        lower=lambda x: -upper(x),
        upper=upper,
        x_range=(0.0, 1.0),
        key_points=[
            _kp(r"\Pi", 0, 0, _pi),
            _kp("M", 1, 1, _M),
            _kp("W", 1, -1, _W),
        ],
        reference=(
            "J. I. Orenday Lares and M. Rockel: exact region of Chatterjee's xi "
            "and Blomqvist's beta, arXiv:2606.30033; |beta|^3 <= 2 xi."
        ),
        source="arXiv:2606.30033",
        status="preprint",
        notes="No boundary copula class is provided (boundary_copula returns None).",
    )


def build_default_regions() -> list[ExactRegion]:
    """All regions registered by default."""
    return [xi_rho(), xi_nu(), rho_nu(), xi_footrule_si(), xi_beta()]
