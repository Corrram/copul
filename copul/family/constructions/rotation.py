r"""
Rotations, reflections and transposition of bivariate copulas.

All eight symmetries of the unit square (the dihedral group :math:`D_4`)
act on copulas.  If :math:`(U, V) \sim C`, the transformed copula
:math:`g\,C` is the distribution of

* ``transpose``:            :math:`(V, U)`, :math:`C^\top(u,v) = C(v,u)`;
* ``reflect(axis="u")``:    :math:`(1-U, V)`, :math:`C^{\sigma_1}(u,v) = v - C(1-u, v)`;
* ``reflect(axis="v")``:    :math:`(U, 1-V)`, :math:`C^{\sigma_2}(u,v) = u - C(u, 1-v)`;
* ``rotate(angle=180)`` = ``reflect(axis="both")`` = ``survival``:
  :math:`(1-U, 1-V)`, :math:`\hat C(u,v) = u + v - 1 + C(1-u, 1-v)`;
* ``rotate(angle=90)`` (counter-clockwise rotation of the scatter plot):
  :math:`(1-V, U)`, :math:`C_{90}(u,v) = v - C(v, 1-u)`;
* ``rotate(angle=270)``: :math:`(V, 1-U)`, :math:`C_{270}(u,v) = u - C(1-v, u)`.

For exchangeable :math:`C` the 90/270 degree rotations coincide with the
reflections ``"u"``/``"v"``, i.e. with the "rotated" copulas of the
VineCopula/pyvinecopulib convention (:math:`C_{90}(u,v) = v - C(1-u,v)`).
For non-exchangeable copulas (e.g. Khoudraji copulas) the geometric rotation
used here differs from that convention; use :func:`reflect` for the latter.

Compositions are collapsed: ``rotate(rotate(C, 90), 270)`` returns ``C``.

Measure relations (verified in the test-suite)
----------------------------------------------
Let :math:`s = \pm 1` be the concordance sign of the transformation
(:math:`s=+1` for the identity, transposition, the 180 degree rotation and
the anti-diagonal reflection; :math:`s=-1` otherwise).  Then

* :math:`\rho, \tau, \beta, \gamma` are multiplied by :math:`s`;
* Spearman's footrule: :math:`\phi(gC) = \phi(C)` if :math:`s=1` and
  :math:`\phi(gC) = \phi(C) - \tfrac32\gamma(C)` if :math:`s=-1`;
* Chatterjee's :math:`\xi` is invariant under reflections and
  :math:`\xi(C^\top) = \xi_2(C)` (conditioning on the other variable);
* Blest's :math:`\nu` (no transposition involved):
  :math:`\nu(C^{\sigma_1}) = \nu(C) - 2\rho(C)`,
  :math:`\nu(C^{\sigma_2}) = -\nu(C)`, :math:`\nu(\hat C) = 2\rho(C) - \nu(C)`;
* the tail coefficients are swapped by the 180 degree rotation
  (:math:`\lambda_L(\hat C) = \lambda_U(C)`) and invariant under transposition;
* distances to independence (Schweizer–Wolff :math:`\sigma`, Hoeffding's
  :math:`\Phi^2`, :math:`\kappa`, :math:`L^p`, Blum–Kiefer–Rosenblatt) and
  the mutual information are invariant under all eight symmetries.

All other combinations are evaluated by the numerical engine.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
Thm. 2.4.4 and Ex. 2.30 (survival copula, reflections).
"""

from __future__ import annotations

import numpy as np

from copul.family.constructions._base import (
    NumericBivCopula,
    component_callables,
    component_is_ac,
    component_pdf,
    component_rvs,
    ensure_numeric_copula,
)

__all__ = ["TransformedCopula", "reflect", "rotate", "survival", "transpose"]

_SWAP = np.array([[0, 1], [1, 0]])


def _matrix(swap: bool, flip_u: bool, flip_v: bool) -> np.ndarray:
    f = np.diag([-1 if flip_u else 1, -1 if flip_v else 1])
    return f @ _SWAP if swap else f


def _decompose(m: np.ndarray) -> tuple[bool, bool, bool]:
    if m[0, 1] == 0:
        return False, bool(m[0, 0] < 0), bool(m[1, 1] < 0)
    return True, bool(m[0, 1] < 0), bool(m[1, 0] < 0)


class TransformedCopula(NumericBivCopula):
    r"""Image of a bivariate copula under a symmetry of the unit square.

    The transformation maps a sample :math:`(U, V)\sim C` to
    :math:`(X, Y)`: first :math:`(U, V)\mapsto(V, U)` if ``swap``, then
    :math:`X\mapsto 1-X` if ``flip_u`` and :math:`Y\mapsto 1-Y` if
    ``flip_v``.  Usually created through :func:`rotate`, :func:`reflect`,
    :func:`transpose` or :func:`survival`.

    Parameters
    ----------
    base : BivCopula
        Fully specified bivariate copula.
    swap, flip_u, flip_v : bool
        The transformation (see above).

    Attributes
    ----------
    base : BivCopula
        The transformed copula.
    matrix : numpy.ndarray
        Signed permutation matrix acting on :math:`(U-\tfrac12, V-\tfrac12)`.
    """

    def __init__(self, base, swap: bool = False, flip_u: bool = False, flip_v: bool = False):
        self.base = ensure_numeric_copula(base, "base")
        self.swap, self.flip_u, self.flip_v = bool(swap), bool(flip_u), bool(flip_v)
        self.matrix = _matrix(self.swap, self.flip_u, self.flip_v)
        self._base_cdf, self._base_h1, self._base_h2 = component_callables(base)
        self._ac = component_is_ac(base)
        super().__init__()

    # -- description ---------------------------------------------------------
    @property
    def sign(self) -> int:
        """Concordance sign :math:`s=\\pm1` of the transformation."""
        return int(np.prod(self.matrix[self.matrix != 0]))

    @property
    def angle(self):
        """Rotation angle in degrees, or ``None`` for reflections."""
        for a, m in _ROTATIONS.items():
            if np.array_equal(m, self.matrix):
                return a
        return None

    def __repr__(self):
        a = self.angle
        if a is not None:
            return f"rotate({self.base!r}, {a})"
        if not self.swap:
            axis = "u" if self.flip_u else "v"
            return f"reflect({self.base!r}, axis={axis!r})"
        if not self.flip_u and not self.flip_v:
            return f"transpose({self.base!r})"
        return f"reflect({self.base!r}, axis='antidiagonal')"

    __str__ = __repr__

    @property
    def is_absolutely_continuous(self) -> bool:
        return self._ac

    # -- base in swapped coordinates ------------------------------------------
    def _B(self, x, y):
        return self._base_cdf(y, x) if self.swap else self._base_cdf(x, y)

    def _B1(self, x, y):
        return self._base_h2(y, x) if self.swap else self._base_h1(x, y)

    def _B2(self, x, y):
        return self._base_h1(y, x) if self.swap else self._base_h2(x, y)

    def _coords(self, u, v):
        return (1.0 - u if self.flip_u else u), (1.0 - v if self.flip_v else v)

    # -- numerics ---------------------------------------------------------------
    def _cdf(self, u, v):
        x, y = self._coords(u, v)
        b = self._B(x, y)
        if self.flip_u and self.flip_v:
            return u + v - 1.0 + b
        if self.flip_u:
            return v - b
        if self.flip_v:
            return u - b
        return b

    def _h1(self, u, v):
        x, y = self._coords(u, v)
        b1 = self._B1(x, y)
        return 1.0 - b1 if self.flip_v else b1

    def _h2(self, u, v):
        x, y = self._coords(u, v)
        b2 = self._B2(x, y)
        return 1.0 - b2 if self.flip_u else b2

    def _pdf(self, u, v):
        x, y = self._coords(u, v)
        pdf = component_pdf(self.base)
        return pdf(y, x) if self.swap else pdf(x, y)

    def _rvs(self, n, rng):
        s = component_rvs(self.base, n, rng)
        x, y = (s[:, 1], s[:, 0]) if self.swap else (s[:, 0], s[:, 1])
        if self.flip_u:
            x = 1.0 - x
        if self.flip_v:
            y = 1.0 - y
        return np.column_stack([x, y])

    # -- measure relations ------------------------------------------------------
    def _concordance(self, name):
        return self.sign * float(getattr(self.base, name)())

    def spearmans_rho(self, *args, **kwargs):
        r""":math:`\rho(gC) = s\,\rho(C)`."""
        return self._concordance("spearmans_rho")

    def kendalls_tau(self, *args, **kwargs):
        r""":math:`\tau(gC) = s\,\tau(C)`."""
        return self._concordance("kendalls_tau")

    def ginis_gamma(self, *args, **kwargs):
        r""":math:`\gamma(gC) = s\,\gamma(C)`."""
        return self._concordance("ginis_gamma")

    def spearmans_footrule(self, *args, **kwargs):
        r""":math:`\phi(gC) = \phi(C)` (:math:`s=1`) or :math:`\phi(C) - \tfrac32\gamma(C)`."""
        phi = float(self.base.spearmans_footrule())
        if self.sign > 0:
            return phi
        return phi - 1.5 * float(self.base.ginis_gamma())

    def blests_nu(self, *args, **kwargs):
        r"""Blest's :math:`\nu` for the transformations not involving a transposition."""
        if self.swap:
            raise NotImplementedError("no closed relation for nu under transposition")
        nu = float(self.base.blests_nu())
        if not self.flip_u and not self.flip_v:
            return nu
        if self.flip_v and not self.flip_u:
            return -nu
        rho = float(self.base.spearmans_rho())
        if self.flip_u and not self.flip_v:
            return nu - 2.0 * rho
        return 2.0 * rho - nu

    def chatterjees_xi(self, *args, condition_on_y=False, **kwargs):
        r""":math:`\xi` is reflection invariant; transposition swaps :math:`\xi` and :math:`\xi_2`."""
        cond = bool(condition_on_y) != self.swap
        return float(self.base.chatterjees_xi(condition_on_y=cond))

    def _tail(self, lower: bool):
        if self.sign < 0:
            raise NotImplementedError("tail coefficients of the off-diagonal corners")
        swapped = self.flip_u and self.flip_v
        use_lower = lower != swapped
        return float(self.base.lambda_L() if use_lower else self.base.lambda_U())

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L`; swapped with :math:`\lambda_U` by the 180 degree rotation."""
        return self._tail(True)

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U`; swapped with :math:`\lambda_L` by the 180 degree rotation."""
        return self._tail(False)

    def schweizer_wolff_sigma(self, *args, **kwargs):
        """Invariant under all symmetries of the square."""
        return float(self.base.schweizer_wolff_sigma())

    def hoeffdings_d(self, *args, **kwargs):
        """Invariant under all symmetries of the square."""
        return float(self.base.hoeffdings_d())

    def uniform_distance(self, *args, **kwargs):
        """Invariant under all symmetries of the square."""
        return float(self.base.uniform_distance())

    def blum_kiefer_rosenblatt(self, *args, **kwargs):
        """Invariant under all symmetries of the square."""
        return float(self.base.blum_kiefer_rosenblatt())

    def mutual_information(self, *args, **kwargs):
        """Invariant under all symmetries of the square."""
        return float(self.base.mutual_information())


_ROTATIONS = {
    0: _matrix(False, False, False),
    90: _matrix(True, True, False),
    180: _matrix(False, True, True),
    270: _matrix(True, False, True),
}

_REFLECTIONS = {
    "u": _matrix(False, True, False),
    "v": _matrix(False, False, True),
    "both": _matrix(False, True, True),
    "diagonal": _matrix(True, False, False),
    "antidiagonal": _matrix(True, True, True),
}


def _apply(C, m: np.ndarray):
    """Apply the signed permutation ``m`` to ``C`` (collapsing compositions)."""
    if isinstance(C, TransformedCopula):
        m = m @ C.matrix
        C = C.base
    else:
        ensure_numeric_copula(C)
    if np.array_equal(m, np.eye(2, dtype=int)):
        return C
    swap, fu, fv = _decompose(m)
    return TransformedCopula(C, swap=swap, flip_u=fu, flip_v=fv)


def rotate(C, angle: int = 180):
    r"""Rotate (the scatter plot of) a copula counter-clockwise by ``angle`` degrees.

    Parameters
    ----------
    C : BivCopula
        Fully specified bivariate copula.
    angle : {0, 90, 180, 270}
        Rotation angle (multiples of 90; negative values allowed).

    Returns
    -------
    TransformedCopula or BivCopula
        ``C`` itself for ``angle % 360 == 0``.

    Notes
    -----
    With :math:`(U,V)\sim C`: 90 degrees gives :math:`(1-V, U)` and
    :math:`C_{90}(u,v) = v - C(v, 1-u)`; 180 degrees the survival copula
    :math:`\hat C(u,v) = u+v-1+C(1-u,1-v)`; 270 degrees gives
    :math:`(V, 1-U)` and :math:`C_{270}(u,v) = u - C(1-v, u)`.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.family.constructions import rotate
    >>> R = rotate(cp.Clayton(2), 180)          # survival Clayton
    >>> round(R.lambda_U(), 12) == round(2 ** -0.5, 12)
    True
    >>> round(rotate(cp.Clayton(2), 90).kendalls_tau(), 12)
    -0.5
    """
    a = int(angle) % 360
    if a not in _ROTATIONS:
        raise ValueError(f"angle must be a multiple of 90 degrees, got {angle}")
    return _apply(C, _ROTATIONS[a])


def reflect(C, axis: str = "u"):
    r"""Reflect a copula.

    Parameters
    ----------
    C : BivCopula
        Fully specified bivariate copula.
    axis : {"u", "v", "both", "diagonal", "antidiagonal"}
        ``"u"``: :math:`(1-U, V)`, :math:`v - C(1-u, v)`;
        ``"v"``: :math:`(U, 1-V)`, :math:`u - C(u, 1-v)`;
        ``"both"``: the survival copula;
        ``"diagonal"``: the transpose :math:`C(v,u)`;
        ``"antidiagonal"``: :math:`(1-V, 1-U)`, :math:`u+v-1+C(1-v,1-u)`.
    """
    key = str(axis).lower()
    aliases = {"x": "u", "y": "v", "1": "u", "2": "v", "survival": "both", "transpose": "diagonal"}
    key = aliases.get(key, key)
    if key not in _REFLECTIONS:
        raise ValueError(f"axis must be one of {sorted(_REFLECTIONS)}, got {axis!r}")
    return _apply(C, _REFLECTIONS[key])


def transpose(C):
    r"""Transposed copula :math:`C^\top(u,v) = C(v,u)` (distribution of :math:`(V,U)`)."""
    return _apply(C, _REFLECTIONS["diagonal"])


def survival(C):
    r"""Survival copula :math:`\hat C(u,v) = u + v - 1 + C(1-u, 1-v)` (= ``rotate(C, 180)``)."""
    return _apply(C, _REFLECTIONS["both"])
