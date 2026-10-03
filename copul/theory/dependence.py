r"""
Positive and negative dependence concepts of bivariate copulas.

Every check works on any fully specified bivariate copula object of
:mod:`copul` and returns a :class:`PropertyResult` (truthy iff the property
holds) recording *how* the answer was obtained:

``"exact"``
    a published family-level characterization (e.g. the Gaussian copula is
    TP2 iff :math:`\rho\ge 0`), an exact finite algorithm (checkerboard
    copulas) or an implication of such a fact through the hierarchy below;
``"symbolic"``
    a one-dimensional characterization in terms of a generator, evaluated
    on a dense grid in high precision (Archimedean copulas);
``"grid"``
    a dense, locally refined grid check of the defining inequality using the
    vectorized numerical API (``cdf``, :math:`\partial_1 C`,
    :math:`\partial_2 C`, density).  A grid check can miss violations
    between grid points, but it never reports violations that are not there
    (beyond its numerical tolerance).

Conventions
-----------
Throughout :math:`(U, V)\sim C`.  The index ``i`` denotes the *conditioning*
variable: ``i = 1`` gives the properties of :math:`V` given :math:`U`
(written ``(V|U)``), ``i = 2`` those of :math:`U` given :math:`V`.

==============  ==========================================================
key             definition (for all :math:`u, v` resp. all
                :math:`u_1\le u_2`, :math:`v_1\le v_2`)
==============  ==========================================================
``PQD``         :math:`C(u,v)\ge uv`
``NQD``         :math:`C(u,v)\le uv`
``LTD(V|U)``    :math:`u\mapsto P(V\le v\mid U\le u)=C(u,v)/u` nonincreasing
``LTI(V|U)``    :math:`u\mapsto C(u,v)/u` nondecreasing
``RTI(V|U)``    :math:`u\mapsto P(V>v\mid U>u)=\frac{1-u-v+C(u,v)}{1-u}`
                nondecreasing
``RTD(V|U)``    the same map nonincreasing
``SI(V|U)``     :math:`u\mapsto P(V\le v\mid U=u)=\partial_1C(u,v)`
                nonincreasing (:math:`V` stochastically increasing in
                :math:`U`; equivalently :math:`C(\cdot,v)` concave)
``SD(V|U)``     :math:`\partial_1 C(u,v)` nondecreasing in :math:`u`
                (:math:`C(\cdot, v)` convex)
``LCSD``        the function :math:`C` is TP2:
                :math:`C(u_1,v_1)C(u_2,v_2)\ge C(u_1,v_2)C(u_2,v_1)`
``RCSI``        the survival function :math:`\bar C(u,v)=1-u-v+C(u,v)` is
                TP2
``TP2``         the density :math:`c` is TP2 (positive likelihood ratio
                dependence); only absolutely continuous copulas qualify
``RR2``         the density is reverse regular,
                :math:`c(u_1,v_1)c(u_2,v_2)\le c(u_1,v_2)c(u_2,v_1)`
==============  ==========================================================

The ``(U|V)`` variants are the same conditions for the transposed copula
:math:`C^\top(u,v) = C(v,u)`.  Aliases: ``PLOD`` (= ``PQD`` in the bivariate
case), ``CI``/``CIS`` (= ``SI``), ``TP2_CDF`` (= ``LCSD``, Nelsen 2006,
Cor. 5.2.17), ``TP2_SURVIVAL`` (= ``RCSI``), ``PLR`` / ``TP2_DENSITY``
(= ``TP2``), ``NLR`` / ``RR2_DENSITY`` (= ``RR2``).

Implication hierarchy (:data:`IMPLICATIONS`; Nelsen 2006, Sect. 5.2, in
particular Thm. 5.2.19; Joe 1997, Thm. 2.3)::

    TP2 => SI(V|U), SI(U|V), LCSD, RCSI
    SI(V|U) => LTD(V|U), RTI(V|U)          (same for (U|V))
    LCSD => LTD(V|U), LTD(U|V);  RCSI => RTI(V|U), RTI(U|V)
    LTD(.|.) => PQD;  RTI(.|.) => PQD

and the negative counterparts obtained by the reflection
:math:`(U,V)\mapsto(U,1-V)` (Nelsen 2006, Thm. 2.4.4)::

    RR2 => SD(V|U), SD(U|V);  SD => LTI, RTD (same conditioning);
    LTI(.|.) => NQD;  RTD(.|.) => NQD

Exact family characterizations
------------------------------
* independence :math:`\Pi` has every property; :math:`M` every positive
  one except ``TP2`` (no density), :math:`W` every negative one except
  ``RR2``;
* checkerboard copulas (``BivCheckPi``, ``BivCheckMin``, ``BivCheckW``,
  ``BivCheckMixed``) and straight shuffles of :math:`M`: exact finite
  algorithms of :mod:`copul.checkerboard._biv_engine` for the quadrant,
  tail and stochastic monotonicity properties, a matrix TP2 test for the
  density and, for independence-kernel checkerboards, an exact test of
  ``LCSD``/``RCSI`` (see :func:`_checkerboard_facts`);
* Gaussian copula: TP2 density iff :math:`\rho\ge0`, RR2 iff
  :math:`\rho\le0` (Karlin & Rinott 1980; the mixed log-derivative of the
  bivariate normal density is :math:`\rho/(1-\rho^2)`);
* Farlie--Gumbel--Morgenstern copula,
  :math:`c=1+\theta(1-2u)(1-2v)`: TP2 iff :math:`\theta\ge0`, RR2 iff
  :math:`\theta\le0` (direct computation:
  :math:`c_{11}c_{22}-c_{12}c_{21}=4\theta(u_2-u_1)(v_2-v_1)`);
* Student-t copula: only the sign facts that follow from
  :math:`\tau=\tfrac2\pi\arcsin\rho` (Lindskog, McNeil & Schmock 2003) and
  the monotonicity of :math:`\tau` in the concordance order (Nelsen 2006,
  Sect. 5.1): not PQD for :math:`\rho<0`, not NQD for :math:`\rho>0`, and
  neither for :math:`\rho=0`.  (The t copula is *not* PQD for all
  :math:`\rho>0`; the remaining cases are left to the grid.)
* bivariate extreme-value copulas are SI in both variables
  (Garralda-Guillem 2000) and LCSD; hence LTD, RTI and PQD (Joe 1997,
  Ch. 6), and none of the negative properties unless :math:`C=\Pi`;
* Archimedean copulas :math:`C(u,v)=\psi(\varphi(u)+\varphi(v))`,
  :math:`\psi=\varphi^{[-1]}`: a non-strict generator makes :math:`C`
  vanish on a set of positive measure, so no positive property holds;
  otherwise LTD (:math:`\Leftrightarrow` LCSD) iff :math:`\log\psi` is
  convex, SI iff :math:`\log(-\psi')` is convex and TP2 iff
  :math:`\log\psi''` is convex, and LTI/SD/RR2 iff the respective
  functions are log-concave (Capéraà & Genest 1993; Müller & Scarsini
  2005; Nelsen 2006, Sect. 5.2).  Laplace-transform generators (Clayton
  :math:`\theta>0`, Gumbel--Hougaard, Frank :math:`\theta>0`, Joe, AMH
  :math:`\theta\in[0,1)` and the BB families) are completely monotone
  (Kimberling 1974; Joe 1997, Ch. 4); by Bernstein's theorem and the
  Cauchy--Schwarz inequality :math:`\psi''` is then log-convex (Widder
  1941, Ch. IV), so the density is TP2;
* rotations, reflections and the transposition of a copula inherit the
  (correspondingly mapped) facts of the base copula; mixtures inherit the
  properties defined by inequalities that are linear in :math:`C`
  (quadrant, tail and stochastic monotonicity) when every component has
  them.

References
----------
* Capéraà, P. & Genest, C. (1993). Spearman's rho is larger than Kendall's
  tau for positively dependent random variables. *J. Nonparametr. Stat.* 2,
  183--194.
* Garralda-Guillem, A. I. (2000). Structure de dépendance des lois de
  valeurs extrêmes bivariées. *C. R. Acad. Sci. Paris* 330, 593--596.
* Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman &
  Hall, Ch. 2.
* Karlin, S. & Rinott, Y. (1980). Classes of orderings of measures and
  related correlation inequalities I. *J. Multivariate Anal.* 10, 467--498.
* Kimberling, C. H. (1974). A probabilistic interpretation of complete
  monotonicity. *Aequationes Math.* 10, 152--164.
* Widder, D. V. (1941). *The Laplace Transform*. Princeton University
  Press.
* Lindskog, F., McNeil, A. & Schmock, U. (2003). Kendall's tau for
  elliptical distributions. In *Credit Risk*, Physica, 149--156.
* Müller, A. & Scarsini, M. (2005). Archimedean copulae and positive
  dependence. *J. Multivariate Anal.* 93, 434--445.
* Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
  Sect. 5.2.

Examples
--------
>>> import copul as cp
>>> from copul.theory.dependence import check_property, dependence_profile
>>> r = check_property(cp.Clayton(2), "SI", i=1)
>>> bool(r), r.method
(True, 'exact')
>>> bool(check_property(cp.Gaussian(-0.3), "PQD"))
False
>>> dependence_profile(cp.Frank(-3))["SD(V|U)"].holds
True
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "IMPLICATIONS",
    "PROPERTIES",
    "DependenceProfile",
    "DependenceProperty",
    "PropertyResult",
    "check_property",
    "dependence_profile",
    "exact_facts",
    "implication_violations",
    "is_lcsd",
    "is_ltd",
    "is_lti",
    "is_nqd",
    "is_pqd",
    "is_rcsi",
    "is_rr2_density",
    "is_rtd",
    "is_rti",
    "is_sd",
    "is_si",
    "is_tp2_cdf",
    "is_tp2_density",
    "is_tp2_survival",
    "resolve_property",
]

# ---------------------------------------------------------------------------
# property catalogue
# ---------------------------------------------------------------------------

_NELSEN = "Nelsen (2006), Sect. 5.2"


@dataclass(frozen=True)
class DependenceProperty:
    """Description of a bivariate dependence property.

    Attributes
    ----------
    key : str
        Canonical key, e.g. ``"SI(V|U)"``.
    concept : str
        Concept without the conditioning, e.g. ``"SI"``.
    kind : str
        ``"qd"`` (quadrant), ``"sm"`` (stochastic monotonicity), ``"lt"`` /
        ``"rt"`` (left / right tail monotonicity), ``"cs"`` (corner sets) or
        ``"dens"`` (density).
    sign : int
        ``+1`` for positive, ``-1`` for negative dependence.
    cond : int or None
        Conditioning variable (``1`` = :math:`U`, ``2`` = :math:`V`).
    side : str or None
        ``"L"`` / ``"R"`` for tail and corner set properties.
    description : str
        The defining condition.
    reference : str
        Literature reference of the definition.
    """

    key: str
    concept: str
    kind: str
    sign: int
    cond: int | None
    side: str | None
    description: str
    reference: str


def _mk(concept, kind, sign, cond, side, desc, ref=_NELSEN):
    if cond is None:
        key = concept
    else:
        key = f"{concept}({'V|U' if cond == 1 else 'U|V'})"
    return DependenceProperty(key, concept, kind, sign, cond, side, desc, ref)


def _catalogue():
    out = [
        _mk("PQD", "qd", 1, None, None, "C(u,v) >= uv", _NELSEN + ".1"),
        _mk("NQD", "qd", -1, None, None, "C(u,v) <= uv", _NELSEN + ".1"),
    ]
    tail = {
        ("lt", 1): ("LTD", "P(V<=v | U<=u) = C(u,v)/u nonincreasing in u"),
        ("lt", -1): ("LTI", "C(u,v)/u nondecreasing in u"),
        ("rt", 1): ("RTI", "P(V>v | U>u) = (1-u-v+C(u,v))/(1-u) nondecreasing in u"),
        ("rt", -1): ("RTD", "(1-u-v+C(u,v))/(1-u) nonincreasing in u"),
    }
    for (kind, sign), (concept, desc) in tail.items():
        for cond in (1, 2):
            d = desc if cond == 1 else desc + " (for the transposed copula)"
            out.append(_mk(concept, kind, sign, cond, kind[0].upper(), d, _NELSEN + ".2"))
    for sign, concept, desc in (
        (1, "SI", "P(V<=v | U=u) = d1 C(u,v) nonincreasing in u (C concave in u)"),
        (-1, "SD", "d1 C(u,v) nondecreasing in u (C convex in u)"),
    ):
        for cond in (1, 2):
            d = desc if cond == 1 else desc + " (for the transposed copula)"
            out.append(_mk(concept, "sm", sign, cond, None, d, _NELSEN + ".3"))
    out += [
        _mk(
            "LCSD",
            "cs",
            1,
            None,
            "L",
            "C is TP2 (left corner set decreasing)",
            "Nelsen (2006), Cor. 5.2.17",
        ),
        _mk(
            "RCSI",
            "cs",
            1,
            None,
            "R",
            "1-u-v+C(u,v) is TP2 (right corner set increasing)",
            "Nelsen (2006), Cor. 5.2.17",
        ),
        _mk(
            "TP2",
            "dens",
            1,
            None,
            None,
            "the density is TP2 (positive likelihood ratio dependence)",
            "Nelsen (2006), Sect. 5.2.3; Joe (1997), Sect. 2.1",
        ),
        _mk(
            "RR2",
            "dens",
            -1,
            None,
            None,
            "the density is reverse regular of order 2",
            "Joe (1997), Sect. 2.1",
        ),
    ]
    return out


#: canonical key -> :class:`DependenceProperty`, ordered from weak to strong
PROPERTIES: dict[str, DependenceProperty] = {p.key: p for p in _catalogue()}

#: evaluation order (weakest first) used by :func:`dependence_profile`
_ORDER = (
    "PQD",
    "NQD",
    "LTD(V|U)",
    "LTD(U|V)",
    "RTI(V|U)",
    "RTI(U|V)",
    "LTI(V|U)",
    "LTI(U|V)",
    "RTD(V|U)",
    "RTD(U|V)",
    "SI(V|U)",
    "SI(U|V)",
    "SD(V|U)",
    "SD(U|V)",
    "LCSD",
    "RCSI",
    "TP2",
    "RR2",
)

_HIER = "Nelsen (2006), Sect. 5.2 and Thm. 5.2.19; Joe (1997), Thm. 2.3"

#: implications ``(P, Q)`` meaning "P implies Q" (see module docstring)
IMPLICATIONS: tuple[tuple[str, str], ...] = (
    ("TP2", "SI(V|U)"),
    ("TP2", "SI(U|V)"),
    ("TP2", "LCSD"),
    ("TP2", "RCSI"),
    ("SI(V|U)", "LTD(V|U)"),
    ("SI(V|U)", "RTI(V|U)"),
    ("SI(U|V)", "LTD(U|V)"),
    ("SI(U|V)", "RTI(U|V)"),
    ("LCSD", "LTD(V|U)"),
    ("LCSD", "LTD(U|V)"),
    ("RCSI", "RTI(V|U)"),
    ("RCSI", "RTI(U|V)"),
    ("LTD(V|U)", "PQD"),
    ("LTD(U|V)", "PQD"),
    ("RTI(V|U)", "PQD"),
    ("RTI(U|V)", "PQD"),
    ("RR2", "SD(V|U)"),
    ("RR2", "SD(U|V)"),
    ("SD(V|U)", "LTI(V|U)"),
    ("SD(V|U)", "RTD(V|U)"),
    ("SD(U|V)", "LTI(U|V)"),
    ("SD(U|V)", "RTD(U|V)"),
    ("LTI(V|U)", "NQD"),
    ("LTI(U|V)", "NQD"),
    ("RTD(V|U)", "NQD"),
    ("RTD(U|V)", "NQD"),
)

_POSITIVE = tuple(k for k in _ORDER if PROPERTIES[k].sign > 0)
_NEGATIVE = tuple(k for k in _ORDER if PROPERTIES[k].sign < 0)

_ALIASES = {
    "PLOD": "PQD",
    "PUOD": "PQD",
    "NLOD": "NQD",
    "NUOD": "NQD",
    "CI": "SI",
    "CIS": "SI",
    "CD": "SD",
    "CDS": "SD",
    "TP2_CDF": "LCSD",
    "TP2CDF": "LCSD",
    "TP2_SURVIVAL": "RCSI",
    "TP2SURVIVAL": "RCSI",
    "TP2_DENSITY": "TP2",
    "TP2DENSITY": "TP2",
    "PLR": "TP2",
    "RR2_DENSITY": "RR2",
    "RR2DENSITY": "RR2",
    "NLR": "RR2",
}
_CONDITIONED = {"LTD", "LTI", "RTI", "RTD", "SI", "SD"}


def resolve_property(prop: str | DependenceProperty, i: int | None = None) -> str:
    """Canonical key of a property name.

    Parameters
    ----------
    prop : str
        A key (``"SI(V|U)"``), a concept (``"SI"``, combined with ``i``) or an
        alias (``"CIS"``, ``"TP2_cdf"``, ``"PLR"``, ...); case-insensitive.
    i : {1, 2}, optional
        Conditioning variable for conditioned concepts (default 1).

    Returns
    -------
    str

    Examples
    --------
    >>> resolve_property("si", i=2)
    'SI(U|V)'
    >>> resolve_property("tp2_cdf")
    'LCSD'
    """
    if isinstance(prop, DependenceProperty):
        return prop.key
    s = str(prop).strip().upper().replace(" ", "").replace("-", "_")
    if s in PROPERTIES:
        if i is not None and PROPERTIES[s].cond is not None and PROPERTIES[s].cond != i:
            raise ValueError(f"{prop!r} conflicts with i={i}")
        return s
    cond = None
    for tag, c in (("(V|U)", 1), ("(U|V)", 2), ("_V_U", 1), ("_U_V", 2)):
        if s.endswith(tag):
            s, cond = s[: -len(tag)], c
            break
    s = _ALIASES.get(s, s)
    if s in _CONDITIONED:
        if cond is not None and i is not None and cond != i:
            raise ValueError(f"{prop!r} conflicts with i={i}")
        c = cond or i or 1
        if c not in (1, 2):
            raise ValueError("i must be 1 or 2")
        return f"{s}({'V|U' if c == 1 else 'U|V'})"
    if s in PROPERTIES and cond is None:
        return s
    raise KeyError(f"Unknown dependence property {prop!r}. Known: {list(PROPERTIES)}")


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class PropertyResult:
    """Outcome of a dependence-property (or ordering) check.

    Attributes
    ----------
    property : str
        Canonical property key (or the name of the ordering).
    holds : bool
        Whether the property holds (for ``method="grid"``: on the grid).
    method : str
        ``"exact"``, ``"symbolic"``, ``"grid"`` (or ``"numeric"`` for checks
        based on numerically evaluated functionals).
    worst_violation : float
        Largest violation of the defining inequality that was found
        (``0.0`` if none; in the natural scale of the checked quantity, e.g.
        :math:`uv - C(u,v)` for PQD).  Values below the tolerance do not
        refute the property.
    where : dict or None
        Location of the worst violation (grid checks), e.g.
        ``{"u": 0.3, "v": 0.6}``.
    reason : str
        Citation / explanation of an exact result or a short description of
        the numerical check.
    info : dict
        Additional diagnostics (tolerance, number of evaluations, ...).
    """

    property: str
    holds: bool
    method: str
    worst_violation: float = 0.0
    where: dict | None = None
    reason: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.holds)

    def __repr__(self) -> str:
        extra = ""
        if self.method in ("grid", "numeric") and self.worst_violation:
            extra = f", worst_violation={self.worst_violation:.3g}"
        return f"PropertyResult({self.property}={self.holds}, method={self.method!r}{extra})"


def _exact(key, holds, reason, method="exact", **info) -> PropertyResult:
    return PropertyResult(key, bool(holds), method, 0.0, None, reason, dict(info))


# ---------------------------------------------------------------------------
# implication closure
# ---------------------------------------------------------------------------


def _close(facts: dict[str, PropertyResult]) -> dict[str, PropertyResult]:
    """Close a set of facts under :data:`IMPLICATIONS` (modus ponens/tollens)."""
    facts = dict(facts)
    changed = True
    while changed:
        changed = False
        for p, q in IMPLICATIONS:
            fp, fq = facts.get(p), facts.get(q)
            if fp is not None and fp.holds and fq is None:
                facts[q] = _exact(q, True, f"implied by {p} ({_HIER})", fp.method, source=p)
                changed = True
            elif fq is not None and not fq.holds and fp is None:
                facts[p] = _exact(
                    p, False, f"{q} fails and {p} implies {q} ({_HIER})", fq.method, source=q
                )
                changed = True
    return facts


def implication_violations(results: dict[str, Any]) -> list[tuple[str, str]]:
    """Implications ``P => Q`` violated by ``results`` (``P`` holds, ``Q`` fails).

    ``results`` maps canonical keys to booleans or :class:`PropertyResult`.
    """
    out = []
    for p, q in IMPLICATIONS:
        if p in results and q in results and bool(results[p]) and not bool(results[q]):
            out.append((p, q))
    return out


def _all(holds: bool, keys, reason, method="exact", **info):
    return {k: _exact(k, holds, reason, method, **info) for k in keys}


# ---------------------------------------------------------------------------
# helpers on copula objects
# ---------------------------------------------------------------------------


def _free_parameters(C) -> list:
    from copul.measures.backend import free_parameters

    return free_parameters(C)


def _is_specified_bivariate(C) -> bool:
    """Whether ``C`` is a fully specified bivariate copula object of :mod:`copul`."""
    from copul.family.core.biv_core_copula import BivCoreCopula

    try:
        return isinstance(C, BivCoreCopula) and not _free_parameters(C)
    except Exception:
        return False


def _require_specified(C) -> None:
    free = _free_parameters(C)
    if free:
        raise ValueError(
            f"{type(C).__name__} has free parameters {free}; dependence checks need a "
            "fully specified copula."
        )


def _is_ac(C) -> bool:
    from copul.family.core.numeric_api import is_absolutely_continuous

    return is_absolutely_continuous(C)


def _float_attr(C, name):
    try:
        return float(getattr(C, name))
    except Exception:
        return None


def _cdf_point(C, u, v) -> float:
    from copul.measures.backend import numeric_backend

    return float(np.asarray(numeric_backend(C).cdf(np.array([u]), np.array([v])))[0])


def _is_independence_value(C) -> bool:
    """``C(1/2,1/2) = 1/4`` (used for families where this characterizes Pi)."""
    try:
        return abs(_cdf_point(C, 0.5, 0.5) - 0.25) < 1e-13
    except Exception:
        return False


def _cache_get(C, name):
    from copul.measures.backend import _param_key

    try:
        hit = C.__dict__.get(name)
    except AttributeError:
        return None, None
    key = _param_key(C)
    if hit is not None and hit[0] == key:
        return key, hit[1]
    return key, None


def _cache_set(C, name, key, value):
    try:
        C.__dict__[name] = (key, value)
    except AttributeError:  # pragma: no cover - objects without __dict__
        pass


# ---------------------------------------------------------------------------
# exact facts of specific families
# ---------------------------------------------------------------------------

_PI_REASON = "C = Pi: equality in every defining inequality (density 1)"


def _independence_facts():
    return _all(True, _ORDER, _PI_REASON)


def _upper_frechet_facts():
    facts = _all(True, _POSITIVE, "C = M (Nelsen 2006, Sect. 5.2)")
    facts["TP2"] = _exact("TP2", False, "M has no density")
    facts["NQD"] = _exact("NQD", False, "M > Pi at (1/2, 1/2)")
    return facts


def _lower_frechet_facts():
    facts = _all(True, _NEGATIVE, "C = W (Nelsen 2006, Sect. 5.2)")
    facts["RR2"] = _exact("RR2", False, "W has no density")
    facts["PQD"] = _exact("PQD", False, "W < Pi at (1/2, 1/2)")
    return facts


def _sign_facts(sign: int, reason: str, method="exact"):
    """``sign > 0``: TP2 density; ``sign < 0``: RR2; with the opposite QD failing."""
    if sign > 0:
        return {
            "TP2": _exact("TP2", True, reason, method),
            "NQD": _exact("NQD", False, reason + "; C >= Pi and C != Pi", method),
        }
    return {
        "RR2": _exact("RR2", True, reason, method),
        "PQD": _exact("PQD", False, reason + "; C <= Pi and C != Pi", method),
    }


def _gaussian_facts(C):
    rho = _float_attr(C, "rho")
    if rho is None:
        return {}
    if rho == 0:
        return _independence_facts()
    reason = (
        "Gaussian copula: the density is TP2 iff rho >= 0 and RR2 iff rho <= 0 "
        "(mixed log-derivative rho/(1-rho^2); Karlin & Rinott 1980)"
    )
    return _sign_facts(1 if rho > 0 else -1, reason)


def _fgm_facts(C):
    th = _float_attr(C, "theta")
    if th is None:
        return {}
    if th == 0:
        return _independence_facts()
    reason = (
        "FGM copula: c = 1 + theta(1-2u)(1-2v) gives "
        "c11 c22 - c12 c21 = 4 theta (u2-u1)(v2-v1), so the density is TP2 iff "
        "theta >= 0 and RR2 iff theta <= 0"
    )
    return _sign_facts(1 if th > 0 else -1, reason)


def _student_t_facts(C):
    rho = _float_attr(C, "rho")
    if rho is None:
        return {}
    ref = (
        "Student-t copula: tau = (2/pi) arcsin(rho) (Lindskog, McNeil & Schmock 2003) "
        "and tau is monotone in the concordance order (Nelsen 2006, Sect. 5.1)"
    )
    if rho < 0:
        return {"PQD": _exact("PQD", False, ref + "; tau < 0")}
    if rho > 0:
        return {"NQD": _exact("NQD", False, ref + "; tau > 0")}
    reason = (
        "Student-t copula with rho = 0: (X, -Y) has the same law, hence "
        "C(u,v) = u - C(u,1-v), Spearman's rho is 0 and C != Pi, so C - Pi changes sign"
    )
    return {"PQD": _exact("PQD", False, reason), "NQD": _exact("NQD", False, reason)}


def _ev_facts(C):
    if _is_independence_value(C):
        return _independence_facts()
    ref_si = "bivariate extreme-value copulas are SI (Garralda-Guillem 2000)"
    ref_cs = (
        "bivariate extreme-value copulas are LCSD: log C = -l(x, y) with the convex, "
        "1-homogeneous stable tail dependence function l, so l_xy = -(x/y) l_xx <= 0 "
        "(Joe 1997, Ch. 6)"
    )
    facts = {
        "SI(V|U)": _exact("SI(V|U)", True, ref_si),
        "SI(U|V)": _exact("SI(U|V)", True, ref_si),
        "LCSD": _exact("LCSD", True, ref_cs),
        "NQD": _exact("NQD", False, "extreme-value copulas satisfy C >= Pi, and C != Pi"),
    }
    return facts


def _strip_max0(expr):
    import sympy as sp

    def rep(x):
        args = [a for a in x.args if not (a.is_number and a == 0)]
        return args[0] if len(args) == 1 else sp.Max(*args)

    return expr.replace(
        lambda x: isinstance(x, sp.Max) and any(a.is_number and a == 0 for a in x.args), rep
    )


def _archimedean_curvatures(C, n_points: int = 121, dps: int = 30):
    r"""Normalized log-curvatures of :math:`\psi`, :math:`-\psi'` and :math:`\psi''`.

    For :math:`g\in\{\psi, -\psi', \psi''\}` the function :math:`\log g` is
    convex iff :math:`r = (g g'' - g'^2)/(g'^2 + |g g''|)\ge 0`.  Evaluated
    in ``dps``-digit arithmetic (mpmath) at :math:`s=\varphi(u)` for
    ``n_points`` values of :math:`u` equally spaced in logit scale between
    :math:`10^{-10}` and :math:`1-10^{-10}`; points where the family's
    expressions of :math:`\psi` and :math:`\varphi` are not mutually
    consistent to :math:`10^{-9}` (relative) are discarded.

    Returns ``{k: (min_r, max_r, u_at_min, u_at_max, n_used)}`` for
    ``k = 0, 1, 2`` or ``None``.
    """
    import mpmath as mp
    import sympy as sp

    y = getattr(C, "y", None)
    t = getattr(C, "t", None)
    psi = getattr(C.inv_generator, "func", None)
    gen = getattr(C.generator, "func", None)
    if not isinstance(psi, sp.Expr) or not isinstance(gen, sp.Expr) or y is None:
        return None
    if isinstance(gen, sp.Piecewise):
        gen = gen.args[0][0]
    psi = _strip_max0(psi)
    if psi.free_symbols - {y} or gen.free_symbols - {t} or psi.has(sp.Piecewise):
        return None
    derivs = [psi]
    for _ in range(4):
        derivs.append(sp.diff(derivs[-1], y))
    fs = [sp.lambdify(y, e, modules="mpmath") for e in derivs]
    phi = sp.lambdify(t, gen, modules="mpmath")
    acc: dict[int, list] = {0: [], 1: [], 2: []}
    with mp.workdps(dps):
        for z in np.linspace(-23.0, 23.0, n_points):
            u = 1 / (1 + mp.exp(-mp.mpf(z)))
            try:
                s = phi(u)
                if isinstance(s, mp.mpc) or not mp.isfinite(s) or s < 0:
                    continue
                vals = [f(s) for f in fs]
                if any(isinstance(x, mp.mpc) or not mp.isfinite(x) for x in vals):
                    continue
                if abs(vals[0] - u) > 1e-9 * u:  # inconsistent psi / phi expressions
                    continue
            except Exception:
                continue
            for k in range(3):
                sg = -1 if k == 1 else 1
                g, g1, g2 = sg * vals[k], sg * vals[k + 1], sg * vals[k + 2]
                if g <= 0:
                    continue
                q = g * g2 - g1**2
                nrm = g1**2 + abs(g * g2)
                if nrm == 0:
                    continue
                acc[k].append((float(q / nrm), float(u)))
    out = {}
    for k, vals in acc.items():
        if len(vals) < n_points // 3:
            return None
        rs = np.array([a for a, _ in vals])
        us = np.array([b for _, b in vals])
        out[k] = (float(rs.min()), float(rs.max()), float(us[rs.argmin()]), float(us[rs.argmax()]))
        out[k] = (*out[k], len(vals))
    return out


_LT_FAMILIES = None


def _lt_family_rule(C):
    """Exact TP2 for named Archimedean families with Laplace-transform generators."""
    global _LT_FAMILIES
    if _LT_FAMILIES is None:
        from copul.family.archimedean import AliMikhailHaq, BivClayton, Frank, GumbelHougaard, Joe

        _LT_FAMILIES = (
            (BivClayton, lambda t: t > 0, "gamma"),
            (GumbelHougaard, lambda t: t >= 1, "positive stable"),
            (Frank, lambda t: t > 0, "logarithmic series"),
            (Joe, lambda t: t >= 1, "Sibuya"),
            (AliMikhailHaq, lambda t: 0 <= t < 1, "geometric"),
        )
    th = _float_attr(C, "theta")
    if th is None:
        return None
    for cls, ok, frailty in _LT_FAMILIES:
        if type(C) is cls and ok(th):
            return frailty
    return None


def _archimedean_facts(C):
    import sympy as sp

    facts: dict[str, PropertyResult] = {}
    frailty = _lt_family_rule(C)
    if frailty is not None:
        if _is_independence_value(C):
            return _independence_facts()
        reason = (
            f"{type(C).__name__}: psi is the Laplace transform of a {frailty} frailty, "
            "hence completely monotone and psi'' log-convex (Bernstein's theorem), so the "
            "density psi''(phi(u)+phi(v)) phi'(u) phi'(v) is TP2 (Müller & Scarsini 2005)"
        )
        return _sign_facts(1, reason)
    try:
        gen0 = C._generator_at_0
        strict = gen0 == sp.oo or (isinstance(gen0, float) and math.isinf(gen0))
    except Exception:
        return facts
    if not strict:
        facts["PQD"] = _exact(
            "PQD",
            False,
            "non-strict Archimedean generator: C(u,v) = 0 < uv on "
            "{phi(u) + phi(v) >= phi(0)} (Nelsen 2006, Sect. 4.2)",
        )
    try:
        curv = _archimedean_curvatures(C)
    except Exception as e:  # pragma: no cover - defensive
        log.debug("Archimedean curvature check failed for %s: %s", type(C).__name__, e)
        curv = None
    if curv is None:
        return facts
    tol = 1e-8
    names = {0: "log(psi)", 1: "log(-psi')", 2: "log(psi'')"}
    pos_keys = {0: ("LTD(V|U)", "LTD(U|V)", "LCSD"), 1: ("SI(V|U)", "SI(U|V)"), 2: ("TP2",)}
    neg_keys = {0: ("LTI(V|U)", "LTI(U|V)"), 1: ("SD(V|U)", "SD(U|V)"), 2: ("RR2",)}
    ref = "Capéraà & Genest (1993); Müller & Scarsini (2005)"
    ac = _is_ac(C)
    for k, (rmin, rmax, umin, umax, n_used) in curv.items():
        info = {"r_min": rmin, "r_max": rmax, "u_at_min": umin, "u_at_max": umax, "n": n_used}
        if strict:
            convex = rmin >= -tol
            how = "convex" if convex else f"not convex (r={rmin:.3g} at u={umin:.3g})"
            for key in pos_keys[k]:
                holds = convex and (key != "TP2" or ac)
                facts[key] = _exact(
                    key,
                    holds,
                    f"Archimedean: {key} iff {names[k]} is convex; {names[k]} is {how} ({ref})",
                    "symbolic",
                    **info,
                )
        concave = rmax <= tol
        how = "concave" if concave else f"not concave (r={rmax:.3g} at u={umax:.3g})"
        for key in neg_keys[k]:
            holds = concave and (key != "RR2" or ac)
            facts[key] = _exact(
                key,
                holds,
                f"Archimedean: {key} iff {names[k]} is concave on [0, phi(0)); "
                f"{names[k]} is {how} ({ref})",
                "symbolic",
                **info,
            )
    return facts


def _lt_archimedean_facts(C):
    if _is_independence_value(C):
        return _independence_facts()
    reason = (
        f"{type(C).__name__}: Laplace-transform generator (completely monotone psi), so "
        "psi'' is log-convex and the density is TP2 (Müller & Scarsini 2005)"
    )
    return _sign_facts(1, reason)


def _matrix_tp2(P, sign=1, rtol=1e-12):
    """All-pairs TP2 (``sign=1``) / RR2 (``sign=-1``) test of a nonnegative matrix."""
    P = np.asarray(P, dtype=float)
    m = P.shape[0]
    for i in range(m - 1):
        a = P[i][None, :, None] * P[i + 1 :][:, None, :]  # P[i,j] P[i',j']
        b = P[i][None, None, :] * P[i + 1 :][:, :, None]  # P[i,j'] P[i',j]
        d = sign * (a - b)
        jj = np.triu(np.ones((P.shape[1], P.shape[1]), dtype=bool), k=1)
        if np.any((d < -rtol * (a + b))[:, jj]):
            return False
    return True


def _node_minors_ok(G, rtol=1e-12):
    a = G[:-1, :-1] * G[1:, 1:]
    b = G[:-1, 1:] * G[1:, :-1]
    return bool(np.all(a - b >= -rtol * (a + b)))


def _checkerboard_matrix(C):
    """``(P, S)`` of a checkerboard copula or a straight shuffle of M, else ``None``."""
    from copul.checkerboard import _biv_engine as eng
    from copul.checkerboard._biv_mixin import BivCheckerboardMixin
    from copul.checkerboard.shuffle_min import ShuffleOfMin

    if isinstance(C, BivCheckerboardMixin):
        P = np.asarray(C.matr, dtype=float)
        P = P / P.sum()
        return P, eng._signs(P, C._kernel_signs())
    if isinstance(C, ShuffleOfMin):
        n = C.n
        P = np.zeros((n, n))
        P[np.arange(n), C.pi0] = 1.0 / n
        return P, np.ones((n, n), dtype=int)
    return None


def _checkerboard_facts(C, PS=None):
    r"""Exact facts of checkerboard copulas (incl. straight shuffles of M).

    Quadrant, tail and stochastic monotonicity use the exact algorithms of
    :mod:`copul.checkerboard._biv_engine` (``(U|V)`` via the transposed mass
    matrix).  The density of an independence-kernel checkerboard is the
    step function :math:`mn\Delta_{ij}`, so it is TP2 iff the mass matrix is
    TP2.  For independence-kernel checkerboards :math:`C` (and :math:`\bar C`)
    are bilinear on every cell; there
    :math:`C\,\partial_{uv}C-\partial_uC\,\partial_vC = C_{00}C_{11}-C_{10}C_{01}`
    is the constant 2x2 minor of the corner values, :math:`\log C` is
    continuous and piecewise smooth, and supermodularity is additive over
    rectangles, so ``LCSD`` (``RCSI``) holds iff the adjacent 2x2 minors of
    the node values :math:`C(k/m, l/n)` (of :math:`\bar C`) are nonnegative --
    provided these are positive in the interior (otherwise the grid check is
    used).
    """
    from copul.checkerboard import _biv_engine as eng

    P, S = PS if PS is not None else _checkerboard_matrix(C)
    ref = "exact checkerboard algorithm (copul.checkerboard._biv_engine)"
    facts = {
        "PQD": _exact("PQD", eng.quadrant_dependence(P, S, positive=True), ref),
        "NQD": _exact("NQD", eng.quadrant_dependence(P, S, positive=False), ref),
    }
    for cond in (1, 2):
        PP = P if cond == 1 else P.T
        SS = S if (S is None or cond == 1) else S.T
        tag = "V|U" if cond == 1 else "U|V"
        si, sd = eng.cis_direction(PP, SS, which=1)
        facts[f"SI({tag})"] = _exact(f"SI({tag})", si, ref)
        facts[f"SD({tag})"] = _exact(f"SD({tag})", sd, ref)
        for kind in ("ltd", "lti", "rti", "rtd"):
            key = f"{kind.upper()}({tag})"
            facts[key] = _exact(key, eng.tail_monotonicity(PP, SS, kind), ref)
    pure_pi = S is None or not np.any((S != 0) & (P > 0))
    if not pure_pi:
        facts["TP2"] = _exact("TP2", False, "singular kernel cells: no density")
        facts["RR2"] = _exact("RR2", False, "singular kernel cells: no density")
        return facts
    dref = "density m n Delta_ij is TP2 (RR2) iff the mass matrix is TP2 (RR2)"
    facts["TP2"] = _exact("TP2", _matrix_tp2(P, 1), dref)
    facts["RR2"] = _exact("RR2", _matrix_tp2(P, -1), dref)
    m, n = P.shape
    G = np.zeros((m + 1, n + 1))
    G[1:, 1:] = P.cumsum(0).cumsum(1)
    uu = np.arange(m + 1)[:, None] / m
    vv = np.arange(n + 1)[None, :] / n
    Sbar = 1.0 - uu - vv + G
    cref = (
        "independence-kernel checkerboard: node 2x2 minors decide TP2 of the "
        "piecewise bilinear function"
    )
    inner = G[1:, 1:]
    if np.all(inner > 1e-15):
        facts["LCSD"] = _exact("LCSD", _node_minors_ok(inner), cref)
    inner_s = Sbar[:-1, :-1]
    if np.all(inner_s > 1e-15):
        facts["RCSI"] = _exact("RCSI", _node_minors_ok(inner_s), cref)
    return facts


def _transfer_key(key: str, swap: bool, flip_u: bool, flip_v: bool) -> str | None:
    """Property of the base copula equivalent to ``key`` of the transformed one.

    The transformed pair is ``(X, Y)``: ``(U, V)`` (or ``(V, U)`` if ``swap``)
    followed by ``X -> 1-X`` (``flip_u``) and ``Y -> 1-Y`` (``flip_v``).  A
    flip of the conditioning variable exchanges left and right tails and
    reverses the sign; a flip of the other variable reverses the sign.
    """
    p = PROPERTIES[key]
    parity = (-1) ** (int(flip_u) + int(flip_v))
    if p.kind == "qd":
        return "PQD" if p.sign * parity > 0 else "NQD"
    if p.kind == "dens":
        return "TP2" if p.sign * parity > 0 else "RR2"
    if p.kind == "cs":
        if flip_u != flip_v:
            return None
        side = p.side if not flip_u else ("R" if p.side == "L" else "L")
        return "LCSD" if side == "L" else "RCSI"
    base_cond = (2 if p.cond == 1 else 1) if swap else p.cond
    fc = flip_u if p.cond == 1 else flip_v
    fo = flip_v if p.cond == 1 else flip_u
    sign = p.sign * (-1) ** (int(fc) + int(fo))
    tag = "V|U" if base_cond == 1 else "U|V"
    if p.kind == "sm":
        return f"{'SI' if sign > 0 else 'SD'}({tag})"
    side = p.side if not fc else ("R" if p.side == "L" else "L")
    concept = {("L", 1): "LTD", ("L", -1): "LTI", ("R", 1): "RTI", ("R", -1): "RTD"}[(side, sign)]
    return f"{concept}({tag})"


def _transformed_facts(C):
    base_facts = exact_facts(C.base)
    out = {}
    for key in _ORDER:
        bk = _transfer_key(key, C.swap, C.flip_u, C.flip_v)
        if bk is not None and bk in base_facts:
            f = base_facts[bk]
            out[key] = _exact(
                key,
                f.holds,
                f"{key} of the transformed copula <=> {bk} of the base copula ({f.reason})",
                f.method,
            )
    return out


_LINEAR_KINDS = ("qd", "sm", "lt", "rt")


def _mixture_facts(C):
    comps = [C.copulas[i] for i in getattr(C, "_active", range(len(C.copulas)))]
    cf = [exact_facts(c) for c in comps]
    out = {}
    for key in _ORDER:
        if PROPERTIES[key].kind not in _LINEAR_KINDS:
            continue
        fs = [f.get(key) for f in cf]
        if fs and all(f is not None and f.holds for f in fs):
            method = "symbolic" if any(f.method == "symbolic" for f in fs) else "exact"
            out[key] = _exact(
                key,
                True,
                "every mixture component has the property, whose defining inequality "
                "is linear in C",
                method,
            )
    return out


def _family_facts(C) -> dict[str, PropertyResult]:
    from copul.checkerboard.shuffle_min import ShuffleOfMin
    from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
    from copul.family.bb.lt_archimedean import LTArchimedeanCopula
    from copul.family.constructions.mixture import MixtureCopula
    from copul.family.constructions.rotation import TransformedCopula
    from copul.family.elliptical.gaussian import Gaussian
    from copul.family.elliptical.student_t import StudentT
    from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
    from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
    from copul.family.frechet.frechet import Frechet
    from copul.family.frechet.lower_frechet import LowerFrechet
    from copul.family.frechet.upper_frechet import UpperFrechet
    from copul.family.other.farlie_gumbel_morgenstern import FarlieGumbelMorgenstern
    from copul.family.other.independence_copula import IndependenceCopula

    if isinstance(C, (BivIndependenceCopula, IndependenceCopula)):
        return _independence_facts()
    if isinstance(C, UpperFrechet):
        return _upper_frechet_facts()
    if isinstance(C, LowerFrechet):
        return _lower_frechet_facts()
    if isinstance(C, Frechet) and type(C).__name__ == "Frechet":
        a, b = _float_attr(C, "alpha"), _float_attr(C, "beta")
        if a == 0 and b == 0:
            return _independence_facts()
        if a == 1:
            return _upper_frechet_facts()
        if b == 1:
            return _lower_frechet_facts()
    if isinstance(C, ShuffleOfMin):
        if C.is_identity:
            return _upper_frechet_facts()
        if C.is_reverse:
            return _lower_frechet_facts()
    if _checkerboard_matrix(C) is not None:
        return _checkerboard_facts(C)
    if isinstance(C, Gaussian):
        return _gaussian_facts(C)
    if isinstance(C, StudentT):
        return _student_t_facts(C)
    if isinstance(C, FarlieGumbelMorgenstern):
        return _fgm_facts(C)
    if isinstance(C, TransformedCopula):
        return _transformed_facts(C)
    if isinstance(C, MixtureCopula):
        return _mixture_facts(C)
    if isinstance(C, LTArchimedeanCopula):
        return _lt_archimedean_facts(C)
    if isinstance(C, BivArchimedeanCopula):
        return _archimedean_facts(C)
    if isinstance(C, BivExtremeValueCopula):
        return _ev_facts(C)
    return {}


def exact_facts(C) -> dict[str, PropertyResult]:
    """All properties of ``C`` decided without a grid search.

    Combines the family characterizations listed in the module docstring
    with the absence of a density (no ``TP2``/``RR2`` for copulas that are
    not absolutely continuous) and closes the result under
    :data:`IMPLICATIONS`.  Cached on the copula instance.

    Returns
    -------
    dict
        Canonical key -> :class:`PropertyResult` with ``method`` ``"exact"``
        or ``"symbolic"``.
    """
    _require_specified(C)
    key, hit = _cache_get(C, "_copul_dependence_facts")
    if hit is not None:
        return dict(hit)
    try:
        facts = _family_facts(C)
    except Exception as e:  # pragma: no cover - defensive
        log.debug("family facts failed for %s: %s", type(C).__name__, e)
        facts = {}
    facts = dict(facts)
    if not _is_ac(C):
        for k in ("TP2", "RR2"):
            facts.setdefault(k, _exact(k, False, "not absolutely continuous: no density"))
    closed = _close(facts)
    bad = implication_violations(closed)
    if bad:  # pragma: no cover - would indicate a wrong characterization
        log.warning("inconsistent exact facts for %s: %s", type(C).__name__, bad)
    _cache_set(C, "_copul_dependence_facts", key, closed)
    return dict(closed)


# ---------------------------------------------------------------------------
# grid checks
# ---------------------------------------------------------------------------

_DEFAULT_N = 65
#: default base tolerances by ingredient, (closed/symbolic source, finite differences)
_TOLS = {"cdf": (1e-10, 1e-10), "h": (1e-9, 1e-6), "pdf": (1e-6, 1e-3)}


def _nodes(n: int) -> np.ndarray:
    """Chebyshev--Gauss--Lobatto interior nodes on (0, 1) (clustered at 0 and 1)."""
    k = np.arange(1, n + 1)
    return 0.5 * (1.0 - np.cos(np.pi * k / (n + 1)))


class _Grid:
    """Numerical ingredients of a copula on a tensor grid (lazily, cached)."""

    def __init__(self, C, n: int):
        from copul.measures.backend import numeric_backend

        self.C = C
        self.be = numeric_backend(C)
        self.n = int(n)
        self.x = _nodes(self.n)
        #: refinements stay within [lo, 1 - lo] (half the outermost node)
        self.lo = 0.5 * float(self.x[0])
        self._g: dict[str, np.ndarray] = {}
        self.n_evals = 0

    def f(self, what):
        return self.be.get(what)

    def eval(self, what, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        self.n_evals += u.size
        with np.errstate(all="ignore"):
            out = np.asarray(self.f(what)(u, v), dtype=float)
        return np.broadcast_to(out, u.shape).astype(float)

    def grid(self, what):
        g = self._g.get(what)
        if g is None:
            U, V = np.meshgrid(self.x, self.x, indexing="ij")
            g = self.eval(what, U, V)
            self._g[what] = g
        return g

    def tol(self, what):
        src = self.be.source.get(what if what != "h" else "h1", "")
        if what == "h":
            src = self.be.source.get("h1", "")
            if self.be.source.get("h2", "") == "finite_differences":
                src = "finite_differences"
        lo, hi = _TOLS["h" if what in ("h", "h1", "h2") else what]
        return hi if src == "finite_differences" else lo


def _grid_result(key, holds, worst, where, desc, tol, grid, **info):
    info.update({"tol": tol, "n_grid": grid.n, "n_evals": grid.n_evals})
    return PropertyResult(key, bool(holds), "grid", float(max(worst, 0.0)), where, desc, info)


def _zoom_points(center, radius, k=7, lo=1e-9):
    s = np.linspace(-radius, radius, k)
    return np.clip(center + s, lo, 1 - lo)


def _check_qd(g: _Grid, key, tol, refine):
    s = PROPERTIES[key].sign
    U, V = np.meshgrid(g.x, g.x, indexing="ij")
    slack = s * (g.grid("cdf") - U * V)
    flat = np.argsort(slack, axis=None)[:4]
    best = float(slack.flat[flat[0]])
    where = {"u": float(U.flat[flat[0]]), "v": float(V.flat[flat[0]])}
    if refine:
        h = 1.0 / g.n
        for idx in flat:
            cu, cv = float(U.flat[idx]), float(V.flat[idx])
            r = 2 * h
            for _ in range(6):
                pu, pv = np.meshgrid(
                    _zoom_points(cu, r, lo=g.lo), _zoom_points(cv, r, lo=g.lo), indexing="ij"
                )
                sl = s * (g.eval("cdf", pu, pv) - pu * pv)
                j = int(np.nanargmin(sl))
                if sl.flat[j] < best:
                    best = float(sl.flat[j])
                    where = {"u": float(pu.flat[j]), "v": float(pv.flat[j])}
                cu, cv = float(pu.flat[j]), float(pv.flat[j])
                r *= 0.4
    holds = best >= -tol
    desc = "grid check of C(u,v) " + (">= uv" if s > 0 else "<= uv")
    return _grid_result(key, holds, -best, where, desc, tol, g)


def _pair_violations(Gc, x, kind, sign):
    """Violations over all pairs ``a < a'`` along axis 0 of ``Gc[a, b]``.

    Returns ``(viol, allowed_scale, scale)`` arrays of shape ``(n, n, n_b)``
    with ``viol > allowed`` meaning a violation (``allowed`` = ``tol *
    allowed_scale``) and ``viol / scale`` the violation on the natural scale.
    """
    xa = x[:, None, None]
    xb = x[None, :, None]
    A = Gc[:, None, :]
    B = Gc[None, :, :]
    if kind == "sm":
        # nonincreasing (sign +1) / nondecreasing (sign -1) in the conditioning variable
        d = B - A
        return sign * d, np.full_like(d, 2.0), np.ones_like(d)
    if kind == "lt":
        # C(a,b)/a nonincreasing: D = a C(a',b) - a' C(a,b) <= 0
        d = xa * B - xb * A
        return sign * d, np.broadcast_to(xa + xb, d.shape), np.broadcast_to(xa * xb, d.shape)
    # rt: S(a,b)/(1-a) nondecreasing: D = (1-a) S(a',b) - (1-a') S(a,b) >= 0
    d = (1 - xa) * B - (1 - xb) * A
    return (
        -sign * d,
        np.broadcast_to(2 - xa - xb, d.shape),
        np.broadcast_to((1 - xa) * (1 - xb), d.shape),
    )


def _line_values(g: _Grid, key, a, b):
    """Checked function along the conditioning coordinate ``a`` at other coordinate ``b``."""
    p = PROPERTIES[key]
    A, B = np.meshgrid(a, b, indexing="ij")
    U, V = (A, B) if p.cond == 1 else (B, A)
    if p.kind == "sm":
        return g.eval("h1", U, V) if p.cond == 1 else g.eval("h2", U, V)
    c = g.eval("cdf", U, V)
    if p.kind == "rt":
        return 1.0 - U - V + c
    return c


def _check_monotone(g: _Grid, key, tol, refine):
    p = PROPERTIES[key]
    x = g.x
    if p.kind == "sm":
        Gc = g.grid("h1") if p.cond == 1 else g.grid("h2").T
    else:
        c = g.grid("cdf")
        Gc = c if p.cond == 1 else c.T
        if p.kind == "rt":
            U, V = np.meshgrid(x, x, indexing="ij")
            Gc = 1.0 - U - V + Gc
    viol, allowed, scale = _pair_violations(Gc, x, p.kind, p.sign)
    mask = np.triu(np.ones((g.n, g.n), dtype=bool), k=1)[:, :, None]
    excess = np.where(mask, viol - tol * allowed, -np.inf)
    nat = np.where(mask, viol / scale, -np.inf)
    best_excess = float(np.max(excess))
    idx = np.unravel_index(int(np.argmax(excess)), excess.shape)
    worst = float(nat[idx])
    where = {"cond": (float(x[idx[0]]), float(x[idx[1]])), "other": float(x[idx[2]])}
    if refine:
        per_line = excess.reshape(-1, excess.shape[2])
        line_best = per_line.max(axis=0)
        for jb in np.argsort(line_best)[::-1][:3]:
            ia, ib = np.unravel_index(int(np.argmax(per_line[:, jb])), excess.shape[:2])
            lo = x[max(ia - 1, 0)]
            hi = x[min(ib + 1, g.n - 1)]
            a = np.unique(np.concatenate([x, np.linspace(lo, hi, 49)]))
            bs = [x[jb]]
            if jb > 0:
                bs.append(0.5 * (x[jb - 1] + x[jb]))
            if jb < g.n - 1:
                bs.append(0.5 * (x[jb] + x[jb + 1]))
            b = np.array(bs)
            Gl = _line_values(g, key, a, b)
            v2, al2, sc2 = _pair_violations(Gl, a, p.kind, p.sign)
            m2 = np.triu(np.ones((a.size, a.size), dtype=bool), k=1)[:, :, None]
            ex2 = np.where(m2, v2 - tol * al2, -np.inf)
            k2 = np.unravel_index(int(np.argmax(ex2)), ex2.shape)
            if ex2[k2] > best_excess:
                best_excess = float(ex2[k2])
                worst = float(v2[k2] / sc2[k2])
                where = {"cond": (float(a[k2[0]]), float(a[k2[1]])), "other": float(b[k2[2]])}
    holds = best_excess <= 0
    return _grid_result(key, holds, worst, where, f"grid check: {p.description}", tol, g)


def _tp2_adjacent(F, sign, abs_tol, rel_tol):
    """Worst violation over adjacent 2x2 minors (see :func:`_tp2_violation`)."""
    F = np.asarray(F, dtype=float)
    f11, f22 = F[:-1, :-1], F[1:, 1:]
    f12, f21 = F[:-1, 1:], F[1:, :-1]
    a = f11 * f22
    b = f12 * f21
    with np.errstate(all="ignore"):
        viol = sign * (b - a)
        allowed = abs_tol * (f11 + f22 + f12 + f21) + rel_tol * (np.abs(a) + np.abs(b))
        ex = np.where(np.isfinite(viol) & np.isfinite(allowed), viol - allowed, -np.inf)
    k = int(np.argmax(ex))
    if not np.isfinite(ex.flat[k]):
        return -np.inf, 0.0, None
    i, j = np.unravel_index(k, ex.shape)
    den = abs(a[i, j]) + abs(b[i, j])
    worst = float(viol[i, j] / den) if den > 0 else float(viol[i, j])
    return float(ex.flat[k]), worst, (int(i), int(i) + 1, int(j), int(j) + 1)


def _tp2_violation(F, sign, abs_tol, rel_tol):
    """Worst all-pairs TP2 (``sign=1``) / RR2 violation of a matrix ``F``.

    Returns ``(best_excess, worst_rel, (i, i2, j, j2))``; ``best_excess > 0``
    means a violation beyond the tolerance
    ``abs_tol * (F11 + F22 + F12 + F21) + rel_tol * (F11 F22 + F12 F21)``.
    Non-finite entries are ignored.
    """
    F = np.asarray(F, dtype=float)
    m, n = F.shape
    jj = np.triu(np.ones((n, n), dtype=bool), k=1)
    best, worst, loc = -np.inf, 0.0, None
    for i in range(m - 1):
        f1 = F[i]
        f2 = F[i + 1 :]
        a = f1[None, :, None] * f2[:, None, :]  # F[i,j] F[i2,j2]
        b = f1[None, None, :] * f2[:, :, None]  # F[i,j2] F[i2,j]
        ssum = f1[None, :, None] + f1[None, None, :] + f2[:, None, :] + f2[:, :, None]
        with np.errstate(all="ignore"):
            viol = sign * (b - a)
            allowed = abs_tol * ssum + rel_tol * (np.abs(a) + np.abs(b))
            ex = np.where(
                jj[None] & np.isfinite(viol) & np.isfinite(allowed), viol - allowed, -np.inf
            )
        k = int(np.argmax(ex))
        if ex.flat[k] > best:
            best = float(ex.flat[k])
            i2, j, j2 = np.unravel_index(k, ex.shape)
            den = abs(a[i2, j, j2]) + abs(b[i2, j, j2])
            worst = float(viol[i2, j, j2] / den) if den > 0 else float(viol[i2, j, j2])
            loc = (i, i + 1 + i2, j, j2)
    return best, worst, loc


def _tp2_function(g: _Grid, key):
    p = PROPERTIES[key]
    if p.kind == "dens":
        return lambda u, v: g.eval("pdf", u, v)
    if p.side == "L":
        return lambda u, v: g.eval("cdf", u, v)
    return lambda u, v: 1.0 - u - v + g.eval("cdf", u, v)


def _check_tp2(g: _Grid, key, tol, refine):
    p = PROPERTIES[key]
    x = g.x
    sign = p.sign
    if p.kind == "dens":
        F = g.grid("pdf")
        abs_tol, rel_tol = 0.0, tol
    else:
        F = g.grid("cdf")
        if p.side == "R":
            U, V = np.meshgrid(x, x, indexing="ij")
            F = 1.0 - U - V + F
        abs_tol, rel_tol = tol, 1e-12
    # adjacent 2x2 minors on the full grid (local violations) and all pairs on
    # every second node (violations between distant rows / columns and
    # matrices with zeros, where adjacent minors do not suffice)
    best, worst, loc = _tp2_adjacent(F, sign, abs_tol, rel_tol)
    sub = np.arange(0, x.size, 2)
    b2, w2, l2 = _tp2_violation(F[np.ix_(sub, sub)], sign, abs_tol, rel_tol)
    if l2 is not None and b2 > best:
        best, worst = b2, w2
        loc = tuple(int(sub[k]) for k in l2)
    where = None
    if loc is not None:
        i, i2, j, j2 = loc
        where = {"u": (float(x[i]), float(x[i2])), "v": (float(x[j]), float(x[j2]))}
    if refine and loc is not None:
        fn = _tp2_function(g, key)
        i, i2, j, j2 = loc
        cu = [float(x[i]), float(x[i2])]
        cv = [float(x[j]), float(x[j2])]
        r = 1.0 / g.n
        for _ in range(4):
            us = np.unique(np.concatenate([_zoom_points(c, r, 5, g.lo) for c in cu]))
            vs = np.unique(np.concatenate([_zoom_points(c, r, 5, g.lo) for c in cv]))
            Uu, Vv = np.meshgrid(us, vs, indexing="ij")
            Fl = fn(Uu, Vv)
            b2, w2, l2 = _tp2_violation(Fl, sign, abs_tol, rel_tol)
            if l2 is None:
                break
            a1, a2, c1, c2 = l2
            cu, cv = [float(us[a1]), float(us[a2])], [float(vs[c1]), float(vs[c2])]
            if b2 > best:
                best, worst = b2, w2
                where = {"u": tuple(cu), "v": tuple(cv)}
            r *= 0.5
    holds = best <= 0
    return _grid_result(key, holds, worst, where, f"grid check: {p.description}", tol, g)


def _grid_check(g: _Grid, key, tol=None, refine=True) -> PropertyResult:
    p = PROPERTIES[key]
    if p.kind == "dens" and not _is_ac(g.C):
        return _exact(key, False, "not absolutely continuous: no density")
    what = {"qd": "cdf", "lt": "cdf", "rt": "cdf", "cs": "cdf", "sm": "h", "dens": "pdf"}[p.kind]
    if tol is None:
        tol = g.tol(what)
    if p.kind == "qd":
        return _check_qd(g, key, tol, refine)
    if p.kind in ("lt", "rt", "sm"):
        return _check_monotone(g, key, tol, refine)
    return _check_tp2(g, key, tol, refine)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

_METHODS = ("auto", "exact", "grid")


def check_property(
    copula,
    prop: str,
    i: int | None = None,
    *,
    method: str = "auto",
    n_grid: int | None = None,
    tol: float | None = None,
    refine: bool = True,
) -> PropertyResult:
    r"""Check a dependence property of a bivariate copula.

    Parameters
    ----------
    copula : bivariate copula
        A fully specified copula object.
    prop : str
        Property key, concept or alias (see :func:`resolve_property`).
    i : {1, 2}, optional
        Conditioning variable of conditioned concepts (default 1, i.e.
        properties of :math:`V` given :math:`U`).
    method : {"auto", "exact", "grid"}
        ``"auto"`` uses an exact / symbolic characterization when available
        and the grid check otherwise; ``"exact"`` raises ``ValueError`` if no
        characterization applies; ``"grid"`` forces the grid check.
    n_grid : int, optional
        Number of grid points per axis of the grid check (default 65,
        Chebyshev nodes clustered at the boundary; refined locally near the
        worst points).
    tol : float, optional
        Tolerance of the grid check on the scale of the checked ingredient
        (absolute for :math:`C` and :math:`\partial_i C`, relative for the
        density); by default chosen from the accuracy of the numerical
        source (closed form vs. finite differences).
    refine : bool
        Refine the grid locally near the worst points.

    Returns
    -------
    PropertyResult
    """
    key = resolve_property(prop, i)
    method = str(method).lower()
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    _require_specified(copula)
    if method != "grid":
        facts = exact_facts(copula)
        if key in facts:
            return facts[key]
        if method == "exact":
            raise ValueError(f"No exact characterization of {key} for {type(copula).__name__}.")
    g = _Grid(copula, n_grid or _DEFAULT_N)
    return _grid_check(g, key, tol=tol, refine=refine)


def _make_check(concept, doc):
    def fn(copula, i: int = 1, **kwargs) -> PropertyResult:
        return check_property(copula, concept, i=i, **kwargs)

    fn.__name__ = f"is_{concept.lower()}"
    fn.__doc__ = doc + "\n\n    See :func:`check_property` for the keyword arguments.\n"
    return fn


def _make_plain(key, name, doc):
    def fn(copula, **kwargs) -> PropertyResult:
        return check_property(copula, key, **kwargs)

    fn.__name__ = name
    fn.__doc__ = doc + "\n\n    See :func:`check_property` for the keyword arguments.\n"
    return fn


is_pqd = _make_plain("PQD", "is_pqd", r"Positive quadrant dependence :math:`C\ge\Pi`.")
is_nqd = _make_plain("NQD", "is_nqd", r"Negative quadrant dependence :math:`C\le\Pi`.")
is_ltd = _make_check("LTD", "Left tail decreasing; ``i`` is the conditioning variable.")
is_lti = _make_check("LTI", "Left tail increasing; ``i`` is the conditioning variable.")
is_rti = _make_check("RTI", "Right tail increasing; ``i`` is the conditioning variable.")
is_rtd = _make_check("RTD", "Right tail decreasing; ``i`` is the conditioning variable.")
is_si = _make_check(
    "SI",
    r"Stochastically increasing: ``i=1`` checks that :math:`u\mapsto\partial_1C(u,v)` "
    "is nonincreasing (V stochastically increasing in U).",
)
is_sd = _make_check("SD", "Stochastically decreasing; ``i`` is the conditioning variable.")
is_lcsd = _make_plain("LCSD", "is_lcsd", "Left corner set decreasing (C is TP2).")
is_rcsi = _make_plain("RCSI", "is_rcsi", "Right corner set increasing (survival function TP2).")
is_tp2_cdf = _make_plain("LCSD", "is_tp2_cdf", "TP2 distribution function (= LCSD).")
is_tp2_survival = _make_plain("RCSI", "is_tp2_survival", "TP2 survival function (= RCSI).")
is_tp2_density = _make_plain("TP2", "is_tp2_density", "TP2 density (absolutely continuous only).")
is_rr2_density = _make_plain("RR2", "is_rr2_density", "RR2 density (absolutely continuous only).")


@dataclass
class DependenceProfile:
    """All dependence properties of a copula (see :func:`dependence_profile`).

    Attributes
    ----------
    copula : str
        Description of the copula.
    results : dict
        Canonical key -> :class:`PropertyResult` (weakest first).
    violations : list of tuple
        Implications ``(P, Q)`` of :data:`IMPLICATIONS` violated by the
        results (empty for a consistent profile).
    """

    copula: str
    results: dict[str, PropertyResult]
    violations: list[tuple[str, str]]

    @property
    def consistent(self) -> bool:
        """Whether the results respect the implication hierarchy."""
        return not self.violations

    def __getitem__(self, key) -> PropertyResult:
        return self.results[resolve_property(key)]

    def __contains__(self, key) -> bool:
        try:
            return resolve_property(key) in self.results
        except KeyError:
            return False

    def holds(self, key) -> bool:
        """Whether property ``key`` holds."""
        return bool(self[key])

    def as_dict(self) -> dict[str, bool]:
        """``{key: holds}``."""
        return {k: bool(r) for k, r in self.results.items()}

    def to_frame(self):
        """The profile as a :class:`pandas.DataFrame` (one row per property)."""
        import pandas as pd

        rows = [
            {
                "property": k,
                "holds": r.holds,
                "method": r.method,
                "worst_violation": r.worst_violation,
                "reason": r.reason,
            }
            for k, r in self.results.items()
        ]
        return pd.DataFrame(rows).set_index("property")

    def __repr__(self) -> str:
        lines = [f"DependenceProfile({self.copula})"]
        for k, r in self.results.items():
            lines.append(f"  {k:<10} {r.holds!s:<6} {r.method}")
        if self.violations:
            lines.append(f"  violated implications: {self.violations}")
        return "\n".join(lines)


def dependence_profile(
    copula,
    properties: Iterable[str] | None = None,
    *,
    method: str = "auto",
    n_grid: int | None = None,
    tol: float | None = None,
    refine: bool = True,
    propagate: bool = True,
) -> DependenceProfile:
    """Check all (or the given) dependence properties of a copula.

    Parameters
    ----------
    copula : bivariate copula
        Fully specified copula.
    properties : iterable of str, optional
        Subset of properties (default: all of :data:`PROPERTIES`).
    method : {"auto", "grid"}
        ``"grid"`` ignores exact characterizations.
    n_grid, tol, refine
        Grid-check settings, see :func:`check_property`.
    propagate : bool
        Evaluate from weak to strong properties and settle a property by
        modus tollens when an implied (weaker) property is already known to
        fail (the result then inherits that method).  With
        ``propagate=False`` every property is checked independently and
        :attr:`DependenceProfile.violations` tests the coherence of the
        individual checks.

    Returns
    -------
    DependenceProfile
    """
    method = str(method).lower()
    if method not in ("auto", "grid"):
        raise ValueError("method must be 'auto' or 'grid'")
    _require_specified(copula)
    keys = list(_ORDER) if properties is None else [resolve_property(p) for p in properties]
    keys = [k for k in _ORDER if k in set(keys)]
    facts = exact_facts(copula) if method == "auto" else {}
    implied_by = {}
    for p, q in IMPLICATIONS:
        implied_by.setdefault(p, []).append(q)
    g = None
    results: dict[str, PropertyResult] = {}
    for key in keys:
        if key in facts:
            results[key] = facts[key]
            continue
        if propagate:
            failing = [
                q for q in _implied_closure(key, implied_by) if q in results and not results[q]
            ]
            if failing:
                q = failing[0]
                results[key] = PropertyResult(
                    key,
                    False,
                    results[q].method,
                    0.0,
                    results[q].where,
                    f"{q} fails and {key} implies {q} ({_HIER})",
                    {"source": q},
                )
                continue
        if g is None:
            g = _Grid(copula, n_grid or _DEFAULT_N)
        results[key] = _grid_check(g, key, tol=tol, refine=refine)
    return DependenceProfile(str(copula), results, implication_violations(results))


def _implied_closure(key, implied_by):
    out, stack = [], list(implied_by.get(key, []))
    while stack:
        q = stack.pop()
        if q not in out:
            out.append(q)
            stack.extend(implied_by.get(q, []))
    return out
