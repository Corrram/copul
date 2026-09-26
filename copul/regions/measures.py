r"""
Canonical keys for bivariate dependence measures.

Every subpackage of the exact-region toolbox (:mod:`copul.regions`,
:mod:`copul.optim`, :mod:`copul.search`) refers to dependence measures by the
short canonical keys defined here:

=============  ====================================================================
key            definition (bivariate copula :math:`C`)
=============  ====================================================================
``"xi"``       Chatterjee's :math:`\xi(C)=6\int_0^1\!\int_0^1(\partial_1C)^2\,du\,dv-2`
``"rho"``      Spearman's :math:`\rho(C)=12\int\!\!\int C\,du\,dv-3`
``"tau"``      Kendall's :math:`\tau(C)=1-4\int\!\!\int\partial_1C\,\partial_2C\,du\,dv`
``"footrule"`` Spearman's footrule :math:`\psi(C)=6\int_0^1C(t,t)\,dt-2`
``"gamma"``    Gini's :math:`\gamma(C)=4\int_0^1[C(t,t)+C(t,1-t)]\,dt-2`
``"beta"``     Blomqvist's :math:`\beta(C)=4C(\tfrac12,\tfrac12)-1`
``"nu"``       Blest's :math:`\nu(C)=24\int\!\!\int(1-u)\,C(u,v)\,du\,dv-2`
=============  ====================================================================

Aliases such as ``"spearman"``, ``"psi"``, ``"gini"`` or ``"blest"`` are
accepted everywhere and resolved with :func:`resolve`.

Examples
--------
>>> from copul.regions.measures import resolve, label
>>> resolve("Spearman")
'rho'
>>> label("xi")
"Chatterjee's $\\xi$"
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "MEASURES",
    "MeasureInfo",
    "MeasureLike",
    "evaluate",
    "evaluate_many",
    "label",
    "resolve",
    "symbol",
]


@dataclass(frozen=True)
class MeasureInfo:
    """Static metadata of a dependence measure.

    Attributes
    ----------
    key : str
        Canonical key.
    method : str
        Name of the method on copul copula objects that evaluates the measure.
    name : str
        Human-readable name.
    symbol : str
        LaTeX symbol (without dollar signs).
    range : tuple of float
        Range of the measure over all bivariate copulas.
    value_M, value_W, value_Pi : float
        Values at the Fréchet--Hoeffding bounds :math:`M`, :math:`W` and at
        independence :math:`\\Pi`.
    """

    key: str
    method: str
    name: str
    symbol: str
    range: tuple[float, float]
    value_M: float
    value_W: float
    value_Pi: float = 0.0


MEASURES: dict[str, MeasureInfo] = {
    "xi": MeasureInfo("xi", "chatterjees_xi", "Chatterjee's", r"\xi", (0.0, 1.0), 1.0, 1.0),
    "rho": MeasureInfo("rho", "spearmans_rho", "Spearman's", r"\rho", (-1.0, 1.0), 1.0, -1.0),
    "tau": MeasureInfo("tau", "kendalls_tau", "Kendall's", r"\tau", (-1.0, 1.0), 1.0, -1.0),
    "footrule": MeasureInfo(
        "footrule",
        "spearmans_footrule",
        "Spearman's footrule",
        r"\psi",
        (-0.5, 1.0),
        1.0,
        -0.5,
    ),
    "gamma": MeasureInfo("gamma", "ginis_gamma", "Gini's", r"\gamma", (-1.0, 1.0), 1.0, -1.0),
    "beta": MeasureInfo("beta", "blomqvists_beta", "Blomqvist's", r"\beta", (-1.0, 1.0), 1.0, -1.0),
    "nu": MeasureInfo("nu", "blests_nu", "Blest's", r"\nu", (-1.0, 1.0), 1.0, -1.0),
}

_ALIASES: dict[str, str] = {
    "xi": "xi",
    "chatterjee": "xi",
    "chatterjees_xi": "xi",
    "rho": "rho",
    "spearman": "rho",
    "spearmans_rho": "rho",
    "tau": "tau",
    "kendall": "tau",
    "kendalls_tau": "tau",
    "footrule": "footrule",
    "psi": "footrule",
    "phi": "footrule",
    "spearmans_footrule": "footrule",
    "gamma": "gamma",
    "gini": "gamma",
    "ginis_gamma": "gamma",
    "beta": "beta",
    "blomqvist": "beta",
    "blomqvists_beta": "beta",
    "nu": "nu",
    "blest": "nu",
    "blests_nu": "nu",
}

MeasureLike = str | Callable[[Any], float]


def resolve(key: str) -> str:
    """Return the canonical key for ``key`` (case-insensitive, aliases allowed).

    Parameters
    ----------
    key : str
        A canonical key or alias, e.g. ``"Spearman"`` or ``"psi"``.

    Returns
    -------
    str
        One of ``"xi", "rho", "tau", "footrule", "gamma", "beta", "nu"``.

    Raises
    ------
    KeyError
        If ``key`` is unknown.
    """
    if not isinstance(key, str):
        raise TypeError(f"measure key must be a string, got {type(key).__name__}")
    k = key.strip().lower().replace("'", "").replace("-", "_").replace(" ", "_")
    if k not in _ALIASES:
        raise KeyError(f"Unknown dependence measure {key!r}. Known keys: {sorted(MEASURES)}")
    return _ALIASES[k]


def symbol(key: str) -> str:
    """LaTeX symbol of a measure, e.g. ``r"\\xi"``."""
    return MEASURES[resolve(key)].symbol


def label(key: str) -> str:
    """Axis label such as ``"Chatterjee's $\\xi$"``."""
    info = MEASURES[resolve(key)]
    return f"{info.name} ${info.symbol}$"


def evaluate(copula: Any, measure: MeasureLike) -> float:
    """Evaluate a dependence measure on a copul copula object.

    Parameters
    ----------
    copula : object
        Any copula exposing the corresponding method (e.g. ``chatterjees_xi``).
    measure : str or callable
        Measure key/alias, or a callable ``copula -> float``.

    Returns
    -------
    float
        The value, converted to a Python float (SymPy results are evaluated).
    """
    if callable(measure):
        return float(measure(copula))
    info = MEASURES[resolve(measure)]
    val = getattr(copula, info.method)()
    try:
        return float(val)
    except TypeError:  # SymPy objects that need evaluation
        return float(val.evalf())


def evaluate_many(copula: Any, measures) -> dict[str, float]:
    """Evaluate several measures; returns ``{canonical_key: value}``.

    Measures that raise or return NaN are reported as ``nan``.
    """
    out: dict[str, float] = {}
    for m in measures:
        k = resolve(m) if isinstance(m, str) else getattr(m, "__name__", repr(m))
        try:
            out[k] = evaluate(copula, m)
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            out[k] = float(np.nan)
    return out
