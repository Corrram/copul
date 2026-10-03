r"""
Registry of bivariate dependence measures.

Every measure known to :mod:`copul` is described by a :class:`Measure` record
with a *canonical key* (e.g. ``"rho"``), a set of aliases (e.g.
``"spearman"``, ``"spearmans_rho"``), the name of the method implementing it
on copula objects, its LaTeX symbol, the defining formula, its range and its
values at the Fréchet–Hoeffding bounds :math:`M`, :math:`W` and at the
independence copula :math:`\Pi`, and which numerical ingredients it needs
(``"cdf"``, ``"h1"`` (:math:`\partial_1 C`), ``"h2"`` (:math:`\partial_2 C`),
``"pdf"``).

Examples
--------
>>> from copul.measures import get_measure
>>> get_measure("spearman").key
'rho'
>>> get_measure("lambda_L").method_name
'lambda_L'
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_MEASURES",
    "MEASURES",
    "Measure",
    "get_measure",
    "list_measures",
    "resolve_key",
]

_INF = float("inf")


@dataclass(frozen=True)
class Measure:
    """Description of a bivariate dependence measure.

    Attributes
    ----------
    key : str
        Canonical key (``"rho"``, ``"tau"``, ``"xi"``, ...).
    name : str
        Human readable name.
    method_name : str
        Name of the method implementing the measure on copula objects.
    aliases : tuple of str
        Alternative (case-insensitive) keys.
    symbol : str
        LaTeX symbol.
    formula : str
        Defining formula in terms of the copula (LaTeX).
    range : tuple of float
        Attainable range over all bivariate copulas.
    at_M, at_W, at_Pi : float
        Values at the upper Fréchet bound, the lower Fréchet bound and the
        independence copula.
    needs : tuple of str
        Numerical ingredients required by the numeric engine
        (subset of ``{"cdf", "h1", "h2", "pdf"}``).
    method_kwargs : dict
        Fixed keyword arguments passed to ``method_name`` (e.g.
        ``{"condition_on_y": True}`` for ``"xi_2"``).
    options : tuple of str
        Names of measure-specific options (e.g. ``("p",)`` for ``"lp"``).
    """

    key: str
    name: str
    method_name: str
    aliases: tuple[str, ...] = ()
    symbol: str = ""
    formula: str = ""
    range: tuple[float, float] = (-1.0, 1.0)
    at_M: float = 1.0
    at_W: float = -1.0
    at_Pi: float = 0.0
    needs: tuple[str, ...] = ("cdf",)
    method_kwargs: dict[str, object] = field(default_factory=dict)
    options: tuple[str, ...] = ()
    doc: str = ""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Measure({self.key!r}, method={self.method_name!r})"


_MEASURE_LIST: list[Measure] = [
    Measure(
        key="xi",
        name="Chatterjee's xi",
        method_name="chatterjees_xi",
        aliases=("chatterjee", "chatterjees_xi", "chatterjee_xi", "xi_1"),
        symbol=r"\xi",
        formula=r"\xi(C) = 6\int_0^1\int_0^1 (\partial_1 C(u,v))^2\,du\,dv - 2",
        range=(0.0, 1.0),
        at_M=1.0,
        at_W=1.0,
        at_Pi=0.0,
        needs=("h1",),
        options=("condition_on_y",),
        doc="Chatterjee's rank correlation, Y regressed on X (conditioning on the first variable).",
    ),
    Measure(
        key="xi_2",
        name="Chatterjee's xi (conditioning on the second variable)",
        method_name="chatterjees_xi",
        aliases=("xi2", "xi_y", "chatterjees_xi_2"),
        symbol=r"\xi_2",
        formula=r"\xi_2(C) = 6\int_0^1\int_0^1 (\partial_2 C(u,v))^2\,du\,dv - 2",
        range=(0.0, 1.0),
        at_M=1.0,
        at_W=1.0,
        at_Pi=0.0,
        needs=("h2",),
        method_kwargs={"condition_on_y": True},
        doc="Chatterjee's xi of the transposed copula, i.e. X regressed on Y.",
    ),
    Measure(
        key="rho",
        name="Spearman's rho",
        method_name="spearmans_rho",
        aliases=("spearman", "spearmans_rho", "spearman_rho", "rho_s"),
        symbol=r"\rho_S",
        formula=r"\rho(C) = 12\int_0^1\int_0^1 C(u,v)\,du\,dv - 3",
        needs=("cdf",),
    ),
    Measure(
        key="tau",
        name="Kendall's tau",
        method_name="kendalls_tau",
        aliases=("kendall", "kendalls_tau", "kendall_tau"),
        symbol=r"\tau",
        formula=r"\tau(C) = 1 - 4\int_0^1\int_0^1 \partial_1 C\,\partial_2 C\,du\,dv",
        needs=("h1", "h2"),
        doc="The h-function representation is valid for copulas with singular components as well.",
    ),
    Measure(
        key="footrule",
        name="Spearman's footrule",
        method_name="spearmans_footrule",
        aliases=(
            "spearmans_footrule",
            "spearman_footrule",
            "psi",
            "phi_footrule",
        ),
        symbol=r"\psi",
        formula=r"\psi(C) = 6\int_0^1 C(t,t)\,dt - 2",
        range=(-0.5, 1.0),
        at_W=-0.5,
        needs=("cdf",),
    ),
    Measure(
        key="gamma",
        name="Gini's gamma",
        method_name="ginis_gamma",
        aliases=("gini", "ginis_gamma", "gini_gamma"),
        symbol=r"\gamma",
        formula=r"\gamma(C) = 4\int_0^1 \bigl[C(t,t) + C(t,1-t)\bigr]\,dt - 2",
        needs=("cdf",),
    ),
    Measure(
        key="beta",
        name="Blomqvist's beta",
        method_name="blomqvists_beta",
        aliases=("blomqvist", "blomqvists_beta", "blomqvist_beta"),
        symbol=r"\beta",
        formula=r"\beta(C) = 4\,C(\tfrac12,\tfrac12) - 1",
        needs=("cdf",),
    ),
    Measure(
        key="nu",
        name="Blest's nu",
        method_name="blests_nu",
        aliases=("blest", "blests_nu", "blest_nu"),
        symbol=r"\nu",
        formula=r"\nu(C) = 24\int_0^1\int_0^1 (1-u)\,C(u,v)\,du\,dv - 2",
        needs=("cdf",),
        doc="Not symmetric in (u, v); nu(C^T) generally differs from nu(C).",
    ),
    Measure(
        key="hoeffdings_d",
        name="Hoeffding's Phi^2 (dependence index)",
        method_name="hoeffdings_d",
        aliases=("phi2", "phi_2", "hoeffding", "hoeffdings_phi_square", "d"),
        symbol=r"\Phi^2",
        formula=r"\Phi^2(C) = 90\int_0^1\int_0^1 (C(u,v)-uv)^2\,du\,dv",
        range=(0.0, 1.0),
        at_W=1.0,
        needs=("cdf",),
    ),
    Measure(
        key="sigma",
        name="Schweizer-Wolff sigma",
        method_name="schweizer_wolff_sigma",
        aliases=("schweizer_wolff", "schweizer_wolff_sigma", "sw_sigma"),
        symbol=r"\sigma",
        formula=r"\sigma(C) = 12\int_0^1\int_0^1 |C(u,v)-uv|\,du\,dv",
        range=(0.0, 1.0),
        at_W=1.0,
        needs=("cdf",),
    ),
    Measure(
        key="kappa",
        name="Uniform distance to independence",
        method_name="uniform_distance",
        aliases=("uniform_distance", "linf", "l_inf", "sup_distance"),
        symbol=r"\kappa",
        formula=r"\kappa(C) = 4\sup_{(u,v)\in[0,1]^2}|C(u,v)-uv|",
        range=(0.0, 1.0),
        at_W=1.0,
        needs=("cdf",),
    ),
    Measure(
        key="lp",
        name="L^p distance to independence",
        method_name="lp_distance",
        aliases=("lp_distance", "lp_concordance", "l_p"),
        symbol=r"\delta_p",
        formula=r"\delta_p(C) = k(p)\int_0^1\int_0^1 |C(u,v)-uv|^p\,du\,dv,"
        r"\quad k(p) = \frac{p+1}{2\,B(p+1,p+2)}",
        range=(0.0, 1.0),
        at_W=1.0,
        needs=("cdf",),
        options=("p",),
        doc="k(p) is chosen such that delta_p(M) = delta_p(W) = 1; p=1 gives "
        "Schweizer-Wolff sigma, p=2 Hoeffding's Phi^2.",
    ),
    Measure(
        key="bkr",
        name="Blum-Kiefer-Rosenblatt coefficient",
        method_name="blum_kiefer_rosenblatt",
        aliases=("blum_kiefer_rosenblatt", "blum_kiefer_rosenblatt_b"),
        symbol=r"B",
        formula=r"B(C) = 30\int_{[0,1]^2}(C(u,v)-uv)^2\,dC(u,v)"
        r" = -60\int_0^1\int_0^1 \partial_1C\,(C-uv)\,(\partial_2C-u)\,du\,dv",
        range=(0.0, 1.0),
        at_W=1.0,
        needs=("cdf", "h1", "h2"),
        doc="The h-function form follows from integrating by parts in v and "
        "is valid for copulas with singular components.",
    ),
    Measure(
        key="mutual_information",
        name="Mutual information",
        method_name="mutual_information",
        aliases=("mi", "mutual_information", "copula_entropy"),
        symbol=r"I",
        formula=r"I(C) = \int_0^1\int_0^1 c(u,v)\,\log c(u,v)\,du\,dv",
        range=(0.0, _INF),
        at_M=_INF,
        at_W=_INF,
        needs=("pdf",),
        doc="Equals minus the copula entropy; infinite for copulas with a singular component.",
    ),
    Measure(
        key="zeta1",
        name="Trutschnig's zeta_1 (D_1 distance to independence)",
        method_name="trutschnig_zeta",
        aliases=("zeta_1", "zeta", "trutschnig_zeta", "trutschnig", "d1_dependence"),
        symbol=r"\zeta_1",
        formula=r"\zeta_1(C) = 3\,D_1(C,\Pi) = 3\int_0^1\int_0^1 |\partial_1 C(u,v) - v|\,du\,dv",
        range=(0.0, 1.0),
        at_M=1.0,
        at_W=1.0,
        at_Pi=0.0,
        needs=("h1",),
        doc="Trutschnig (2011): zeta_1(C) = 0 iff C = Pi and zeta_1(C) = 1 iff C is "
        "completely dependent (V a measurable function of U); not symmetric in (u, v).",
    ),
    Measure(
        key="lambda_l",
        name="Lower tail dependence coefficient",
        method_name="lambda_L",
        aliases=("lambda_L", "lower_tail", "tail_lower", "lambda_lower"),
        symbol=r"\lambda_L",
        formula=r"\lambda_L(C) = \lim_{t\to0^+} C(t,t)/t",
        range=(0.0, 1.0),
        at_W=0.0,
        needs=("cdf",),
        doc="Numerically estimated by extrapolating C(t,t)/t; closed forms should be preferred.",
    ),
    Measure(
        key="lambda_u",
        name="Upper tail dependence coefficient",
        method_name="lambda_U",
        aliases=("lambda_U", "upper_tail", "tail_upper", "lambda_upper"),
        symbol=r"\lambda_U",
        formula=r"\lambda_U(C) = \lim_{t\to1^-} (1-2t+C(t,t))/(1-t)",
        range=(0.0, 1.0),
        at_W=0.0,
        needs=("cdf",),
        doc="Numerically estimated by extrapolation; closed forms should be preferred.",
    ),
]

#: Mapping canonical key -> :class:`Measure`.
MEASURES: dict[str, Measure] = {m.key: m for m in _MEASURE_LIST}

#: Measures returned by ``copula.measures()`` when no keys are given.
DEFAULT_MEASURES: tuple[str, ...] = (
    "xi",
    "rho",
    "tau",
    "footrule",
    "gamma",
    "beta",
    "nu",
)


def _norm(key: str) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


_ALIAS_MAP: dict[str, str] = {}
for _m in _MEASURE_LIST:
    _ALIAS_MAP[_norm(_m.key)] = _m.key
    for _a in _m.aliases:
        _ALIAS_MAP.setdefault(_norm(_a), _m.key)
# method names resolve to the "default" measure of that method
_ALIAS_MAP["chatterjees_xi"] = "xi"


def resolve_key(key: str) -> str:
    """Return the canonical key for ``key`` (case-insensitive alias lookup).

    Raises
    ------
    KeyError
        If the key is unknown.
    """
    if isinstance(key, Measure):
        return key.key
    k = _ALIAS_MAP.get(_norm(key))
    if k is None:
        raise KeyError(f"Unknown dependence measure {key!r}. Known keys: {sorted(MEASURES)}")
    return k


def get_measure(key: str) -> Measure:
    """Return the :class:`Measure` registered under ``key`` or one of its aliases."""
    return MEASURES[resolve_key(key)]


def list_measures() -> list[str]:
    """Canonical keys of all registered measures."""
    return list(MEASURES)


def method_to_keys(method_name: str) -> list[str]:
    """Canonical keys implemented by a given copula method name."""
    return [m.key for m in _MEASURE_LIST if m.method_name == method_name]


def _iter_keys(keys: Iterable[str] | None) -> list[str]:
    if keys is None:
        return list(DEFAULT_MEASURES)
    if isinstance(keys, str):
        return [resolve_key(keys)]
    return [resolve_key(k) for k in keys]
