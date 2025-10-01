"""
Copula Families Module
======================

Provides a comprehensive enumeration of all copulas in the package, organized by:

1) Mathematical Category
   - Archimedean, Elliptical, Extreme Value, Other

2) Semantic Kind
   - FAMILY         : true parametric families for modeling/fitting
   - APPROXIMATION  : approximation constructs (checkerboards, Bernstein, ...)
   - SPECIAL        : notable/special single copulas (Fréchet bounds, Independence, ...)

Module-level public lists (alphabetical class names)
----------------------------------------------------
- cp.families         -> class names of true parametric families
- cp.approximations   -> class names of approximation constructs
- cp.copulas          -> class names of notable/special single copulas

Examples
--------
>>> import copul as cp
>>> cp.families
['AliMikhailHaq', 'Clayton', 'Frank', 'Gaussian', 'GenestGhoudi', ...]
>>> cp.approximations
['BivCheckPi', 'BivCheckW', 'CheckMin', 'CheckPi', ...]
>>> cp.copulas
['ClampedParabolaCopula', 'DiagonalBandCopula', 'Frechet', 'IndependenceCopula', ...]

Programmatic APIs
-----------------
>>> cp.Families.list_all()                       # UPPER_CASE enum names of true families
>>> cp.Families.list_all_classnames()            # class names of true families
>>> cp.Families.list_approx_classnames()         # class names of approximations
>>> cp.Families.list_special_classnames()        # class names of specials
>>> cp.Families.list_by_category('archimedean')  # filter by mathematical category
"""

from __future__ import annotations

import enum
import importlib
import inspect
from typing import Dict, List, Union, Optional
import numpy as np
import re


def _natural_key(s: str):
    # 'Nelsen22' -> ['nelsen', 22, '']; case-insensitive, numbers as ints
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", s)]


# ---------------------------------------------------------------------------
# Category and Kind enums
# ---------------------------------------------------------------------------


class FamilyCategory(enum.Enum):
    ARCHIMEDEAN = "archimedean"
    ELLIPTICAL = "elliptical"
    EXTREME_VALUE = "extreme_value"
    OTHER = "other"


class CopulaKind(enum.Enum):
    FAMILY = "family"  # true parametric families
    APPROXIMATION = "approximation"  # checkerboards, Bernstein, etc.
    SPECIAL = "special"  # Fréchet bounds, Independence, etc.


# ---------------------------------------------------------------------------
# Name sets for Kind classification (kept OUTSIDE the Enum!)
# ---------------------------------------------------------------------------

APPROXIMATION_NAMES = {
    "CHECK_PI",
    "BIV_CHECK_PI",
    "CHECK_MIN",
    "BIV_CHECK_MIN",
    "BIV_CHECK_W",
    # Add "BERNSTEIN" here when implemented, e.g.: "BERNSTEIN"
}

SPECIAL_NAMES = {
    "FRECHET",
    "INDEPENDENCE",
    "LOWER_FRECHET",
    "PI_OVER_SIGMA_MINUS_PI",
    "UPPER_FRECHET",
}


# ---------------------------------------------------------------------------
# Families Enum (registry)
# ---------------------------------------------------------------------------


class Families(enum.Enum):
    """Registry of all copulas with lazy import of their classes."""

    # -----------------------
    # Archimedean Copulas
    # -----------------------
    CLAYTON = "copul.family.archimedean.Clayton"
    NELSEN1 = "copul.family.archimedean.Nelsen1"
    NELSEN2 = "copul.family.archimedean.Nelsen2"
    NELSEN3 = "copul.family.archimedean.Nelsen3"
    ALI_MIKHAIL_HAQ = "copul.family.archimedean.AliMikhailHaq"
    NELSEN4 = "copul.family.archimedean.Nelsen4"
    GUMBEL_HOUGAARD = "copul.family.archimedean.GumbelHougaard"
    NELSEN5 = "copul.family.archimedean.Nelsen5"
    FRANK = "copul.family.archimedean.Frank"
    NELSEN6 = "copul.family.archimedean.Nelsen6"
    JOE = "copul.family.archimedean.Joe"
    NELSEN7 = "copul.family.archimedean.Nelsen7"
    NELSEN8 = "copul.family.archimedean.Nelsen8"
    NELSEN9 = "copul.family.archimedean.Nelsen9"
    GUMBEL_BARNETT = "copul.family.archimedean.GumbelBarnett"
    NELSEN10 = "copul.family.archimedean.Nelsen10"
    NELSEN11 = "copul.family.archimedean.Nelsen11"
    NELSEN12 = "copul.family.archimedean.Nelsen12"
    NELSEN13 = "copul.family.archimedean.Nelsen13"
    NELSEN14 = "copul.family.archimedean.Nelsen14"
    NELSEN15 = "copul.family.archimedean.Nelsen15"
    GENEST_GHOUDI = "copul.family.archimedean.GenestGhoudi"
    NELSEN16 = "copul.family.archimedean.Nelsen16"
    NELSEN17 = "copul.family.archimedean.Nelsen17"
    NELSEN18 = "copul.family.archimedean.Nelsen18"
    NELSEN19 = "copul.family.archimedean.Nelsen19"
    NELSEN20 = "copul.family.archimedean.Nelsen20"
    NELSEN21 = "copul.family.archimedean.Nelsen21"
    NELSEN22 = "copul.family.archimedean.Nelsen22"

    # -----------------------
    # Extreme Value Copulas
    # -----------------------
    JOE_EV = "copul.family.extreme_value.JoeEV"
    BB5 = "copul.family.extreme_value.BB5"
    CUADRAS_AUGE = "copul.family.extreme_value.CuadrasAuge"
    GALAMBOS = "copul.family.extreme_value.Galambos"
    GUMBEL_HOUGAARD_EV = "copul.family.extreme_value.GumbelHougaardEV"
    HUESLER_REISS = "copul.family.extreme_value.HueslerReiss"
    TAWN = "copul.family.extreme_value.Tawn"
    T_EV = "copul.family.extreme_value.tEV"
    MARSHALL_OLKIN = "copul.family.extreme_value.MarshallOlkin"

    # -----------------------
    # Elliptical Copulas
    # -----------------------
    GAUSSIAN = "copul.family.elliptical.Gaussian"
    T = "copul.family.elliptical.StudentT"

    # -----------------------
    # Other / Approximations / Specials
    # -----------------------
    BERNSTEIN = "copul.checkerboard.bernstein.Bernstein"  # when implemented
    BIV_CHECK_PI = "copul.checkerboard.biv_check_pi.BivCheckPi"
    BIV_CHECK_MIN = "copul.checkerboard.biv_check_min.BivCheckMin"
    BIV_CHECK_W = "copul.checkerboard.biv_check_w.BivCheckW"
    CHECK_PI = "copul.checkerboard.check_pi.CheckPi"
    CHECK_MIN = "copul.checkerboard.check_min.CheckMin"
    SHUFFLE_OF_MIN = "copul.checkerboard.shuffle_min.ShuffleOfMin"

    FARLIE_GUMBEL_MORGENSTERN = "copul.family.other.FarlieGumbelMorgenstern"
    FRECHET = "copul.family.other.Frechet"
    INDEPENDENCE = "copul.family.other.IndependenceCopula"
    LOWER_FRECHET = "copul.family.other.LowerFrechet"
    MARDIA = "copul.family.other.Mardia"
    PI_OVER_SIGMA_MINUS_PI = (
        "copul.family.other.pi_over_sigma_minus_pi.PiOverSigmaMinusPi"
    )
    PLACKETT = "copul.family.other.Plackett"
    RAFTERY = "copul.family.other.Raftery"
    UPPER_FRECHET = "copul.family.other.UpperFrechet"
    CLAMPED_PARABOLA = "copul.family.other.ClampedParabolaCopula"
    DIAGONAL_BAND = "copul.family.other.DiagonalBandCopula"
    XI_RHO_BOUNDARY = "copul.family.other.XiRhoBoundaryCopula"

    # -----------------------------------------------------------------------
    # Lazy class import
    # -----------------------------------------------------------------------
    @property
    def cls(self):
        module_path, class_name = self.value.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    # -----------------------------------------------------------------------
    # Kind & Category helpers
    # -----------------------------------------------------------------------
    @classmethod
    def get_kind(cls, member: "Families") -> CopulaKind:
        name = member.name
        module_path = member.value  # 'package.module.Class'
        # Heuristic: anything in a checkerboard/approx module is an approximation
        if (
            "checkerboard" in module_path
            or "approx" in module_path
            or "bernstein" in module_path
        ):
            return CopulaKind.APPROXIMATION
        # Explicit notable/special singletons
        if name in SPECIAL_NAMES:
            return CopulaKind.SPECIAL
        return CopulaKind.FAMILY

    @classmethod
    def get_category(cls, family: Union["Families", str]) -> FamilyCategory:
        """Mathematical category based on import path."""
        if isinstance(family, str):
            family = cls[family]
        module_path = family.value
        if "archimedean" in module_path:
            return FamilyCategory.ARCHIMEDEAN
        if "elliptical" in module_path:
            return FamilyCategory.ELLIPTICAL
        if "extreme_value" in module_path:
            return FamilyCategory.EXTREME_VALUE
        return FamilyCategory.OTHER

    # -----------------------------------------------------------------------
    # Listing APIs
    # -----------------------------------------------------------------------
    @classmethod
    def list_names(
        cls,
        kind: Union[CopulaKind, str, None] = None,
        category: Union[FamilyCategory, str, None] = None,
    ) -> List[str]:
        """
        Enum names filtered by semantic kind and/or mathematical category.
        kind     : {'family','approximation','special'} or None
        category : {'archimedean','elliptical','extreme_value','other'} or None
        """
        if isinstance(kind, str):
            kind = CopulaKind(kind.lower())
        if isinstance(category, str):
            category = FamilyCategory(category.lower())

        def _ok(m: "Families") -> bool:
            return (kind is None or cls.get_kind(m) == kind) and (
                category is None or cls.get_category(m) == category
            )

        return [m.name for m in cls if _ok(m)]

    @classmethod
    def list_classes(
        cls,
        kind: Union[CopulaKind, str, None] = None,
        category: Union[FamilyCategory, str, None] = None,
    ) -> List[type]:
        """Same as list_names but returns the class objects."""
        return [cls[name].cls for name in cls.list_names(kind=kind, category=category)]

    # Convenience shorthands (enum names)
    @classmethod
    def list_all(cls) -> List[str]:
        """UPPER_CASE enum names of true modeling families."""
        return cls.list_names(kind=CopulaKind.FAMILY)

    @classmethod
    def list_by_category(cls, category: Union[FamilyCategory, str]) -> List[str]:
        """UPPER_CASE enum names of true families within a given category."""
        return cls.list_names(kind=CopulaKind.FAMILY, category=category)

    @classmethod
    def list_approximations(cls) -> List[str]:
        """UPPER_CASE enum names of approximation constructs."""
        return cls.list_names(kind=CopulaKind.APPROXIMATION)

    @classmethod
    def list_specials(cls) -> List[str]:
        """UPPER_CASE enum names of notable/special copulas."""
        return cls.list_names(kind=CopulaKind.SPECIAL)

    # Convenience shorthands (class-name lists)
    @classmethod
    def _classnames_for(
        cls, enum_names: List[str], lowercase: bool = False
    ) -> List[str]:
        names = [cls[name].cls.__name__ for name in enum_names]
        names.sort()
        return [n.lower() for n in names] if lowercase else names

    @classmethod
    def list_all_classnames(cls, lowercase: bool = False) -> List[str]:
        """Class names of true families (sorted)."""
        return cls._classnames_for(cls.list_all(), lowercase=lowercase)

    @classmethod
    def list_approx_classnames(cls, lowercase: bool = False) -> List[str]:
        """Class names of approximation constructs (sorted)."""
        return cls._classnames_for(cls.list_approximations(), lowercase=lowercase)

    @classmethod
    def list_special_classnames(cls, lowercase: bool = False) -> List[str]:
        """Class names of notable/special copulas (sorted)."""
        return cls._classnames_for(cls.list_specials(), lowercase=lowercase)

    # -----------------------------------------------------------------------
    # Factory & Introspection
    # -----------------------------------------------------------------------
    @classmethod
    def create(cls, family_name: str, *args, **kwargs):
        """Instantiate a copula by enum NAME with parameters."""
        return cls[family_name].cls(*args, **kwargs)

    @classmethod
    def get_params_info(cls, family_name: str) -> Dict:
        """Inspect __init__ signature and basic param doc of a copula class."""
        family_class = cls[family_name].cls
        result: Dict[str, Dict[str, Optional[Union[str, bool]]]] = {}
        signature = inspect.signature(family_class.__init__)
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
            param_info = {
                "default": (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else None
                ),
                "doc": "",
                "required": param.default is inspect.Parameter.empty,
            }
            result[param_name] = param_info

        # naive docstring scrape (optional)
        doc = getattr(family_class.__init__, "__doc__", None)
        if doc:
            for param_name in result:
                token = f":param {param_name}:"
                if token in doc:
                    part = doc.split(token, 1)[1]
                    line = part.split("\n", 1)[0].strip()
                    result[param_name]["doc"] = line
        return result

    @classmethod
    def compare_copulas(
        cls,
        u: np.ndarray,
        families: Optional[List[str]] = None,
        fit_method: str = "ml",
        criteria: str = "aic",
    ) -> List[Dict]:
        """
        Compare multiple copula families on the same dataset.

        Parameters
        ----------
        u : ndarray
            Pseudo-observations in [0,1]^d.
        families : list[str] | None
            Enum NAMES to compare; defaults to a common set.
        fit_method : str
            Method for .fit(), e.g. 'ml' (if available).
        criteria : {'aic','bic','likelihood'}
            Sorting metric; lower is better for aic/bic, higher is better for likelihood.

        Returns
        -------
        list[dict]
            Sorted list of results with keys: 'family', 'copula', 'score', 'params'.
        """
        if families is None:
            families = ["CLAYTON", "GAUSSIAN", "FRANK", "GUMBEL_HOUGAARD", "T", "JOE"]

        results: List[Dict] = []
        for family_name in families:
            try:
                copula = cls.create(family_name)
                if hasattr(copula, "fit"):
                    copula.fit(u, method=fit_method)
                if criteria.lower() == "aic":
                    score = copula.aic(u) if hasattr(copula, "aic") else float("inf")
                elif criteria.lower() == "bic":
                    score = copula.bic(u) if hasattr(copula, "bic") else float("inf")
                else:  # likelihood
                    score = (
                        -copula.log_likelihood(u)
                        if hasattr(copula, "log_likelihood")
                        else float("inf")
                    )
                results.append(
                    {
                        "family": family_name,
                        "copula": copula,
                        "score": score,
                        "params": {
                            param: getattr(copula, param)
                            for param in cls.get_params_info(family_name)
                            if hasattr(copula, param)
                        },
                    }
                )
            except Exception as e:
                print(f"Failed to fit {family_name}: {str(e)}")
                continue

        reverse = criteria.lower() == "likelihood"
        results.sort(key=lambda x: x["score"], reverse=reverse)
        return results


# ---------------------------------------------------------------------------
# Module-level public lists (alphabetical class names)
# ---------------------------------------------------------------------------


def _classnames_for_kind(kind: CopulaKind) -> list[str]:
    # get enum names for this kind (FAMILY / APPROXIMATION / SPECIAL)
    enum_names = Families.list_names(kind=kind)
    # dedup by class object (handles multiple enum names pointing to same class)
    classes = {Families[name].cls for name in enum_names}
    # map to class names and natural-sort (Nelsen2 < Nelsen10 < Nelsen22)
    names = {cls.__name__ for cls in classes}
    return sorted(names, key=_natural_key)


# True parametric families only
families = _classnames_for_kind(CopulaKind.FAMILY)

# Approximation constructs
approximations = _classnames_for_kind(CopulaKind.APPROXIMATION)

# Notable/special one-off copulas
copulas = _classnames_for_kind(CopulaKind.SPECIAL)

# ---------------------------------------------------------------------------
# Optional convenience name groups (enum-name lists)
# ---------------------------------------------------------------------------

COMMON = ["CLAYTON", "FRANK", "GUMBEL_HOUGAARD", "GAUSSIAN", "T", "JOE"]
ARCHIMEDEAN = Families.list_by_category(FamilyCategory.ARCHIMEDEAN)
ELLIPTICAL = Families.list_by_category(FamilyCategory.ELLIPTICAL)
EXTREME_VALUE = Families.list_by_category(FamilyCategory.EXTREME_VALUE)
OTHER = Families.list_by_category(FamilyCategory.OTHER)

# Also expose approximation/special enum-name lists
APPROXIMATIONS = Families.list_approximations()
SPECIALS = Families.list_specials()


# ---------------------------------------------------------------------------
# Public API of this module
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "FamilyCategory",
    "CopulaKind",
    "Families",
    # Module-level lists (class names)
    "families",
    "approximations",
    "copulas",
    # Convenience groups (enum-name lists)
    "COMMON",
    "ARCHIMEDEAN",
    "ELLIPTICAL",
    "EXTREME_VALUE",
    "OTHER",
    "APPROXIMATIONS",
    "SPECIALS",
]
