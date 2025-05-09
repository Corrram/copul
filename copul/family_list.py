"""
Copula Families Module
======================

This module provides a comprehensive enumeration of all copula families available in the package,
organized by their mathematical categories.

Categories
----------
- Archimedean: Copulas derived from a generator function (e.g., Clayton, Gumbel, Frank)
- Elliptical: Copulas derived from elliptical distributions (e.g., Gaussian, Student's t)
- Extreme Value: Copulas suitable for modeling extreme events (e.g., Galambos, Hüsler-Reiss)
- Other: Copulas that don't fit into the above categories (e.g., FGM, Plackett)

Usage
-----
>>> import copul as cp
>>> # Using the enum
>>> clayton_copula = cp.Families.CLAYTON.value()
>>> # Direct instantiation
>>> clayton_copula = cp.Clayton()
>>> # Get all available families
>>> all_families = cp.Families.list_all()
>>> # Get only Archimedean families
>>> archimedean_families = cp.Families.list_by_category('archimedean')
"""

import enum
import importlib
from typing import Dict, List, Union
import inspect
import numpy as np


class FamilyCategory(enum.Enum):
    """
    Enum for categorizing copula families by their mathematical properties.
    """

    ARCHIMEDEAN = "archimedean"
    ELLIPTICAL = "elliptical"
    EXTREME_VALUE = "extreme_value"
    OTHER = "other"


class Families(enum.Enum):
    """
    Comprehensive enumeration of all copula families available in the package.

    Each enum value stores the fully qualified import path of the copula class.
    The class is imported lazily when the `cls` property is accessed.

    Examples
    --------
    >>> import copul as cp
    >>> # Instantiate a Clayton copula using the enum
    >>> clayton = cp.Families.CLAYTON.cls()
    >>> # Get all available families
    >>> all_families = [f.name for f in cp.Families]
    """

    # Archimedean Copulas
    CLAYTON = "copul.families.archimedean.Clayton"
    NELSEN1 = "copul.families.archimedean.Nelsen1"
    NELSEN2 = "copul.families.archimedean.Nelsen2"
    NELSEN3 = "copul.families.archimedean.Nelsen3"
    ALI_MIKHAIL_HAQ = "copul.families.archimedean.AliMikhailHaq"
    NELSEN4 = "copul.families.archimedean.Nelsen4"
    GUMBEL_HOUGAARD = "copul.families.archimedean.GumbelHougaard"
    NELSEN5 = "copul.families.archimedean.Nelsen5"
    FRANK = "copul.families.archimedean.Frank"
    NELSEN6 = "copul.families.archimedean.Nelsen6"
    JOE = "copul.families.archimedean.Joe"
    NELSEN7 = "copul.families.archimedean.Nelsen7"
    NELSEN8 = "copul.families.archimedean.Nelsen8"
    NELSEN9 = "copul.families.archimedean.Nelsen9"
    GUMBEL_BARNETT = "copul.families.archimedean.GumbelBarnett"
    NELSEN10 = "copul.families.archimedean.Nelsen10"
    NELSEN11 = "copul.families.archimedean.Nelsen11"
    NELSEN12 = "copul.families.archimedean.Nelsen12"
    NELSEN13 = "copul.families.archimedean.Nelsen13"
    NELSEN14 = "copul.families.archimedean.Nelsen14"
    NELSEN15 = "copul.families.archimedean.Nelsen15"
    GENEST_GHOUDI = "copul.families.archimedean.GenestGhoudi"
    NELSEN16 = "copul.families.archimedean.Nelsen16"
    NELSEN17 = "copul.families.archimedean.Nelsen17"
    NELSEN18 = "copul.families.archimedean.Nelsen18"
    NELSEN19 = "copul.families.archimedean.Nelsen19"
    NELSEN20 = "copul.families.archimedean.Nelsen20"
    NELSEN21 = "copul.families.archimedean.Nelsen21"
    NELSEN22 = "copul.families.archimedean.Nelsen22"

    # Extreme Value Copulas
    JOE_EV = "copul.families.extreme_value.JoeEV"
    BB5 = "copul.families.extreme_value.BB5"
    CUADRAS_AUGE = "copul.families.extreme_value.CuadrasAuge"
    GALAMBOS = "copul.families.extreme_value.Galambos"
    GUMBEL_HOUGAARD_EV = "copul.families.extreme_value.GumbelHougaardEV"
    HUESLER_REISS = "copul.families.extreme_value.HueslerReiss"
    TAWN = "copul.families.extreme_value.Tawn"
    T_EV = "copul.families.extreme_value.tEV"
    MARSHALL_OLKIN = "copul.families.extreme_value.MarshallOlkin"

    # Elliptical Copulas
    GAUSSIAN = "copul.families.elliptical.Gaussian"
    T = "copul.families.elliptical.StudentT"

    # Other Copulas
    BIV_CHECK_PI = "copul.checkerboard.biv_check_pi.BivCheckPi"
    BIV_CHECK_MIN = "copul.checkerboard.check_min.CheckMin"
    CHECK_PI = "copul.checkerboard.check_pi.CheckPi"
    CHECK_MIN = "copul.checkerboard.check_min.CheckMin"
    BIV_CHECK_W = "copul.checkerboard.biv_check_w.BivCheckW"
    FARLIE_GUMBEL_MORGENSTERN = "copul.families.other.FarlieGumbelMorgenstern"
    FRECHET = "copul.families.other.Frechet"
    INDEPENDENCE = "copul.families.other.IndependenceCopula"
    LOWER_FRECHET = "copul.families.other.LowerFrechet"
    MARDIA = "copul.families.other.Mardia"
    PLACKETT = "copul.families.other.Plackett"
    RAFTERY = "copul.families.other.Raftery"
    UPPER_FRECHET = "copul.families.other.UpperFrechet"

    @property
    def cls(self):
        """
        Lazily import and return the copula class associated with this enum member.
        """
        module_path, class_name = self.value.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_category(cls, family) -> FamilyCategory:
        """
        Get the category of a copula family.

        Parameters
        ----------
        family : Families or str
            The family enum value or name to categorize

        Returns
        -------
        FamilyCategory
            The category of the copula family

        Examples
        --------
        >>> Families.get_category(Families.CLAYTON)
        FamilyCategory.ARCHIMEDEAN
        >>> Families.get_category("CLAYTON")
        FamilyCategory.ARCHIMEDEAN
        """
        if isinstance(family, str):
            family = cls[family]
        # Determine category based on the module path
        module_path = family.value
        if "archimedean" in module_path:
            return FamilyCategory.ARCHIMEDEAN
        elif "elliptical" in module_path:
            return FamilyCategory.ELLIPTICAL
        elif "extreme_value" in module_path:
            return FamilyCategory.EXTREME_VALUE
        else:
            return FamilyCategory.OTHER

    @classmethod
    def list_all(cls) -> List[str]:
        """
        Get a list of all available copula family names.
        """
        return [f.name for f in cls]

    @classmethod
    def list_by_category(cls, category: Union[FamilyCategory, str]) -> List[str]:
        """
        Get a list of copula family names by category.
        """
        if isinstance(category, str):
            category = FamilyCategory(category.lower())
        return [f.name for f in cls if cls.get_category(f) == category]

    @classmethod
    def create(cls, family_name: str, *args, **kwargs):
        """
        Create a copula instance by family name with parameters.
        """
        family = cls[family_name]
        return family.cls(*args, **kwargs)

    @classmethod
    def get_params_info(cls, family_name: str) -> Dict:
        """
        Get information about the parameters of a copula family.
        """
        family_class = cls[family_name].cls
        result = {}
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

        # Extract parameter documentation if available
        doc = family_class.__init__.__doc__
        if doc:
            for param_name in result:
                param_pattern = f":param {param_name}:"
                if param_pattern in doc:
                    param_doc = doc.split(param_pattern)[1].split("\n")[0].strip()
                    result[param_name]["doc"] = param_doc
        return result

    @classmethod
    def compare_copulas(
        cls,
        u: np.ndarray,
        families: List[str] = None,
        fit_method: str = "ml",
        criteria: str = "aic",
    ) -> Dict:
        """
        Compare multiple copula families on the same dataset.
        """
        if families is None:
            families = ["CLAYTON", "GAUSSIAN", "FRANK", "GUMBEL_HOUGAARD", "T", "JOE"]

        results = []
        for family_name in families:
            try:
                copula = cls.create(family_name)
                if hasattr(copula, "fit"):
                    copula.fit(u, method=fit_method)
                if criteria == "aic":
                    score = copula.aic(u) if hasattr(copula, "aic") else float("inf")
                elif criteria == "bic":
                    score = copula.bic(u) if hasattr(copula, "bic") else float("inf")
                else:
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


# Legacy support for the `families` list
families = [f.cls.__name__ for f in Families]

# Add some useful constants
COMMON_COPULAS = ["CLAYTON", "FRANK", "GUMBEL_HOUGAARD", "GAUSSIAN", "T", "JOE"]
ARCHIMEDEAN_COPULAS = Families.list_by_category(FamilyCategory.ARCHIMEDEAN)
ELLIPTICAL_COPULAS = Families.list_by_category(FamilyCategory.ELLIPTICAL)
EXTREME_VALUE_COPULAS = Families.list_by_category(FamilyCategory.EXTREME_VALUE)
OTHER_COPULAS = Families.list_by_category(FamilyCategory.OTHER)
