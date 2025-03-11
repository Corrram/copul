"""
Copula Families Module
=====================

This module provides a comprehensive enumeration of all copula families available in the package,
organized by their mathematical categories.

Categories
---------
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
from typing import Dict, List, Union
import inspect
import numpy as np

from copul.families import archimedean, elliptical, extreme_value, other


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

    Each enum value represents a copula family and provides access to the corresponding class.
    Families are organized into categories based on their mathematical properties.

    Examples
    --------
    >>> import copul as cp
    >>> # Instantiate a Clayton copula using the enum
    >>> clayton = cp.Families.CLAYTON.value()
    >>> # Directly instantiate a Clayton copula
    >>> clayton = cp.Clayton()
    >>> # Generate sample data
    >>> import numpy as np
    >>> u = np.array([[0.5, 0.7], [0.3, 0.9]])
    >>> # Calculate CDF
    >>> clayton.cdf(u)

    See Also
    --------
    FamilyCategory : Enum for categorizing copula families
    """

    # Archimedean Copulas
    CLAYTON = archimedean.Clayton
    NELSEN1 = archimedean.Nelsen1
    NELSEN2 = archimedean.Nelsen2
    NELSEN3 = archimedean.Nelsen3
    ALI_MIKHAIL_HAQ = archimedean.AliMikhailHaq
    NELSEN4 = archimedean.Nelsen4
    GUMBEL_HOUGAARD = archimedean.GumbelHougaard
    NELSEN5 = archimedean.Nelsen5
    FRANK = archimedean.Frank
    NELSEN6 = archimedean.Nelsen6
    JOE = archimedean.Joe
    NELSEN7 = archimedean.Nelsen7
    NELSEN8 = archimedean.Nelsen8
    NELSEN9 = archimedean.Nelsen9
    GUMBEL_BARNETT = archimedean.GumbelBarnett
    NELSEN10 = archimedean.Nelsen10
    NELSEN11 = archimedean.Nelsen11
    NELSEN12 = archimedean.Nelsen12
    NELSEN13 = archimedean.Nelsen13
    NELSEN14 = archimedean.Nelsen14
    NELSEN15 = archimedean.Nelsen15
    GENEST_GHOUDI = archimedean.GenestGhoudi
    NELSEN16 = archimedean.Nelsen16
    NELSEN17 = archimedean.Nelsen17
    NELSEN18 = archimedean.Nelsen18
    NELSEN19 = archimedean.Nelsen19
    NELSEN20 = archimedean.Nelsen20
    NELSEN21 = archimedean.Nelsen21
    NELSEN22 = archimedean.Nelsen22

    # Extreme Value Copulas
    JOE_EV = extreme_value.JoeEV
    BB5 = extreme_value.BB5
    CUADRAS_AUGE = extreme_value.CuadrasAuge
    GALAMBOS = extreme_value.Galambos
    GUMBEL_HOUGAARD_EV = extreme_value.GumbelHougaard
    HUESSLER_REISS = extreme_value.HueslerReiss
    TAWN = extreme_value.Tawn
    T_EV = extreme_value.tEV
    MARSHALL_OLKIN = extreme_value.MarshallOlkin

    # Elliptical Copulas
    GAUSSIAN = elliptical.Gaussian
    T = elliptical.StudentT
    # LAPLACE = elliptical.Laplace  # Currently commented out

    # Other Copulas
    # B11 = other.B11  # Currently commented out
    CHECKERBOARD = other.BivCheckPi
    FARLIE_GUMBEL_MORGENSTERN = other.FarlieGumbelMorgenstern
    FRECHET = other.Frechet
    INDEPENDENCE = other.IndependenceCopula
    LOWER_FRECHET = other.LowerFrechet
    MARDIA = other.Mardia
    PLACKETT = other.Plackett
    RAFTERY = other.Raftery
    UPPER_FRECHET = other.UpperFrechet

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

        module_path = family.value.__module__

        # Check which category the module belongs to based on its path
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

        Returns
        -------
        List[str]
            Names of all available copula families

        Examples
        --------
        >>> Families.list_all()
        ['CLAYTON', 'NELSEN1', 'NELSEN2', ...]
        """
        return [f.name for f in cls]

    @classmethod
    def list_by_category(cls, category: Union[FamilyCategory, str]) -> List[str]:
        """
        Get a list of copula family names by category.

        Parameters
        ----------
        category : FamilyCategory or str
            The category to filter by (e.g., 'archimedean', 'elliptical')

        Returns
        -------
        List[str]
            Names of copula families in the specified category

        Examples
        --------
        >>> Families.list_by_category('archimedean')
        ['CLAYTON', 'NELSEN1', 'NELSEN2', ...]
        >>> Families.list_by_category(FamilyCategory.ARCHIMEDEAN)
        ['CLAYTON', 'NELSEN1', 'NELSEN2', ...]
        """
        if isinstance(category, str):
            category = FamilyCategory(category.lower())

        return [f.name for f in cls if cls.get_category(f) == category]

    @classmethod
    def create(cls, family_name: str, *args, **kwargs):
        """
        Create a copula instance by family name with parameters.

        Parameters
        ----------
        family_name : str
            Name of the copula family to instantiate
        *args, **kwargs
            Arguments to pass to the copula constructor

        Returns
        -------
        Copula
            Instance of the requested copula family

        Examples
        --------
        >>> clayton = Families.create('CLAYTON', theta=2.0)
        >>> gaussian = Families.create('GAUSSIAN', corr=0.7)
        """
        family = cls[family_name]
        return family.value(*args, **kwargs)

    @classmethod
    def get_params_info(cls, family_name: str) -> Dict:
        """
        Get information about the parameters of a copula family.

        Parameters
        ----------
        family_name : str
            Name of the copula family

        Returns
        -------
        Dict
            Dictionary with parameter names, default values, and documentation

        Examples
        --------
        >>> Families.get_params_info('CLAYTON')
        {'theta': {'default': 1.0, 'doc': 'Dependency parameter...'}}
        """
        family_class = cls[family_name].value
        result = {}

        # Get the signature of the __init__ method
        signature = inspect.signature(family_class.__init__)

        # Extract parameter information
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            param_info = {
                "default": param.default
                if param.default is not inspect.Parameter.empty
                else None,
                "doc": "",  # Will be filled if docstring is available
                "required": param.default is inspect.Parameter.empty,
            }

            result[param_name] = param_info

        # Try to extract parameter documentation from docstring
        doc = family_class.__init__.__doc__
        if doc:
            # Simple docstring parsing - could be improved
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

        Parameters
        ----------
        u : np.ndarray
            Data to fit copulas on, shape (n_samples, n_dimensions)
        families : List[str], optional
            List of family names to compare, default is all families
        fit_method : str, optional
            Method used for fitting ('ml', 'mpl', etc.), default is 'ml'
        criteria : str, optional
            Criteria for comparing models ('aic', 'bic', 'likelihood'), default is 'aic'

        Returns
        -------
        Dict
            Dictionary with model comparison results sorted by selected criteria

        Examples
        --------
        >>> import numpy as np
        >>> u = np.random.rand(100, 2)  # Uniform data
        >>> results = Families.compare_copulas(u, ['CLAYTON', 'GAUSSIAN', 'FRANK'])
        >>> best_family = results[0]['family']  # Get best family
        """
        if families is None:
            # Use a subset of common families by default to avoid very slow computation
            families = ["CLAYTON", "GAUSSIAN", "FRANK", "GUMBEL_HOUGAARD", "T", "JOE"]

        results = []

        for family_name in families:
            try:
                # Create the copula
                copula = cls.create(family_name)

                # Fit the copula
                if hasattr(copula, "fit"):
                    copula.fit(u, method=fit_method)

                # Calculate criteria
                if criteria == "aic":
                    score = copula.aic(u) if hasattr(copula, "aic") else float("inf")
                elif criteria == "bic":
                    score = copula.bic(u) if hasattr(copula, "bic") else float("inf")
                else:  # Default to likelihood
                    score = (
                        -copula.log_likelihood(u)
                        if hasattr(copula, "log_likelihood")
                        else float("inf")
                    )

                # Store result
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
                # Skip models that fail to fit
                print(f"Failed to fit {family_name}: {str(e)}")
                continue

        # Sort by score (lower is better for AIC/BIC, higher for likelihood)
        reverse = criteria.lower() == "likelihood"
        results.sort(key=lambda x: x["score"], reverse=reverse)

        return results


# Legacy support for the `families` list
families = [f.value.__name__ for f in Families]

# Add some useful constants
COMMON_COPULAS = ["CLAYTON", "FRANK", "GUMBEL_HOUGAARD", "GAUSSIAN", "T", "JOE"]
ARCHIMEDEAN_COPULAS = Families.list_by_category(FamilyCategory.ARCHIMEDEAN)
ELLIPTICAL_COPULAS = Families.list_by_category(FamilyCategory.ELLIPTICAL)
EXTREME_VALUE_COPULAS = Families.list_by_category(FamilyCategory.EXTREME_VALUE)
OTHER_COPULAS = Families.list_by_category(FamilyCategory.OTHER)
