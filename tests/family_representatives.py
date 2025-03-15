"""
Module defining copula family representatives with normalized access.
This dictionary is both case-insensitive and underscore-insensitive.
"""

from typing import Mapping


class NormalizedDict(dict):
    """
    A dictionary that provides normalized key access while preserving original keys.
    Normalization includes:
    - Case insensitivity (GumbelHougaard = gumbelhougaard)
    - Underscore insensitivity (GumbelHougaard = gumbel_hougaard)

    Examples:
    ---------
    >>> d = NormalizedDict({"GumbelHougaard": 2})
    >>> d["gumbelhougaard"]  # Returns 2
    >>> d["gumbel_hougaard"]  # Returns 2
    >>> d["GUMBEL_HOUGAARD"]  # Returns 2
    """

    def __init__(self, data=None, **kwargs):
        super().__init__()
        self._keys_map = {}  # Maps normalized keys to original keys

        if data is not None:
            self.update(data)
        if kwargs:
            self.update(kwargs)

    @staticmethod
    def _normalize_key(key):
        """
        Normalize a key by converting to lowercase and removing underscores.

        Parameters
        ----------
        key : Any
            The key to normalize

        Returns
        -------
        Any
            The normalized key if it's a string, otherwise the original key
        """
        if isinstance(key, str):
            return key.lower().replace("_", "")
        return key

    def __getitem__(self, key):
        normalized_key = self._normalize_key(key)
        original_key = self._keys_map.get(normalized_key, key)
        return super().__getitem__(original_key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._keys_map[self._normalize_key(key)] = key
        super().__setitem__(key, value)

    def __delitem__(self, key):
        normalized_key = self._normalize_key(key)
        original_key = self._keys_map.get(normalized_key, key)
        super().__delitem__(original_key)
        if normalized_key in self._keys_map:
            del self._keys_map[normalized_key]

    def __contains__(self, key):
        normalized_key = self._normalize_key(key)
        return normalized_key in self._keys_map

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, mapping=None, **kwargs):
        if mapping is not None:
            for key, value in mapping.items():
                self[key] = value
        if kwargs:
            for key, value in kwargs.items():
                self[key] = value

    def copy(self):
        return NormalizedDict(super().copy())

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    # Support for | operator (Python 3.9+)
    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        result = self.copy()
        result.update(other)
        return result


# Define the Archimedean representatives with normalized access
archimedean_representatives = NormalizedDict(
    {
        "Clayton": 0.5,
        "Nelsen2": 1.5,
        "AliMikhailHaq": 0.5,
        "GumbelHougaard": 2,
        "Frank": 0.5,
        "Joe": 1.5,
        "Nelsen7": 0.5,
        "Nelsen8": 2,
        "GumbelBarnett": 0.5,
        "Nelsen10": 0.5,
        "Nelsen11": 0.5,
        "Nelsen12": 2,
        "Nelsen13": 0.5,
        "Nelsen14": 2,
        "GenestGhoudi": 2,
        "Nelsen16": 0.5,
        "Nelsen17": 0.5,
        "Nelsen18": 2,
        "Nelsen19": 0.5,
        "Nelsen20": 0.5,
        "Nelsen21": 3,
        "Nelsen22": 0.5,
    }
)

# Define additional family representatives
additional_representatives = NormalizedDict(
    {
        "JoeEV": (0.5, 0.5, 2),
        "BB5": (2, 2),
        "CuadrasAuge": 0.5,
        "Galambos": 0.5,
        "GumbelHougaardEV": 3,
        "HueslerReiss": 2,
        "Tawn": (0.5, 0.5, 2),
        "tEV": (0.5, 2),
        "MarshallOlkin": (0.8, 0.5),
        "Gaussian": 0.5,
        "StudentT": (0.5, 2),
        # "Laplace": 0.5,
        # "B11": 0.5,
        "FarlieGumbelMorgenstern": 0.5,
        "Frechet": (0.5, 0.2),
        "Mardia": 0.5,
        "Plackett": 2,
        "Raftery": 0.5,
    }
)

# Combine all representatives
family_representatives = archimedean_representatives | additional_representatives
