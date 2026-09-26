"""
Lazy module proxies for heavy optional-at-import-time dependencies.

``import copul`` should not pay for ``matplotlib.pyplot`` or ``pandas`` unless
plotting or data-frame functionality is actually used.  The proxies below
import the real module on first attribute access and forward everything to it,
so ``plt.figure(...)`` or ``pd.DataFrame(...)`` behave exactly as with a plain
``import``.  ``unittest.mock.patch("<module>.plt.show")`` keeps working.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any


class LazyModule:
    """Proxy that imports ``name`` on first attribute access."""

    def __init__(self, name: str, hint: str | None = None) -> None:
        object.__setattr__(self, "_lazy_name", name)
        object.__setattr__(self, "_lazy_hint", hint)

    def _load(self) -> ModuleType:
        try:
            return importlib.import_module(self._lazy_name)
        except ImportError as exc:  # pragma: no cover - depends on environment
            hint = self._lazy_hint or f"pip install {self._lazy_name.split('.')[0]}"
            raise ImportError(
                f"This functionality requires the optional dependency "
                f"'{self._lazy_name}'. Install it with `{hint}`."
            ) from exc

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)

    def __dir__(self) -> list[str]:
        return dir(self._load())

    def __repr__(self) -> str:
        loaded = self._lazy_name in sys.modules
        return f"<lazy module {self._lazy_name!r} ({'loaded' if loaded else 'not loaded'})>"


def is_pandas_instance(obj: Any, *names: str) -> bool:
    """``isinstance(obj, pandas.<name>)`` without importing pandas.

    If pandas has not been imported yet, ``obj`` cannot be a pandas object.
    """
    pd = sys.modules.get("pandas")
    if pd is None:
        return False
    return isinstance(obj, tuple(getattr(pd, n) for n in names))


plt = LazyModule("matplotlib.pyplot", "pip install matplotlib")
mcolors = LazyModule("matplotlib.colors", "pip install matplotlib")
pd = LazyModule("pandas", "pip install pandas")

__all__ = ["LazyModule", "is_pandas_instance", "mcolors", "pd", "plt"]
