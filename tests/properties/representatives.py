"""
Representative instances of every bivariate copula class for the universal
property tests.

Every member of :class:`copul.family_list.Families` is covered (with the
parameters of ``tests/family_representatives.py`` where available), plus
negative-dependence variants of the main families, the checkerboard,
Bernstein and shuffle constructions and the boundary families of
``copul.family.other``.
"""

from __future__ import annotations

from functools import cache

import numpy as np

import copul as cp
from copul.family_list import Families
from tests.family_representatives import family_representatives


def _doubly_stochastic(k, seed):
    rng = np.random.default_rng(seed)
    m = rng.random((k, k)) + 0.05
    for _ in range(200):  # Sinkhorn
        m /= m.sum(axis=1, keepdims=True)
        m /= m.sum(axis=0, keepdims=True)
    return m / m.sum()


def _from_family_reps(name):
    cls = Families[name].cls
    param = family_representatives.get(cls.__name__)
    if param is None:
        return cls()
    if isinstance(param, tuple):
        return cls(*param)
    return cls(param)


# explicit parameters for Families members without an entry in
# tests/family_representatives.py (or with a class name differing from it)
_EXPLICIT = {
    "CLAYTON": lambda: cp.Clayton(1.5),
    "NELSEN1": lambda: cp.Nelsen1(0.7),
    "BERNSTEIN": lambda: cp.BernsteinCopula(_doubly_stochastic(3, 1)),
    "BIV_CHECK_PI": lambda: cp.BivCheckPi(_doubly_stochastic(4, 2)),
    "BIV_CHECK_MIN": lambda: cp.BivCheckMin(_doubly_stochastic(3, 3)),
    "BIV_CHECK_W": lambda: cp.BivCheckW(_doubly_stochastic(3, 4)),
    "CHECK_PI": lambda: cp.CheckPi(_doubly_stochastic(3, 5)),
    "CHECK_MIN": lambda: cp.CheckMin(_doubly_stochastic(4, 6)),
    "SHUFFLE_OF_MIN": lambda: cp.ShuffleOfMin([2, 4, 1, 3]),
    "INDEPENDENCE": lambda: cp.IndependenceCopula(),
    "LOWER_FRECHET": lambda: cp.LowerFrechet(),
    "UPPER_FRECHET": lambda: cp.UpperFrechet(),
    "PI_OVER_SIGMA_MINUS_PI": lambda: Families.PI_OVER_SIGMA_MINUS_PI.cls(),
    "DIAGONAL_BAND": lambda: cp.DiagonalBandCopula(0.4),
    "XI_NU_BOUNDARY": lambda: cp.XiNuBoundaryCopula(1.5),
    "XI_PSI_BOUNDARY": lambda: cp.XiPsiApproxLowerBoundaryCopula(0.3, 0.4),
    "XI_RHO_BOUNDARY": lambda: cp.XiRhoBoundaryCopula(1.5),
}

# additional variants / classes outside the Families registry
_EXTRA = {
    "Clayton(-0.5)": lambda: cp.Clayton(-0.5),
    "Frank(-4)": lambda: cp.Frank(-4),
    "AliMikhailHaq(-0.7)": lambda: cp.AliMikhailHaq(-0.7),
    "GumbelHougaard(4)": lambda: cp.GumbelHougaard(4),
    "Joe(4)": lambda: cp.Joe(4),
    "Gaussian(-0.7)": lambda: cp.Gaussian(-0.7),
    "StudentT(-0.3,5.5)": lambda: cp.StudentT(-0.3, 5.5),
    "FGM(-0.8)": lambda: cp.FarlieGumbelMorgenstern(-0.8),
    "Plackett(0.2)": lambda: cp.Plackett(0.2),
    "Mardia(-0.4)": lambda: cp.Mardia(-0.4),
    "Frechet(0.3,0.3)": lambda: cp.Frechet(0.3, 0.3),
    "XiRhoBoundary(-0.8)": lambda: cp.XiRhoBoundaryCopula(-0.8),
    "XiBetaBoundary(0.6)": lambda: cp.XiBetaBoundaryCopula(0.6),
    "XiBetaBoundary(-0.4)": lambda: cp.XiBetaBoundaryCopula(-0.4),
    "MedianSwap(0.2)": lambda: cp.MedianSwapCopula(0.2),
    "VThreshold(0.7)": lambda: cp.VThresholdCopula(0.7),
    "VThreshold(1.4)": lambda: cp.VThresholdCopula(1.4),
    "EndSwap(0.2)": lambda: cp.EndSwapCopula(0.2),
    "B11(0.4)": lambda: cp.B11(0.4),
    "BivCheckMixed": lambda: cp.BivCheckMixed(
        _doubly_stochastic(3, 7), sign=[[1, 0, -1], [0, 1, 0], [-1, 0, 1]]
    ),
    "BivBlockDiagMixed": lambda: cp.BivBlockDiagMixed(
        [2, 1], sign=[[1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ),
    "BivIndependenceCopula": lambda: cp.BivIndependenceCopula(),
}


def _all_factories():
    out = {}
    for member in Families:
        name = member.name
        out[name] = _EXPLICIT.get(name) or (lambda name=name: _from_family_reps(name))
    out.update(_EXTRA)
    return out


FACTORIES = _all_factories()
IDS = sorted(FACTORIES)


@cache
def instance(name):
    """Cached copula instance for representative ``name``."""
    return FACTORIES[name]()
