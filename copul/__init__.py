"""
copul -- copulas and dependence measures for research.

Symbolic (SymPy) and numerical bivariate copula families, checkerboard and
Bernstein approximations, a registry-based engine for dependence measures
(Chatterjee's xi, Spearman's rho, Kendall's tau, Blest's nu, ...), exact
regions between measures, exact optimisation over checkerboard copulas and a
counterexample search.

>>> import copul as cp
>>> cop = cp.Clayton(theta=2)
>>> round(cop.kendalls_tau(), 6)
0.5
>>> cp.compute_measures(cop, ["tau", "beta"])  # doctest: +SKIP
{'tau': 0.5, 'beta': 0.49...}

Subpackages
-----------
measures
    Registry of dependence measures, numerical engine, ``measure_curve``.
optim
    Exact LP/QP optimisation over checkerboard copulas (needs ``cvxpy``,
    ``pip install copul[optim]``; imported lazily at use time).
regions
    Registry of known exact regions between pairs of measures.
search
    Random checkerboards, counterexample search and inequality checks.
"""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from copul import measures, optim, regions, search, stats
from copul.chatterjee import xi_ncalculate
from copul.checkerboard.bernstein import Bernstein, BernsteinCopula
from copul.checkerboard.biv_bernstein import BivBernstein, BivBernsteinCopula
from copul.checkerboard.biv_block_diag_mixed import BivBlockDiagMixed
from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_mixed import BivCheckMixed
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_w import BivCheckW
from copul.checkerboard.check_min import CheckMin
from copul.checkerboard.check_pi import CheckPi
from copul.checkerboard.checkerboarder import Checkerboarder, from_data, from_samples
from copul.checkerboard.matrix import from_matrix
from copul.checkerboard.shuffle_min import ShuffleOfMin
from copul.family.archimedean import (
    AliMikhailHaq,
    BivClayton,
    Clayton,
    Frank,
    GenestGhoudi,
    GumbelBarnett,
    GumbelHougaard,
    Joe,
    Nelsen1,
    Nelsen2,
    Nelsen3,
    Nelsen4,
    Nelsen5,
    Nelsen6,
    Nelsen7,
    Nelsen8,
    Nelsen9,
    Nelsen10,
    Nelsen11,
    Nelsen12,
    Nelsen13,
    Nelsen14,
    Nelsen15,
    Nelsen16,
    Nelsen17,
    Nelsen18,
    Nelsen19,
    Nelsen20,
    Nelsen21,
    Nelsen22,
)
from copul.family.archimedean.archimedean_copula import from_generator
from copul.family.bb import BB1, BB2, BB3, BB6, BB7, BB8, BB9, BB10
from copul.family.constructions import (
    gluing,
    khoudraji,
    mixture,
    ordinal_sum,
    reflect,
    rotate,
    survival,
    transpose,
)
from copul.family.copula_builder import (
    from_cdf,
    from_cond_distr_1,
    from_cond_distr_2,
    from_pdf,
)
from copul.family.core.biv_copula import BivCopula
from copul.family.elliptical import Gaussian, Laplace, StudentT
from copul.family.extreme_value import (
    BB5,
    CuadrasAuge,
    Galambos,
    GumbelHougaardEV,
    HueslerReiss,
    JoeEV,
    MarshallOlkin,
    Tawn,
    tEV,
)
from copul.family.extreme_value.biv_extreme_value_copula import from_pickands
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.frechet.frechet import Frechet
from copul.family.frechet.lower_frechet import LowerFrechet
from copul.family.frechet.mardia import Mardia
from copul.family.frechet.upper_frechet import UpperFrechet
from copul.family.other.b11 import B11
from copul.family.other.clamped_parabola_copula import XiNuBoundaryCopula
from copul.family.other.diagonal_band_copula import DiagonalBandCopula
from copul.family.other.diagonal_strip_copula import XiPsiApproxLowerBoundaryCopula
from copul.family.other.end_swap_copula import EndSwapCopula
from copul.family.other.farlie_gumbel_morgenstern import FarlieGumbelMorgenstern
from copul.family.other.independence_copula import IndependenceCopula
from copul.family.other.median_swap_copula import MedianSwapCopula
from copul.family.other.plackett import Plackett
from copul.family.other.raftery import Raftery
from copul.family.other.v_threshold_copula import VThresholdCopula
from copul.family.other.xi_beta_boundary_copula import XiBetaBoundaryCopula
from copul.family.other.xi_rho_boundary_copula import XiRhoBoundaryCopula
from copul.family_list import Families, approximations, copulas, families
from copul.measures import compute as compute_measures
from copul.measures import from_measure, measure_curve
from copul.regions import get as get_region
from copul.schur_order.bounds_from_xi import bounds_from_xi
from copul.schur_order.cis_rearranger import CISRearranger
from copul.schur_order.cis_verifier import CISVerifier
from copul.schur_order.corner_set_verifier import CornerSetVerifier
from copul.schur_order.ltd_verifier import LTDVerifier
from copul.schur_order.plod_verifier import PLODVerifier
from copul.search import check_inequality, find_counterexample, random_checkerboards
from copul.star_product import markov_product
from copul.stats import EmpiricalCopula, estimate, fit, gof_test, pseudo_obs, select

try:
    __version__ = _pkg_version("copul")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

# Library code never configures logging; applications do.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "B11",
    "BB1",
    "BB2",
    "BB3",
    "BB5",
    "BB6",
    "BB7",
    "BB8",
    "BB9",
    "BB10",
    "AliMikhailHaq",
    "Bernstein",
    "BernsteinCopula",
    "BivBernstein",
    "BivBernsteinCopula",
    "BivBlockDiagMixed",
    "BivCheckMin",
    "BivCheckMixed",
    "BivCheckPi",
    "BivCheckW",
    "BivClayton",
    "BivCopula",
    "BivIndependenceCopula",
    "CISRearranger",
    "CISVerifier",
    "CheckMin",
    "CheckPi",
    "Checkerboarder",
    "Clayton",
    "CornerSetVerifier",
    "CuadrasAuge",
    "DiagonalBandCopula",
    "EmpiricalCopula",
    "EndSwapCopula",
    "Families",
    "FarlieGumbelMorgenstern",
    "Frank",
    "Frechet",
    "Galambos",
    "Gaussian",
    "GenestGhoudi",
    "GumbelBarnett",
    "GumbelHougaard",
    "GumbelHougaardEV",
    "HueslerReiss",
    "IndependenceCopula",
    "Joe",
    "JoeEV",
    "LTDVerifier",
    "Laplace",
    "LowerFrechet",
    "Mardia",
    "MarshallOlkin",
    "MedianSwapCopula",
    "Nelsen1",
    "Nelsen2",
    "Nelsen3",
    "Nelsen4",
    "Nelsen5",
    "Nelsen6",
    "Nelsen7",
    "Nelsen8",
    "Nelsen9",
    "Nelsen10",
    "Nelsen11",
    "Nelsen12",
    "Nelsen13",
    "Nelsen14",
    "Nelsen15",
    "Nelsen16",
    "Nelsen17",
    "Nelsen18",
    "Nelsen19",
    "Nelsen20",
    "Nelsen21",
    "Nelsen22",
    "PLODVerifier",
    "Plackett",
    "Raftery",
    "ShuffleOfMin",
    "StudentT",
    "Tawn",
    "UpperFrechet",
    "VThresholdCopula",
    "XiBetaBoundaryCopula",
    "XiNuBoundaryCopula",
    "XiPsiApproxLowerBoundaryCopula",
    "XiRhoBoundaryCopula",
    "__version__",
    "approximations",
    "bounds_from_xi",
    "check_inequality",
    "compute_measures",
    "copulas",
    "estimate",
    "families",
    "find_counterexample",
    "fit",
    "from_cdf",
    "from_cond_distr_1",
    "from_cond_distr_2",
    "from_data",
    "from_generator",
    "from_matrix",
    "from_measure",
    "from_pdf",
    "from_pickands",
    "from_samples",
    "get_region",
    "gluing",
    "gof_test",
    "khoudraji",
    "markov_product",
    "measure_curve",
    "measures",
    "mixture",
    "optim",
    "ordinal_sum",
    "pseudo_obs",
    "random_checkerboards",
    "reflect",
    "regions",
    "rotate",
    "search",
    "select",
    "stats",
    "survival",
    "tEV",
    "transpose",
    "xi_ncalculate",
]
