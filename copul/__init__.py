import logging
import os
import sys

from copul.chatterjee import xi_ncalculate
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_w import BivCheckW
from copul.checkerboard.check_min import CheckMin
from copul.checkerboard.check_pi import CheckPi
from copul.checkerboard.bernstein import BernsteinCopula, Bernstein
from copul.checkerboard.shuffle_min import ShuffleOfMin
from copul.checkerboard.biv_bernstein import BivBernsteinCopula, BivBernstein
from copul.checkerboard.checkerboarder import Checkerboarder, from_data
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
from copul.family.core.biv_copula import BivCopula
from copul.family.copula_builder import from_cdf
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
from copul.family.other.farlie_gumbel_morgenstern import FarlieGumbelMorgenstern
from copul.family.frechet.frechet import Frechet
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.frechet.lower_frechet import LowerFrechet
from copul.family.frechet.mardia import Mardia
from copul.family.other.plackett import Plackett
from copul.family.other.raftery import Raftery
from copul.family.other.diagonal_band_copula import DiagonalBandCopula
from copul.family.frechet.upper_frechet import UpperFrechet
from copul.family.other.xi_rho_boundary_copula import XiRhoBoundaryCopula
from copul.family.other.clamped_parabola_copula import ClampedParabolaCopula

from copul.family_list import Families, families, approximations, copulas
from copul.schur_order.cis_rearranger import CISRearranger
from copul.schur_order.cis_verifier import CISVerifier
from copul.schur_order.ltd_verifier import LTDVerifier
from copul.schur_order.plod_verifier import PLODVerifier
from copul.star_product import markov_product

__all__ = [
    "xi_ncalculate",
    # Checkerboard related objects
    "Bernstein",
    "BernsteinCopula",
    "BivBernstein",
    "BivBernsteinCopula",
    "BivCheckPi",
    "BivCheckMin",
    "BivCheckW",
    "BivClayton",
    "CheckMin",
    "CheckPi",
    "Checkerboarder",
    "from_data",
    # Families & Builders
    "BivCopula",
    "DiagonalBandCopula",
    "from_cdf",
    "approximations",
    "copulas",
    "families",
    "Families",
    # Archimedean copulas
    "AliMikhailHaq",
    "Clayton",
    "Frank",
    "GenestGhoudi",
    "GumbelHougaard",
    "GumbelHougaardEV",
    "GumbelBarnett",
    "Joe",
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
    # Extreme Value copulas
    "BB5",
    "CuadrasAuge",
    "Galambos",
    "HueslerReiss",
    "JoeEV",
    "MarshallOlkin",
    "Tawn",
    "tEV",
    # Elliptical copulas
    "Gaussian",
    "Laplace",
    "StudentT",
    # Other copulas
    "FarlieGumbelMorgenstern",
    "Frechet",
    "BivIndependenceCopula",
    "LowerFrechet",
    "Mardia",
    "Plackett",
    "Raftery",
    "XiRhoBoundaryCopula",
    "ClampedParabolaCopula",
    "UpperFrechet",
    # Miscellaneous
    "CISRearranger",
    "CISVerifier",
    "LTDVerifier",
    "PLODVerifier",
    "ShuffleOfMin",
    "markov_product",
]

logging_format = "%(message)s"
logging.basicConfig(
    level="INFO", format=logging_format, handlers=[logging.StreamHandler()]
)
log = logging.getLogger("ConsoleLogger")

for logger in logging.root.manager.loggerDict:
    if logger[:6] == "dedupe":
        logging.getLogger(logger).setLevel(logging.WARNING)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
