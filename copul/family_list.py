from enum import Enum

from copul.families import archimedean, elliptical, extreme_value, other


class Families(Enum):
    """
    Enum for **all copula families** available in the package.
    Any of the below listed copula families can be called
    via the enum or directly by calling the class from the package.
    For example, we can call the Clayton copula in the following two ways:

    .. highlight:: python
    .. code-block:: python

        >>> import copul as cp
        >>> cp.Families.CLAYTON.value()
        >>> cp.Clayton()
    """

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

    JOE_EV = extreme_value.JoeEV
    BB5 = extreme_value.BB5
    CUADRAS_AUGE = extreme_value.CuadrasAuge
    GALAMBOS = extreme_value.Galambos
    GUMBEL_HOUGAARD_EV = extreme_value.GumbelHougaard
    HUESSLER_REISS = extreme_value.HueslerReiss
    TAWN = extreme_value.Tawn
    T_EV = extreme_value.tEV
    MARSHALL_OLKIN = extreme_value.MarshallOlkin

    GAUSSIAN = elliptical.Gaussian
    T = elliptical.StudentT
    LAPLACE = elliptical.Laplace

    B11 = other.B11
    CHECKERBOARD = other.CheckerboardCopula
    FARLIE_GUMBEL_MORGENSTERN = other.FarlieGumbelMorgenstern
    FRECHET = other.Frechet
    INDEPENDENCE = other.IndependenceCopula
    LOWER_FRECHET = other.LowerFrechet
    MARDIA = other.Mardia
    PLACKETT = other.Plackett
    RAFTERY = other.Raftery
    UPPER_FRECHET = other.UpperFrechet
