from copul.checkerboard.bernstein import BernsteinCopula
from copul.families.core.copula_sampling_mixin import CopulaSamplingMixin
from copul.families.core.biv_core_copula import BivCoreCopula
from typing import TypeAlias


class BivBernsteinCopula(BernsteinCopula, BivCoreCopula, CopulaSamplingMixin):
    params: list = []
    intervals: dict = {}

BivBernstein: TypeAlias = BivBernsteinCopula