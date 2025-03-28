from copul.checkerboard.bernstein import BernsteinCopula
from copul.families.core.biv_core_copula import BivCoreCopula
from copul.families.core.copula_plotting_mixin import CopulaPlottingMixin

class BivBernsteinCopula(BernsteinCopula, BivCoreCopula, CopulaPlottingMixin):
    params: list = []
    intervals: dict = {}