from copul.families.core.copula import Copula
from copul.families.core.biv_core_copula import BivCoreCopula


class BivCopula(Copula, BivCoreCopula):
    def __init__(self, *args, **kwargs):
        if "dimension" in kwargs:
            kwargs.pop("dimension")
        Copula.__init__(self, 2, *args, **kwargs)
        BivCoreCopula.__init__(self)
