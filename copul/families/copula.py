from copul.families.core_copula import CoreCopula
from copul.checkerboard.checkerboarder import Checkerboarder


class Copula(CoreCopula):
    def rvs(self, n=1, precision=2):
        checkerboarder = Checkerboarder(10**precision, dim=self.dimension)
        ccop = checkerboarder.compute_check_pi(self)
        return ccop.rvs(n)
