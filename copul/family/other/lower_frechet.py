from copul.family.other.frechet import Frechet


class LowerFrechet(Frechet):
    _alpha = 0
    _beta = 1

    @property
    def alpha(self):
        return 0

    @property
    def beta(self):
        return 1
