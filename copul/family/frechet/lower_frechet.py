from copul.family.frechet.frechet import Frechet


class LowerFrechet(Frechet):
    _alpha = 0
    _beta = 1

    @property
    def alpha(self):
        return 0

    @property
    def beta(self):
        return 1


if __name__ == "__main__":
    lf = LowerFrechet()
    print(lf.cdf(0.2, 0.3))
