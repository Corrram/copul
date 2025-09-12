from copul.family.core.copula_approximator_mixin import CopulaApproximatorMixin
from copul.family.core.copula_plotting_mixin import CopulaPlottingMixin
from copul.family.core.copula_sampling_mixin import CopulaSamplingMixin
from copul.family.core.core_copula import CoreCopula


class Copula(
    CoreCopula, CopulaSamplingMixin, CopulaPlottingMixin, CopulaApproximatorMixin
):
    pass
