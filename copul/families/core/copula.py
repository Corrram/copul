from copul.families.core.copula_approximator_mixin import CopulaApproximatorMixin
from copul.families.core.copula_plotting_mixin import CopulaPlottingMixin
from copul.families.core.copula_sampling_mixin import CopulaSamplingMixin
from copul.families.core.core_copula import CoreCopula


class Copula(
    CoreCopula, CopulaSamplingMixin, CopulaPlottingMixin, CopulaApproximatorMixin
):
    pass
