"""
Dependence measures of bivariate copulas: registry, numerical engine, dispatch.

Quick start
-----------
>>> import copul as cp
>>> from copul.measures import compute
>>> compute(cp.Clayton(2), ["tau", "rho"])          # doctest: +SKIP
{'tau': 0.5, 'rho': 0.6822338332...}
>>> round(cp.Clayton(2).spearmans_rho(), 10)       # float, numerical
0.6822338333
>>> cp.FarlieGumbelMorgenstern().spearmans_rho()    # free parameter: SymPy
theta/3

Modules
-------
registry
    :class:`Measure` records (keys, aliases, formulas, ranges, values at
    M/W/Pi) -- :func:`get_measure`, :func:`list_measures`.
quadrature
    Vectorized adaptive Gauss--Kronrod and composite Gauss--Legendre rules
    on [0,1] and [0,1]^2 (interior nodes only).
numeric
    Pure functions computing measures from vectorized callables or h-grids,
    e.g. :func:`xi_from_h`, :func:`rho_from_cdf`, :func:`measures_from_h`.
backend
    :func:`numeric_backend` -- vectorized cdf/h1/h2/pdf of copula objects.
engine
    :func:`compute` and the ``method=`` dispatch used by all measure methods.
curves
    :func:`measure_curve` and :func:`from_measure` (calibration).
"""

from copul.measures.backend import NumericBackend, free_parameters, numeric_backend
from copul.measures.curves import (
    MeasureCurve,
    default_parameter_values,
    from_measure,
    measure_curve,
)
from copul.measures.engine import (
    MEASURE_METHODS,
    MeasureResult,
    compute,
    symbolic_measure,
)
from copul.measures.numeric import (
    NumericCopula,
    beta_from_cdf,
    bkr_from_h,
    footrule_from_cdf,
    gamma_from_cdf,
    hoeffdings_d_from_cdf,
    kappa_from_cdf,
    lambda_l_from_cdf,
    lambda_u_from_cdf,
    lp_constant,
    lp_from_cdf,
    measures_from_cdf,
    measures_from_h,
    mutual_information_from_pdf,
    nu_from_cdf,
    nu_from_h,
    rho_from_cdf,
    rho_from_h,
    sigma_from_cdf,
    tau_from_h,
    xi_from_h,
)
from copul.measures.quadrature import (
    gauss_legendre_1d,
    gauss_legendre_2d,
    integrate_1d,
    integrate_1d_batch,
    integrate_2d,
)
from copul.measures.registry import (
    DEFAULT_MEASURES,
    MEASURES,
    Measure,
    get_measure,
    list_measures,
    resolve_key,
)

__all__ = [
    "DEFAULT_MEASURES",
    "MEASURES",
    "MEASURE_METHODS",
    "Measure",
    "MeasureCurve",
    "MeasureResult",
    "NumericBackend",
    "NumericCopula",
    "beta_from_cdf",
    "bkr_from_h",
    "compute",
    "default_parameter_values",
    "footrule_from_cdf",
    "free_parameters",
    "from_measure",
    "gamma_from_cdf",
    "gauss_legendre_1d",
    "gauss_legendre_2d",
    "get_measure",
    "hoeffdings_d_from_cdf",
    "integrate_1d",
    "integrate_1d_batch",
    "integrate_2d",
    "kappa_from_cdf",
    "lambda_l_from_cdf",
    "lambda_u_from_cdf",
    "list_measures",
    "lp_constant",
    "lp_from_cdf",
    "measure_curve",
    "measures_from_cdf",
    "measures_from_h",
    "mutual_information_from_pdf",
    "nu_from_cdf",
    "nu_from_h",
    "numeric_backend",
    "resolve_key",
    "rho_from_cdf",
    "rho_from_h",
    "sigma_from_cdf",
    "symbolic_measure",
    "tau_from_h",
    "xi_from_h",
]
