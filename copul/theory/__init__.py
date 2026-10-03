r"""
Copula theory toolkit.

Functions that work on *any* bivariate copula object of :mod:`copul`
(parametric families, checkerboards, constructions, empirical copulas):
Archimedean and extreme-value theory, dependence concepts and orderings,
Markov-operator theory and distances between copulas, pointwise bounds,
quasi-copulas, diagonal sections and (a)symmetry.

All results implemented here are standard published theory; every function
documents its references (mainly Nelsen, *An Introduction to Copulas*, 2006;
Durante & Sempi, *Principles of Copula Theory*, 2016; Joe, *Dependence
Modeling with Copulas*, 2014).

Submodules
----------
``archimedean``
    Kendall distribution function, generator checks (validity, strictness,
    d-monotonicity, maximal dimension), zero curves, associativity and
    Archimedean characterization, numerical Archimedean copulas from
    generators, Laplace transforms or Kendall distributions.
``extreme_value``
    Pickands checks, max-stability, extremal coefficient, stable tail
    dependence function, extreme-value attractors, tail copulas, Pickands
    estimators, numerical EV copulas from Pickands functions.
``dependence``
    PQD/NQD, LTD/LTI, RTI/RTD, SI/SD, LCSD/RCSI, TP2/RR2 with exact family
    characterizations and dependence profiles checked against the implication
    hierarchy.
``orders``
    Concordance (PQD) order, parameter monotonicity of families, "more SI"
    order.
``markov``
    Markov products and powers, invertibility, complete dependence,
    idempotents, Markov operators, conditional expectations and quantiles.
``distances``
    Sup, :math:`L^p` and Trutschnig's :math:`\partial`-distances between
    copulas and the dependence measure :math:`\zeta_1`.
``quasi``
    Quasi-copulas, 2-increasing defects, pointwise lattice operations.
``bounds``
    Fréchet–Hoeffding bounds, best-possible bounds given a value
    :math:`C(a,b)`, a diagonal, or a value of Kendall's :math:`\tau`,
    Spearman's :math:`\rho` or Blomqvist's :math:`\beta`.
``diagonal``
    Diagonal sections, Bertino and diagonal copulas.
``symmetry``
    Non-exchangeability, radial asymmetry, symmetrizations and tests of
    exchangeability and radial symmetry from data.
"""

from copul.theory import (
    archimedean,
    bounds,
    dependence,
    diagonal,
    distances,
    extreme_value,
    markov,
    orders,
    quasi,
    symmetry,
)
from copul.theory.archimedean import (
    archimedean_from_generator,
    archimedean_from_kendall_distribution,
    associativity_defect,
    check_generator,
    from_laplace_transform,
    generator_properties,
    is_archimedean,
    kendall_distribution,
    kendall_distribution_inverse,
    max_dimension,
    zero_curve,
)
from copul.theory.bounds import (
    ShuffleOfM,
    bounds_given_diagonal,
    bounds_given_measure,
    bounds_given_value,
)
from copul.theory.dependence import (
    PropertyResult,
    check_property,
    dependence_profile,
)
from copul.theory.diagonal import (
    BertinoCopula,
    DiagonalCopula,
    copula_with_diagonal,
    diagonal_section,
    opposite_diagonal,
)
from copul.theory.distances import copula_distance, trutschnig_zeta
from copul.theory.extreme_value import (
    check_pickands,
    ev_attractor,
    ev_copula_from_pickands,
    extremal_coefficient,
    is_extreme_value,
    is_max_stable,
    pickands_estimator,
    pickands_function,
    stable_tail_dependence,
    tail_copula,
)
from copul.theory.markov import (
    MarkovOperator,
    conditional_expectation,
    is_completely_dependent,
    is_idempotent,
    is_invertible,
    is_left_invertible,
    is_mutually_completely_dependent,
    is_right_invertible,
    markov_operator,
    markov_power,
    regression_function,
)
from copul.theory.orders import concordance_order, is_concordance_ordered, is_more_si
from copul.theory.quasi import (
    NumericQuasiCopula,
    copula_max,
    copula_min,
    is_quasi_copula,
    two_increasing_defect,
)
from copul.theory.symmetry import (
    exchangeability_test,
    maximally_nonexchangeable_copula,
    nonexchangeability,
    radial_asymmetry,
    radial_symmetrize,
    radial_symmetry_test,
    symmetrize,
)

__all__ = [
    "BertinoCopula",
    "DiagonalCopula",
    "MarkovOperator",
    "NumericQuasiCopula",
    "PropertyResult",
    "ShuffleOfM",
    "archimedean",
    "archimedean_from_generator",
    "archimedean_from_kendall_distribution",
    "associativity_defect",
    "bounds",
    "bounds_given_diagonal",
    "bounds_given_measure",
    "bounds_given_value",
    "check_generator",
    "check_pickands",
    "check_property",
    "concordance_order",
    "conditional_expectation",
    "copula_distance",
    "copula_max",
    "copula_min",
    "copula_with_diagonal",
    "dependence",
    "dependence_profile",
    "diagonal",
    "diagonal_section",
    "distances",
    "ev_attractor",
    "ev_copula_from_pickands",
    "exchangeability_test",
    "extremal_coefficient",
    "extreme_value",
    "from_laplace_transform",
    "generator_properties",
    "is_archimedean",
    "is_completely_dependent",
    "is_concordance_ordered",
    "is_extreme_value",
    "is_idempotent",
    "is_invertible",
    "is_left_invertible",
    "is_max_stable",
    "is_more_si",
    "is_mutually_completely_dependent",
    "is_quasi_copula",
    "is_right_invertible",
    "kendall_distribution",
    "kendall_distribution_inverse",
    "markov",
    "markov_operator",
    "markov_power",
    "max_dimension",
    "maximally_nonexchangeable_copula",
    "nonexchangeability",
    "opposite_diagonal",
    "orders",
    "pickands_estimator",
    "pickands_function",
    "quasi",
    "radial_asymmetry",
    "radial_symmetrize",
    "radial_symmetry_test",
    "regression_function",
    "stable_tail_dependence",
    "symmetrize",
    "symmetry",
    "tail_copula",
    "trutschnig_zeta",
    "two_increasing_defect",
    "zero_curve",
]
