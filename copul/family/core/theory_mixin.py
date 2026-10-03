r"""
Thin method wrappers around :mod:`copul.theory` for bivariate copulas.

Every method delegates to the function of the same name in
:mod:`copul.theory` (imported lazily), so that, e.g.,
``C.kendall_distribution(t)`` equals
``copul.theory.kendall_distribution(C, t)``.  Subclasses with closed forms
(Archimedean, extreme-value copulas) override some of them.
"""

from __future__ import annotations

from typing import Any


class TheoryMixin:
    """Methods connecting copula objects with :mod:`copul.theory`."""

    # -- Archimedean / Kendall distribution --------------------------------
    def kendall_distribution(self, t: Any = None, **kwargs):
        r"""Kendall distribution function :math:`K_C(t)=P(C(U,V)\le t)`.

        See :func:`copul.theory.archimedean.kendall_distribution`.
        """
        from copul.theory.archimedean import kendall_distribution

        return kendall_distribution(self, t, **kwargs)

    def associativity_defect(self, **kwargs) -> float:
        r""":math:`\sup|C(C(u,v),w)-C(u,C(v,w))|` on a grid.

        See :func:`copul.theory.archimedean.associativity_defect`.
        """
        from copul.theory.archimedean import associativity_defect

        return associativity_defect(self, **kwargs)

    def is_archimedean(self, **kwargs) -> bool:
        """Archimedean characterization (associativity and :math:`C(t,t)<t`).

        See :func:`copul.theory.archimedean.is_archimedean`.
        """
        from copul.theory.archimedean import is_archimedean

        return is_archimedean(self, **kwargs)

    # -- extreme-value theory ----------------------------------------------
    def ev_attractor(self, **kwargs):
        r"""Extreme-value attractor :math:`\lim_n C(u^{1/n},v^{1/n})^n`.

        See :func:`copul.theory.extreme_value.ev_attractor`.
        """
        from copul.theory.extreme_value import ev_attractor

        return ev_attractor(self, **kwargs)

    def tail_copula(self, x: Any, y: Any, lower: bool = True, **kwargs):
        r"""Lower (or upper) tail copula :math:`\Lambda(x,y)`.

        See :func:`copul.theory.extreme_value.tail_copula`.
        """
        from copul.theory.extreme_value import tail_copula

        return tail_copula(self, x, y, lower=lower, **kwargs)

    # -- diagonals -----------------------------------------------------------
    def diagonal_section(self, t: Any = None):
        r"""Diagonal section :math:`\delta_C(t)=C(t,t)`.

        Returns a :class:`copul.theory.diagonal.Diagonal` if ``t`` is ``None``,
        else its values at ``t``.
        """
        from copul.theory.diagonal import diagonal_section

        d = diagonal_section(self)
        return d if t is None else d(t)

    def opposite_diagonal(self, t: Any = None):
        r"""Opposite diagonal section :math:`t\mapsto C(t,1-t)`."""
        from copul.theory.diagonal import opposite_diagonal

        d = opposite_diagonal(self)
        return d if t is None else d(t)

    # -- symmetry ------------------------------------------------------------
    def nonexchangeability(self, p: float = float("inf"), **kwargs) -> float:
        r"""Non-exchangeability :math:`\mu_\infty(C)=3\sup|C-C^\top|` (or :math:`L^p`).

        See :func:`copul.theory.symmetry.nonexchangeability`.
        """
        from copul.theory.symmetry import nonexchangeability

        return nonexchangeability(self, p=p, **kwargs)

    def radial_asymmetry(self, p: float = float("inf"), **kwargs) -> float:
        r""":math:`\sup|C-\hat C|` (or the :math:`L^p` distance).

        See :func:`copul.theory.symmetry.radial_asymmetry`.
        """
        from copul.theory.symmetry import radial_asymmetry

        return radial_asymmetry(self, p=p, **kwargs)

    def symmetrize(self):
        r"""The exchangeable copula :math:`\tfrac12(C+C^\top)`."""
        from copul.theory.symmetry import symmetrize

        return symmetrize(self)

    def radial_symmetrize(self):
        r"""The radially symmetric copula :math:`\tfrac12(C+\hat C)`."""
        from copul.theory.symmetry import radial_symmetrize

        return radial_symmetrize(self)

    # -- quasi-copulas -------------------------------------------------------
    def two_increasing_defect(self, **kwargs):
        """Most negative rectangle volume on a grid (0 for genuine copulas).

        See :func:`copul.theory.quasi.two_increasing_defect`.
        """
        from copul.theory.quasi import two_increasing_defect

        return two_increasing_defect(self, **kwargs)

    # -- Markov theory and distances ----------------------------------------
    def is_completely_dependent(self, **kwargs) -> bool:
        r"""Whether :math:`V=f(U)` a.s., i.e. :math:`\partial_1C\in\{0,1\}` a.e.

        See :func:`copul.theory.markov.is_completely_dependent`.
        """
        from copul.theory.markov import is_completely_dependent

        return bool(is_completely_dependent(self, **kwargs))

    def distance(self, other: Any, metric: str = "sup", **kwargs):
        """Distance to another copula (``"sup"``, ``"L1"``, ``"L2"``, ``"D1"``,
        ``"D2"``, ``"Dinf"``).

        See :func:`copul.theory.distances.copula_distance`.
        """
        from copul.theory.distances import copula_distance

        return copula_distance(self, other, metric=metric, **kwargs)
