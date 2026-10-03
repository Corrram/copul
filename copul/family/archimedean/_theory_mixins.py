r"""
Thin wrapper methods of the Archimedean copula classes around
:mod:`copul.theory.archimedean` (imported lazily to avoid import cycles).
"""

from __future__ import annotations

__all__ = ["ArchimedeanGeneratorMixin", "BivArchimedeanTheoryMixin"]


class ArchimedeanGeneratorMixin:
    """Generator theory methods of Archimedean copulas (any dimension)."""

    def generator_properties(self, d_max: int = 10, **kwargs):
        """Analytic properties of the generator.

        See :func:`copul.theory.archimedean.generator_properties`.
        """
        from copul.theory.archimedean import generator_properties

        return generator_properties(self, d_max=d_max, **kwargs)

    def max_dimension(self, d_max: int = 10, **kwargs):
        r"""Largest :math:`d` for which the generator gives a :math:`d`-copula.

        See :func:`copul.theory.archimedean.max_dimension`.
        """
        from copul.theory.archimedean import max_dimension

        return max_dimension(self, d_max=d_max, **kwargs)


class BivArchimedeanTheoryMixin(ArchimedeanGeneratorMixin):
    """Kendall distribution and zero curve of bivariate Archimedean copulas."""

    def kendall_distribution(self, t, **kwargs):
        r"""Kendall distribution :math:`K_C(t) = t - \varphi(t)/\varphi'(t^+)`.

        See :func:`copul.theory.archimedean.kendall_distribution`.
        """
        from copul.theory.archimedean import kendall_distribution

        return kendall_distribution(self, t, **kwargs)

    def kendall_distribution_inverse(self, p, **kwargs):
        r"""Quantile function :math:`K_C^{-1}(p)` of the Kendall distribution.

        See :func:`copul.theory.archimedean.kendall_distribution_inverse`.
        """
        from copul.theory.archimedean import kendall_distribution_inverse

        return kendall_distribution_inverse(self, p, **kwargs)

    def zero_curve(self, **kwargs):
        r"""Boundary of the zero set :math:`\{C=0\}` and its singular mass.

        See :func:`copul.theory.archimedean.zero_curve`.
        """
        from copul.theory.archimedean import zero_curve

        return zero_curve(self, **kwargs)
