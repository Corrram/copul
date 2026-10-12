"""Symbolic population measures added by the rank-correlation atlas."""

import sympy as sp


def _symbolic(copula, key, args, kwargs, condition_on_y=False):
    copula._set_params(args, kwargs)
    u, v = copula.u, copula.v
    c = getattr(copula.cdf, "func", None)
    if c is None:
        c = copula.cdf().func
    if condition_on_y:
        c = c.subs({u: v, v: u}, simultaneous=True)
    h1, h2 = sp.diff(c, u), sp.diff(c, v)

    def integral(expr):
        return sp.integrate(expr, (u, 0, 1), (v, 0, 1))

    if key == "eta":
        return sp.simplify(12 * integral((2 - u - v) * c) - 2)
    if key == "omega":
        return sp.sqrt(sp.simplify(3 * integral(h1**2 + h2**2) - 2))
    if key == "chi":
        regression = sp.integrate(h1, (v, 0, 1)) - sp.Rational(1, 2)
        return sp.simplify(12 * sp.integrate(regression**2, (u, 0, 1)))
    if key == "integrated_r2":
        return sp.simplify(integral(sp.cancel((h1 - v) ** 2 / (v * (1 - v)))))
    if key in ("hoeffding_d", "tau_star"):
        d = -60 * integral(h1 * (c - u * v) * (h2 - u))
        if key == "hoeffding_d":
            return sp.simplify(d)
        return sp.simplify(sp.Rational(2, 5) * d + 24 * integral((c - u * v) ** 2))
    if key in ("normal_score", "savage_score"):
        # Hoeffding's covariance identity applies to singular copulas too.
        if key == "savage_score":
            weight = 1 / ((1 - u) * (1 - v))
        else:
            weight = 2 * sp.pi * sp.exp(sp.erfinv(2 * u - 1) ** 2 + sp.erfinv(2 * v - 1) ** 2)
        return sp.Integral((c - u * v) * weight, (u, 0, 1), (v, 0, 1))
    if key in ("linfoot", "hellinger", "silvey"):
        if copula.is_absolutely_continuous is not True:
            raise NotImplementedError(
                "Symbolic divergence requires a known absolutely continuous copula; "
                "bind its parameters and use method='numeric' for singular components."
            )
        density = sp.diff(c, u, v)
        if key == "linfoot":
            information = sp.Integral(
                sp.Piecewise((0, sp.Eq(density, 0)), (density * sp.log(density), True)),
                (u, 0, 1),
                (v, 0, 1),
            )
            return sp.sqrt(1 - sp.exp(-2 * information))
        if key == "silvey":
            return sp.Integral(sp.Abs(density - 1) / 2, (u, 0, 1), (v, 0, 1))
        affinity = sp.Integral(sp.sqrt(density), (u, 0, 1), (v, 0, 1))
        return sp.sqrt(4 * (1 - affinity**4) / (2 - affinity**4 + sp.sqrt(4 - 3 * affinity**4)))
    raise NotImplementedError(f"No symbolic representation of {key}.")


class RankMeasuresMixin:
    """Population definitions; numerical evaluation is handled by the shared dispatcher."""

    def blests_eta(self, *args, **kwargs):
        r"""Symmetrized Blest: :math:`\eta=(\nu(C)+\nu(C^\top))/2`."""
        return _symbolic(self, "eta", args, kwargs)

    def normal_score_correlation(self, *args, **kwargs):
        r"""Normal-score correlation :math:`E[\Phi^{-1}(U)\Phi^{-1}(V)]`."""
        return _symbolic(self, "normal_score", args, kwargs)

    def savage_score_correlation(self, *args, **kwargs):
        r"""Savage-score correlation :math:`E[\log(1-U)\log(1-V)]-1`."""
        return _symbolic(self, "savage_score", args, kwargs)

    def siburg_stoimenov(self, *args, **kwargs):
        r"""Siburg--Stoimenov measure :math:`\sqrt{(\xi(C)+\xi(C^\top))/2}`."""
        return _symbolic(self, "omega", args, kwargs)

    def hoeffding_d(self, *args, **kwargs):
        r"""Hoeffding's :math:`D=30\int(C-uv)^2\,dC`.

        This is distinct from the legacy :meth:`hoeffdings_d` method, which
        returns :math:`\Phi^2`. Integration by parts gives
        :math:`D=-60\int\partial_1C(C-uv)(\partial_2C-u)\,du\,dv`,
        valid also for singular copulas.
        """
        return _symbolic(self, "hoeffding_d", args, kwargs)

    def bergsma_dassios(self, *args, **kwargs):
        r"""Sign covariance :math:`\tau^*=2D/5+4\Phi^2/15`, range [0, 2/3]."""
        return _symbolic(self, "tau_star", args, kwargs)

    def correlation_ratio(self, *args, condition_on_y=False, **kwargs):
        r"""Copula correlation ratio :math:`\chi=12\operatorname{Var}(E[V\mid U])`.

        Set ``condition_on_y=True`` to compute the ratio for the transpose.
        The coefficient is the explained variance, without a square root.
        """
        return _symbolic(self, "chi", args, kwargs, condition_on_y)

    def integrated_r2(self, *args, condition_on_y=False, **kwargs):
        r"""Azadkia--Roudaki :math:`\mathcal A=\int(h_1-v)^2/[v(1-v)]`.

        Set ``condition_on_y=True`` to compute the coefficient of the
        transpose. Interior quadrature nodes avoid endpoint denominators.
        """
        return _symbolic(self, "integrated_r2", args, kwargs, condition_on_y)

    def maximal_correlation(self, *args, max_grid=256, **kwargs):
        r"""Gebelein--Renyi maximal correlation :math:`\sup_{f,g}\mathrm{corr}(f(U),g(V))`.

        Parameters
        ----------
        max_grid : int, default 256
            Largest dyadic partition for the generic numerical approximation;
            must be a power of two of at least 16. The finite-partition result
            is a lower approximation, converging to R as the mesh is refined.
            Its refinement difference is a diagnostic, not an error bound.
            Independence checkerboards and Gaussian copulas have exact routes.
        """
        raise NotImplementedError("Maximal correlation requires method='numeric' or 'auto'.")

    def linfoot_correlation(self, *args, **kwargs):
        r"""Linfoot :math:`\mathcal L=\sqrt{1-e^{-2I}}`, equal to 1 with any singular mass."""
        return _symbolic(self, "linfoot", args, kwargs)

    def hellinger_correlation(self, *args, **kwargs):
        r"""Gaussian-calibrated Hellinger correlation (Geenens--Lafaye de Micheaux).

        With affinity :math:`B=\int\sqrt{c_C}`, including the absolutely
        continuous part of mixed copulas, returns
        :math:`2B^{-2}\sqrt{B^4+\sqrt{4-3B^4}-2}`; the value at B=0 is 1.
        """
        return _symbolic(self, "hellinger", args, kwargs)

    def silvey_delta(self, *args, **kwargs):
        r"""Silvey total variation :math:`\Delta_S=(\int|c_C-1|+s_C)/2`.

        Here s_C is the singular mass. This compares probability measures,
        whereas Schweizer--Wolff sigma compares distribution functions.
        """
        return _symbolic(self, "silvey", args, kwargs)
