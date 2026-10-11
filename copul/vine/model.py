r"""Simplified pair-copula constructions and Rosenblatt transforms.

For every edge :math:`e=(a,b\mid D)` the density factor is
:math:`c_{ab;D}(F_{a|D},F_{b|D})`. Conditional distributions recurse as

.. math::

   F_{a|D+b} = \partial_2 C_{ab;D}(F_{a|D}, F_{b|D}),\qquad
   F_{b|D+a} = \partial_1 C_{ab;D}(F_{a|D}, F_{b|D}).

Multiplying these factors gives a normalized copula density under the
regular-vine conditions. The simplifying assumption means each pair
copula is constant with respect to the *values* of the conditioning
variables; it does not imply conditional independence. See Aas et al.
(2009), Insurance: Mathematics and Economics 44, 182--198,
https://doi.org/10.1016/j.insmatheco.2007.02.001.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.stats import qmc

from copul.family.constructions._base import component_is_ac, ensure_numeric_copula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.measures.backend import numeric_backend
from copul.multivariate.base import CopulaND, finish, parse_points
from copul.vine.structure import VineStructure, integer

# Probabilities used in density and inverse recursions are clipped for
# stability. Public transform/fitting inputs must be in the open unit cube.
_EPS = 1e-12


def interior(x):
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("A pair copula produced non-finite conditional probabilities.")
    if np.any((x < -1e-8) | (x > 1 + 1e-8)):
        raise FloatingPointError("A pair copula produced conditional probabilities outside [0, 1].")
    return np.clip(x, _EPS, 1 - _EPS)


def pair_backend(copula):
    ensure_numeric_copula(copula, "pair copula")
    if not component_is_ac(copula):
        raise ValueError("Vines require absolutely continuous pair copulas.")
    be = numeric_backend(copula)
    for name in ("h1", "h2", "pdf"):
        be.get(name)
        if be.source.get(name) == "finite_differences":
            raise ValueError(f"Pair copula needs an analytic or symbolic {name}.")
    return be


class VineCopula(CopulaND):
    """Simplified regular vine with fully specified bivariate copulas.

    Parameters
    ----------
    structure : VineStructure
        Validated trees, pair orientation and forward sampling order.
    pair_copulas : sequence of sequences of bivariate copula objects, optional
        Same tree/edge layout as ``structure.trees``. Omission makes all
        edges independent. Components are copied to isolate the model.

    Notes
    -----
    ``pdf`` and ``logpdf`` use the pair-product formula. ``rosenblatt`` and
    ``inverse_rosenblatt`` preserve the original variable columns; column
    ``order[k]`` conditions on ``order[:k]``. Conditional inverses use the
    pair's analytic inverse when available, otherwise numerical inversion.
    Probabilities in density and inverse calculations are clipped at 1e-12
    to avoid floating point endpoints. Density boundary values use this
    same interior convention.
    The general CDF is a numerical Sobol integral, not a closed form.
    """

    _cheap_cdf = False

    def __init__(self, structure, pair_copulas=None):
        if not isinstance(structure, VineStructure):
            raise TypeError("structure must be a VineStructure.")
        # Revalidate, and detach from a caller's mutable structure object.
        self.structure = VineStructure(structure.trees, structure.order)
        super().__init__(structure.dim)
        self.order = self.structure.order
        if pair_copulas is None:
            pair_copulas = [[BivIndependenceCopula() for _ in t] for t in structure.trees]
        pairs = tuple(tuple(deepcopy(c) for c in t) for t in pair_copulas)
        if len(pairs) != self.dim - 1 or any(
            len(p) != len(t) for p, t in zip(pairs, structure.trees)
        ):
            raise ValueError("pair_copulas must match the tree/edge layout of structure.")
        self.pair_copulas = pairs
        self._backends = tuple(tuple(pair_backend(c) for c in t) for t in pairs)
        self._recipes = {}
        for t, tree in enumerate(structure.trees):
            for j, e in enumerate(tree):
                D = frozenset(e.conditioning)
                self._recipes[e.a, D | {e.b}] = (t, j, e.b, D, 2)
                self._recipes[e.b, D | {e.a}] = (t, j, e.a, D, 1)
        self.fit_result_ = None

    @property
    def is_absolutely_continuous(self):
        return True

    def _initial(self, U):
        return {(i, frozenset()): U[:, i] for i in range(self.dim)}

    def _conditional(self, a, D, cache):
        key = (a, D)
        if key not in cache:
            t, j, b, base, side = self._recipes[key]
            x = self._conditional(a, base, cache)
            y = self._conditional(b, base, cache)
            be = self._backends[t][j]
            cache[key] = interior(be.h2(x, y) if side == 2 else be.h1(y, x))
        return cache[key]

    def _inverse(self, a, D, p, cache):
        if not D:
            return interior(p)
        t, j, b, base, side = self._recipes[a, D]
        y = self._conditional(b, base, cache)
        be = self._backends[t][j]
        x = be.h2_inv(y, p) if side == 2 else be.h1_inv(y, p)
        return self._inverse(a, base, interior(x), cache)

    def _logpdf(self, U):
        cache = self._initial(interior(U))
        out = np.zeros(len(U))
        for t, tree in enumerate(self.structure.trees):
            for j, e in enumerate(tree):
                D = frozenset(e.conditioning)
                x = self._conditional(e.a, D, cache)
                y = self._conditional(e.b, D, cache)
                lp = self._backends[t][j].logpdf(x, y)
                if np.any(np.isnan(lp)) or np.any(np.isposinf(lp)):
                    raise FloatingPointError("Non-finite pair log-density in the vine interior.")
                out += lp
        return out

    def _transform_input(self, u):
        U, shape, scalar = parse_points((u,), self.dim, "transform")
        if not np.all(np.isfinite(U)) or np.any((U <= 0) | (U >= 1)):
            raise ValueError("Transforms require finite observations in the open unit cube.")
        return U, (*shape, self.dim)

    def rosenblatt(self, u):
        """Transform observations to independent uniforms (original columns)."""
        U, shape = self._transform_input(u)
        cache, out = self._initial(U), np.empty_like(U)
        for k, a in enumerate(self.order):
            out[:, a] = self._conditional(a, frozenset(self.order[:k]), cache)
        return out.reshape(shape)

    def inverse_rosenblatt(self, u):
        """Transform independent uniforms to this vine (original columns)."""
        U, shape = self._transform_input(u)
        cache, out = {}, np.empty_like(U)
        for k, a in enumerate(self.order):
            out[:, a] = self._inverse(a, frozenset(self.order[:k]), U[:, a], cache)
            cache[a, frozenset()] = out[:, a]
        return out.reshape(shape)

    def _rvs(self, n, rng):
        return self.inverse_rosenblatt(interior(rng.random((n, self.dim))))

    def rvs(self, n=1, random_state=None):
        """Draw ``n`` observations by inverse Rosenblatt transformation."""
        return super().rvs(integer(n, "n"), random_state)

    def loglik(self, data):
        """Sum of log-densities for observations already on the copula scale."""
        U, _ = self._transform_input(data)
        return float(self._logpdf(U).sum())

    def _cdf_integral(self, U, n_points, random_state):
        if self.dim == 2:
            edge = self.structure.trees[0][0]
            return self._backends[0][0].cdf(U[:, edge.a], U[:, edge.b])
        # Sequential conditioning: C(u) = E[prod_k F(u_k | X_<k)],
        # where X_k is sampled from its conditional law truncated at u_k.
        # This is the probability-integral change of variables on the box.
        m = int(np.ceil(np.log2(n_points)))
        W = qmc.Sobol(self.dim, scramble=True, seed=random_state).random_base2(m)
        W = interior(W)
        out = np.empty(len(U))
        for row, bound in enumerate(U):
            if np.any(bound <= 0):
                out[row] = 0.0
                continue
            active = np.flatnonzero(bound < 1)
            if len(active) <= 1:
                out[row] = np.prod(bound)
                continue
            cache, weight = {}, np.ones(len(W))
            for k, a in enumerate(self.order):
                D = frozenset(self.order[:k])
                if bound[a] >= 1:
                    p = np.ones(len(W))
                else:
                    trial = dict(cache)
                    trial[a, frozenset()] = np.full(len(W), bound[a])
                    p = self._conditional(a, D, trial)
                weight *= p
                if k < self.dim - 1:
                    cache[a, frozenset()] = self._inverse(a, D, interior(W[:, a] * p), cache)
            out[row] = weight.mean()
        return out

    def _cdf(self, U):
        return self._cdf_integral(U, 4096, 0)

    def cdf(self, *args, n_points=4096, random_state=0):
        """Numerical CDF via scrambled Sobol conditional integration.

        ``n_points`` is rounded up to a power of two. Increase it and
        compare different seeds to assess integration error. No exactness
        or deterministic error bound is claimed. Grounding and univariate
        margins are evaluated exactly. In dimension two, the pair's own
        CDF is used. Inputs outside the cube are clipped.
        """
        U, shape, scalar = parse_points(args, self.dim, "cdf")
        if not np.all(np.isfinite(U)):
            raise ValueError("CDF inputs must be finite.")
        n_points = integer(n_points, "n_points", 1)
        out = self._cdf_integral(np.clip(U, 0, 1), n_points, random_state)
        return finish(out, shape, scalar)

    __call__ = cdf

    @classmethod
    def fit(cls, data, **kwargs):
        """Fit a vine; return the model with diagnostics in ``fit_result_``.

        See :func:`copul.vine.fit_vine` for selection and fitting options.
        """
        from copul.vine.fitting import fit_vine

        if cls is CVineCopula:
            if kwargs.pop("vine_type", "cvine") != "cvine":
                raise ValueError("CVineCopula.fit requires vine_type='cvine'.")
            kwargs["vine_type"] = "cvine"
        elif cls is DVineCopula:
            if kwargs.pop("vine_type", "dvine") != "dvine":
                raise ValueError("DVineCopula.fit requires vine_type='dvine'.")
            kwargs["vine_type"] = "dvine"
        return fit_vine(data, **kwargs).copula


class CVineCopula(VineCopula):
    """Canonical vine; roots follow ``order`` (default ``range(dim)``).

    Tree ``t`` has pairs ``(order[t], order[j] | order[:t])``,
    ``j = t+1, ..., d-1`` in that order.
    """

    def __init__(self, dim, pair_copulas=None, order=None):
        dim = integer(dim, "dim", 2)
        super().__init__(VineStructure.cvine(range(dim) if order is None else order), pair_copulas)
        if self.dim != dim:
            raise ValueError("order must have length dim.")


class DVineCopula(VineCopula):
    """Drawable vine whose first tree is the path ``order``.

    Tree ``k-1`` has pairs ``(order[i], order[i+k] | order[i+1:i+k])``
    for ``i = 0, ..., d-k-1`` in that order.
    """

    def __init__(self, dim, pair_copulas=None, order=None):
        dim = integer(dim, "dim", 2)
        super().__init__(VineStructure.dvine(range(dim) if order is None else order), pair_copulas)
        if self.dim != dim:
            raise ValueError("order must have length dim.")


RVineCopula = VineCopula
