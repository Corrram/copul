"""Regular-vine trees, with explicit conditioned pairs and conditioning sets.

An edge ``(a, b | D)`` joins two nodes whose complete sets are ``D + {a}``
and ``D + {b}``. Above the first tree these nodes must be edges sharing a
node in the preceding tree (the proximity condition). See Dißmann et al.
(2013), Computational Statistics & Data Analysis 59, 52--69,
https://doi.org/10.1016/j.csda.2012.08.010.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def permutation(order, dim):
    order = tuple(integer(i, "variable index") for i in order)
    if len(order) != dim or set(order) != set(range(dim)):
        raise ValueError(f"order must be a permutation of range({dim}).")
    return order


@dataclass(frozen=True)
class VineEdge:
    """An oriented pair ``(a, b | conditioning)``; indices are zero based.

    The first argument of the associated bivariate copula corresponds to
    ``a``. The ordering of the conditioning set has no significance.
    """

    a: int
    b: int
    conditioning: tuple[int, ...] = ()

    def __post_init__(self):
        a, b = integer(self.a, "a"), integer(self.b, "b")
        cond = tuple(integer(i, "conditioning index") for i in self.conditioning)
        if a == b or a in cond or b in cond or len(set(cond)) != len(cond):
            raise ValueError("An edge needs distinct conditioned and conditioning indices.")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "conditioning", tuple(sorted(cond)))

    @property
    def complete(self):
        """All variables in this edge, including its conditioning set."""
        return frozenset((self.a, self.b, *self.conditioning))


class VineStructure:
    """Validated regular vine specified by ``d - 1`` lists of edges.

    Parameters
    ----------
    trees : sequence of sequences of VineEdge or (a, b, conditioning)
        Tree ``t`` (zero based) has ``d - t - 1`` edges with ``t``
        conditioning variables. All trees must be connected and acyclic.
    order : sequence of int, optional
        An admissible *forward sampling order*. If omitted, one is derived
        from the conditional distributions encoded by the trees.

    Notes
    -----
    Unlike matrix conventions, this representation preserves pair
    orientation explicitly. Both the graph and proximity conditions are
    checked, not merely the sizes of intersections of variable sets.
    """

    def __init__(self, trees, order=None):
        self.trees = tuple(
            tuple(e if isinstance(e, VineEdge) else VineEdge(*e) for e in tree) for tree in trees
        )
        self.dim = len(self.trees) + 1
        if self.dim < 2:
            raise ValueError("A vine requires at least two variables.")
        nodes = [frozenset((i,)) for i in range(self.dim)]
        previous_links = None
        keys = {(i, frozenset()) for i in range(self.dim)}
        for t, tree in enumerate(self.trees):
            if len(tree) != self.dim - t - 1:
                raise ValueError(f"Tree {t + 1} must have {self.dim - t - 1} edges.")
            parents = list(range(len(nodes)))

            def root(i, parents=parents):
                while parents[i] != i:
                    i = parents[i]
                return i

            links = []
            for e in tree:
                D = frozenset(e.conditioning)
                if len(D) != t or max(e.complete) >= self.dim:
                    raise ValueError(f"Invalid conditioning set or variable in tree {t + 1}.")
                try:
                    i, j = nodes.index(D | {e.a}), nodes.index(D | {e.b})
                except ValueError:
                    raise ValueError(f"Edge {e} has no parents in the preceding tree.") from None
                if previous_links is not None and not (previous_links[i] & previous_links[j]):
                    raise ValueError(f"Edge {e} violates the proximity condition.")
                ri, rj = root(i), root(j)
                if ri == rj:
                    raise ValueError(f"Tree {t + 1} contains a cycle or duplicate edge.")
                parents[ri] = rj
                links.append(frozenset((i, j)))
                for a, b in ((e.a, e.b), (e.b, e.a)):
                    if (a, D) not in keys:
                        raise ValueError(f"Conditional distribution for edge {e} is unavailable.")
                    keys.add((a, D | {b}))
            nodes = [e.complete for e in tree]
            previous_links = links
        if order is None:
            remaining = frozenset(range(self.dim))
            reverse = []
            while remaining:
                choices = [i for i in sorted(remaining) if (i, remaining - {i}) in keys]
                if not choices:
                    raise ValueError("The trees do not admit a Rosenblatt sampling order.")
                i = choices[-1]
                reverse.append(i)
                remaining -= {i}
            order = reverse[::-1]
        self.order = permutation(order, self.dim)
        for k, i in enumerate(self.order):
            if (i, frozenset(self.order[:k])) not in keys:
                raise ValueError("order is not an admissible sampling order for these trees.")

    @classmethod
    def cvine(cls, order):
        """C-vine: ``order[t]`` is the root of tree ``t + 1``."""
        order = tuple(order)
        permutation(order, len(order))
        return cls(
            [
                [VineEdge(order[t], b, order[:t]) for b in order[t + 1 :]]
                for t in range(len(order) - 1)
            ],
            order=order,
        )

    @classmethod
    def dvine(cls, order):
        """D-vine: ``order`` is the path through the first tree."""
        order = tuple(order)
        permutation(order, len(order))
        return cls(
            [
                [
                    VineEdge(order[i], order[i + k], order[i + 1 : i + k])
                    for i in range(len(order) - k)
                ]
                for k in range(1, len(order))
            ],
            order=order,
        )

    def __repr__(self):
        return f"VineStructure(dim={self.dim}, order={self.order})"
