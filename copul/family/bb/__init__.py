r"""
Two-parameter Archimedean BB families of Joe (1997, 2014).

All families are Laplace-transform (frailty) Archimedean copulas, see
:class:`~copul.family.bb.lt_archimedean.LTArchimedeanCopula`: they have
SymPy cdfs (free parameters stay symbolic), stable vectorised numerics,
exact Marshall–Olkin sampling and closed-form tail coefficients.

=========  ======================  =============================  ============================
family     parameters              frailty                        special cases
=========  ======================  =============================  ============================
``BB1``    θ>0, δ≥1                gamma ∘ positive stable        δ=1 Clayton(θ); θ→0 GH(δ)
``BB2``    θ>0, δ>0                gamma ∘ gamma                  δ→0 Clayton(θ)
``BB3``    θ≥1, δ>0                positive stable ∘ gamma        θ=1 Clayton(δ)
``BB6``    θ≥1, δ≥1                Sibuya ∘ positive stable       θ=1 GH(δ); δ=1 Joe(θ)
``BB7``    θ≥1, δ>0                Sibuya ∘ gamma                 θ=1 Clayton(δ); δ→0 Joe(θ)
``BB8``    θ≥1, 0<δ≤1              tilted Sibuya                  δ=1 Joe(θ); θ=1 Π
``BB9``    θ≥1, δ>0                tilted positive stable         θ=1 Π; δ→∞ GH(θ)
``BB10``   θ>0, 0≤π<1              shifted negative binomial      π=0 Π; θ=1 AMH(π)
=========  ======================  =============================  ============================

(``BB4`` and ``BB5`` are Archimax/extreme-value families; ``BB5`` is
:class:`copul.family.extreme_value.BB5`.)

Examples
--------
>>> from copul.family.bb import BB1
>>> C = BB1(theta=2, delta=1.5)
>>> round(C.kendalls_tau(), 12)       # 1 - 2 / (delta (theta + 2))
0.666666666667
>>> X = C.rvs(1000, random_state=0)
>>> X.shape
(1000, 2)

References
----------
Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall.

Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press, Sec. 4.17.
"""

from copul.family.bb.bb1 import BB1
from copul.family.bb.bb2 import BB2
from copul.family.bb.bb3 import BB3
from copul.family.bb.bb6 import BB6
from copul.family.bb.bb7 import BB7
from copul.family.bb.bb8 import BB8
from copul.family.bb.bb9 import BB9
from copul.family.bb.bb10 import BB10
from copul.family.bb.lt_archimedean import LTArchimedeanCopula

__all__ = ["BB1", "BB2", "BB3", "BB6", "BB7", "BB8", "BB9", "BB10", "LTArchimedeanCopula"]
