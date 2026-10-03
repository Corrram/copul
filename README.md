# copul

[![PyPI](https://img.shields.io/pypi/v/copul.svg)](https://pypi.org/project/copul/)
[![Python](https://img.shields.io/pypi/pyversions/copul.svg)](https://pypi.org/project/copul/)
[![Docs](https://readthedocs.org/projects/copul/badge/?version=latest)](https://copul.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**copul** is a Python toolkit for research on bivariate copulas and dependence
measures. It combines

- **symbolic copula families** (SymPy): closed-form cdfs, densities,
  conditional distributions and dependence measures, with parameters left free
  or fixed;
- a **numerical measures engine** for Chatterjee's $\xi$, Spearman's $\rho$,
  Kendall's $\tau$, Blest's $\nu$, Spearman's footrule, Gini's $\gamma$,
  Blomqvist's $\beta$, distances to independence and tail coefficients, with
  one uniform `method=` switch between closed forms, quadrature, SymPy and
  Monte Carlo;
- a **uniform, vectorised numerical API** (`cdf`, `pdf`, `logpdf`,
  conditional distributions and their inverses, fast exact `rvs`) on every
  copula;
- **constructions** (rotations, reflections, survival copulas, mixtures,
  Khoudraji asymmetrisation, ordinal sums, gluing) and Joe's two-parameter
  **BB families**;
- **statistics**: pseudo-observations, the empirical copula, estimators with
  confidence intervals for every measure, independence tests, maximum
  likelihood and moment fitting, model selection and goodness-of-fit tests
  (`copul.stats`);
- a **copula theory toolkit** (`copul.theory`, `copul.sklar`,
  `copul.multivariate`): Sklar's theorem, Archimedean and extreme-value
  theory, dependence concepts and orderings, Markov operators and distances,
  bounds, quasi-copulas, diagonals, symmetry and $d$-dimensional copulas;
- **checkerboard, shuffle-of-min and Bernstein copulas** with exact,
  vectorised formulas for all measures;
- a **research toolkit**: a registry of exact regions between measures
  (`copul.regions`), exact LP/QP optimisation over checkerboard copulas
  (`copul.optim`) and a counterexample search for conjectured inequalities
  (`copul.search`).

Documentation: <https://copul.readthedocs.io>

## Installation

```bash
pip install copul              # core package
pip install "copul[optim]"     # + cvxpy for copul.optim
```

copul requires Python 3.10 or newer.

## Quickstart

### Copula families: cdf, density, sampling

```python
import copul as cp

clayton = cp.Clayton(theta=2)
clayton.cdf(0.3, 0.7)              # 0.28686... (a float)
clayton.pdf(0.3, 0.7)              # 0.62928...
clayton.cond_distr_1(0.3, 0.7)     # P(V <= 0.7 | U = 0.3)
samples = clayton.rvs(100_000, random_state=0)  # exact, vectorised
assert samples.shape == (100_000, 2)
```

Every bivariate copula has the same vectorised numerical API: `cdf`, `pdf`,
`logpdf`, `cond_distr_1`, `cond_distr_2`, their inverses
`cond_distr_1_inv(u, w)` / `cond_distr_2_inv(v, w)` (conditional quantiles)
and `survival_function` accept scalars, broadcastable arrays, an `(N, 2)`
array or `u=`, `v=` keywords. Called without arguments they return the
symbolic expression.

```python
import numpy as np
import copul as cp

P = np.random.default_rng(0).random((5, 2))
gumbel = cp.GumbelHougaard(theta=2)
gumbel.cdf(P)                          # ndarray of shape (5,)
gumbel.logpdf(P[:, 0], P[:, 1])        # numerically stable log-density
v = gumbel.cond_distr_1_inv(P[:, 0], 0.5)  # conditional medians of V | U
np.allclose(gumbel.cond_distr_1(P[:, 0], v), 0.5)  # True
cp.Clayton().cdf(P, theta=2)           # free parameters as keywords
cp.Clayton().cdf()                     # theta-dependent SymPy expression
```

### Dependence measures

Every measure is a method of every bivariate copula. For a fully specified
copula it returns a `float`; the `method=` keyword chooses how it is computed.

```python
import copul as cp

clayton = cp.Clayton(theta=2)
clayton.kendalls_tau()                    # 0.5 (closed form)
clayton.spearmans_rho()                   # 0.68223... (numerical quadrature)
clayton.chatterjees_xi()                  # 0.33429...
clayton.spearmans_rho(method="numeric")   # force adaptive quadrature
clayton.spearmans_rho(method="mc", n_samples=10_000, random_state=0)  # Monte Carlo

# several measures at once, by canonical key or alias
cp.compute_measures(clayton, ["xi", "rho", "tau", "nu", "beta"])
```

`method` is one of `"auto"` (default: closed form if available, otherwise the
numerical engine), `"closed"`, `"numeric"`, `"symbolic"` or `"mc"`.

### Symbolic derivations with free parameters

Leave parameters unspecified and the same methods return SymPy expressions:

```python
import copul as cp

fgm = cp.FarlieGumbelMorgenstern()
fgm.cdf()               # theta*u*v*(1 - u)*(1 - v) + u*v
fgm.spearmans_rho()     # theta/3
fgm.kendalls_tau()      # 2*theta/9
fgm.chatterjees_xi()    # theta**2/15
cp.Clayton().kendalls_tau()  # theta/(theta + 2)
```

### Parameter sweeps and calibration

```python
import copul as cp

curve = cp.Frank().measure_curve(["tau", "rho", "xi"], n=15)  # sweep theta
curve["rho"]            # ndarray of Spearman's rho along the sweep
frank = cp.Frank.from_measure("tau", 0.5)   # Frank copula with tau = 0.5
round(frank.kendalls_tau(), 10)             # 0.5
```

`curve.plot()` draws all measures against the parameter.

## Constructions and further families

```python
import copul as cp

rot = cp.rotate(cp.Clayton(2), 90)            # negative dependence
round(rot.kendalls_tau(), 10)                  # -0.5
surv = cp.survival(cp.Clayton(2))              # upper tail dependence
mix = cp.mixture([cp.Clayton(2), cp.UpperFrechet()], [0.7, 0.3])
asym = cp.khoudraji(cp.BivIndependenceCopula(), cp.GumbelHougaard(3), 0.3, 0.9)
osum = cp.ordinal_sum([(0.0, 0.6, cp.GumbelHougaard(3)), (0.6, 1.0, cp.Clayton(2))])
bb1 = cp.BB1(theta=0.7, delta=1.4)             # Joe's BB1 (Clayton-Gumbel)
round(bb1.kendalls_tau(), 10)                  # 1 - 2 / (delta (theta + 2))
samples = asym.rvs(1000, random_state=0)
```

Constructions return ordinary copula objects: all measures, sampling and
plotting work on them, with exact measure relations where they are classical
(e.g. sign changes under rotations, linearity of $\rho$, $\beta$, $\nu$,
$\gamma$ and the tail coefficients in mixtures). Available BB families:
`BB1`, `BB2`, `BB3`, `BB6`, `BB7`, `BB8`, `BB9`, `BB10` (Joe, 1997/2014), all
with exact frailty samplers.

## Statistics: estimation, fitting and goodness of fit

```python
import copul as cp

X = cp.Clayton(theta=2).rvs(1000, random_state=0)      # or your own data (n x 2)
U = cp.pseudo_obs(X)                                    # rank transform
table = cp.estimate(X, ["tau", "rho", "xi"], ci="asymptotic")   # DataFrame
emp = cp.EmpiricalCopula(X)
emp.cdf(0.5, 0.5)                                       # empirical copula
res = cp.fit(cp.Clayton, X)                             # maximum likelihood
res.copula.kendalls_tau()                               # fitted copula
res_tau = cp.fit(cp.Frank, X, method="itau")            # inversion of tau
ranking = cp.select(X, ["Clayton", "Frank", "Gaussian"])  # AIC ranking
```

`cp.gof_test(res, X, n_boot=200)` runs the Cramér–von Mises test of Genest,
Rémillard and Beaudoin (2009) with a parametric bootstrap, and
`cp.stats.independence_test(X, method="xi")` tests independence with
Kendall's $\tau$, Spearman's $\rho$, Chatterjee's $\xi$, Hoeffding's $D$ or
a Cramér–von Mises statistic.

## Copula theory

`copul.theory`, `copul.sklar` and `copul.multivariate` implement standard copula
theory (Nelsen 2006; Durante & Sempi 2016; Joe 2014) on every copula object:
Sklar's theorem with `scipy.stats` margins, Kendall distribution functions,
generator checks and $d$-monotonicity, associativity, max-stability,
extreme-value attractors and tail copulas, dependence concepts (PQD, LTD, RTI,
SI, TP2, ...) with exact family characterizations, the concordance order,
Markov products and operators, complete dependence, distances between copulas
(sup, $L^p$, Trutschnig's $\partial$-metrics, $\zeta_1$), best-possible bounds
given a value, a diagonal or a value of $\tau$, $\rho$ or $\beta$,
quasi-copulas, Bertino and diagonal copulas, non-exchangeability and radial
asymmetry with tests, and $d$-dimensional Gaussian, Student-t and Archimedean
copulas with multivariate $\rho$, $\tau$ and $\beta$.

```python
from scipy import stats
import copul as cp
from copul import theory as th

H = cp.JointDistribution(cp.Clayton(2), [stats.norm(), stats.expon()])  # Sklar
H.cdf([0.0, 1.0])                              # H(x, y) = C(F(x), G(y))
H.regression(0.5)                              # E[Y | X = 0.5]

C = cp.Clayton(2)
C.kendall_distribution(0.3)                    # K_C(t) = t + t (1 - t^theta) / theta
th.max_dimension(cp.Clayton(-0.3))             # 4: valid up to d = 4
th.is_max_stable(cp.GumbelHougaard(2))         # True
profile = cp.Frank(3).dependence_profile()     # PQD, LTD, RTI, SI, TP2, ...
C.distance(cp.Frank(5), metric="D1")           # Trutschnig's D1 metric
bounds = th.bounds_given_measure("tau", 0.5)   # pointwise best-possible bounds
bounds.upper.cdf(0.3, 0.6)                     # 0.3
cp.khoudraji(cp.BivIndependenceCopula(), cp.GumbelHougaard(3), 0.3, 0.9).nonexchangeability()
C3 = cp.ClaytonND(2.0, dim=3)                  # 3-dimensional Clayton copula
cp.multivariate.kendalls_tau_nd(C3)            # 0.5
```

## Checkerboard copulas

Checkerboard copulas are given by a (not necessarily normalised) mass matrix.
`BivCheckPi` spreads the mass of each cell uniformly, `BivCheckMin` and
`BivCheckW` place it on the (anti-)diagonal of the cell. All measures are
evaluated with exact formulas.

```python
import copul as cp

P = [[1, 0, 0],
     [0, 0, 1],
     [0, 1, 0]]
ccop = cp.BivCheckPi(P)
ccop.spearmans_rho(), ccop.chatterjees_xi()    # (0.4444..., 0.6666...)
cp.BivCheckMin(P).chatterjees_xi()             # 1.0 (a shuffle of M)

# checkerboard approximation of a family and of data
approx = cp.Clayton(theta=2).to_checkerboard(20)
data = cp.Gaussian(rho=0.6).rvs(2000, random_state=1)
empirical = cp.from_data(data, checkerboard_size=10)
empirical.spearmans_rho()

# sample estimator of Chatterjee's xi
cp.xi_ncalculate(data[:, 0], data[:, 1])
```

Shape properties are available as predicates, e.g. `ccop.is_cis()` (a `bool`)
and `ccop.cis_direction()` (directions in which the copula is CIS).

## Research toolkit

### Exact regions between measures

`copul.regions` is a registry of proven exact regions
$\{(\kappa_1(C), \kappa_2(C)) : C \text{ copula}\}$ with closed-form
boundaries, key points, boundary-attaining copula families and references.

```python
import copul as cp

region = cp.get_region("xi", "rho")       # same as copul.regions.get
float(region.upper(0.3))                   # 0.7: max rho given xi = 0.3
bool(region.contains(0.3, 0.5))            # True
C = region.boundary_copula(0.3, side="upper")  # a copula on the boundary
ax = region.plot()                         # matplotlib axes
cp.regions.available()                     # registered (x, y) pairs
```

### Boundary optimisation over checkerboard copulas

On a checkerboard copula with mass matrix $P$, the measures $\rho$, $\nu$,
$\psi$, $\gamma$ and $\beta$ are affine in $P$ and $\xi$ is a convex quadratic.
`copul.optim` turns questions about extremal copulas into exact LPs/QPs
(requires `pip install "copul[optim]"`); every optimum is an honest copula, so
traced boundaries are inner approximations of the exact region.

```python
import copul.optim as co

problem = co.CheckerboardProblem(n=16)
result = problem.maximize("rho", subject_to={"xi": ("<=", 0.3)})
result["rho"], result["xi"]      # (0.698..., 0.2999...) vs. exact bound 0.7
result.copula                    # optimal BivCheckPi

trace = co.trace_boundary("xi", "rho", side="upper", n=16, n_points=6)
trace.points                     # (xi, rho) points on the inner boundary
```

### Counterexample search

```python
import copul as cp

# Is xi <= |rho| for all copulas? No:
ce = cp.find_counterexample("xi", lambda C: abs(C.spearmans_rho()), n_iter=500, rng=0)
ce.copula, ce.lhs, ce.rhs

# The Ansari-Rockel bound |rho| <= M(xi) holds on random SI checkerboards
region = cp.get_region("xi", "rho")
report = cp.check_inequality(
    lambda C: abs(C.spearmans_rho()) - float(region.upper(C.chatterjees_xi())),
    0.0, "<=", n_iter=100, condition="si", rng=0,
)
report.holds, report.max_violation

# random (SI) checkerboard copulas for your own experiments
for C in cp.random_checkerboards(3, grid=4, condition="si", rng=0):
    assert C.is_cis()
```

## Dependence measures

Keys are accepted by `copul.compute_measures`, `measure_curve`,
`from_measure`, `copul.regions` and `copul.optim` (aliases such as
`"spearman"` or `"psi"` also work; see `copul.measures.list_measures()`).
$M$, $W$, $\Pi$ denote the upper and lower Fréchet bounds and independence.

| key | measure | method | definition | $M$ | $W$ | $\Pi$ |
|---|---|---|---|---|---|---|
| `xi` | Chatterjee's $\xi$ | `chatterjees_xi` | $6\int\int (\partial_1 C)^2\,du\,dv - 2$ | 1 | 1 | 0 |
| `rho` | Spearman's $\rho$ | `spearmans_rho` | $12\int\int C\,du\,dv - 3$ | 1 | −1 | 0 |
| `tau` | Kendall's $\tau$ | `kendalls_tau` | $1 - 4\int\int \partial_1 C\,\partial_2 C\,du\,dv$ | 1 | −1 | 0 |
| `nu` | Blest's $\nu$ | `blests_nu` | $24\int\int (1-u)\,C\,du\,dv - 2$ | 1 | −1 | 0 |
| `footrule` | Spearman's footrule $\psi$ | `spearmans_footrule` | $6\int_0^1 C(t,t)\,dt - 2$ | 1 | −½ | 0 |
| `gamma` | Gini's $\gamma$ | `ginis_gamma` | $4\int_0^1 [C(t,t) + C(t,1-t)]\,dt - 2$ | 1 | −1 | 0 |
| `beta` | Blomqvist's $\beta$ | `blomqvists_beta` | $4\,C(\tfrac12,\tfrac12) - 1$ | 1 | −1 | 0 |
| `hoeffdings_d` | Hoeffding's $\Phi^2$ | `hoeffdings_d` | $90\int\int (C - uv)^2\,du\,dv$ | 1 | 1 | 0 |
| `sigma` | Schweizer–Wolff $\sigma$ | `schweizer_wolff_sigma` | $12\int\int \lvert C - uv\rvert\,du\,dv$ | 1 | 1 | 0 |
| `kappa` | uniform distance $\kappa$ | `uniform_distance` | $4\sup\lvert C(u,v) - uv\rvert$ | 1 | 1 | 0 |
| `lp` | $L^p$ distance $\delta_p$ | `lp_distance` | $k(p)\int\int \lvert C - uv\rvert^p\,du\,dv$ | 1 | 1 | 0 |
| `bkr` | Blum–Kiefer–Rosenblatt $B$ | `blum_kiefer_rosenblatt` | $30\int (C - uv)^2\,dC$ | 1 | 1 | 0 |
| `mutual_information` | mutual information | `mutual_information` | $\int\int c\log c\,du\,dv$ | ∞ | ∞ | 0 |
| `lambda_l` | lower tail dependence | `lambda_L` | $\lim_{t\to 0^+} C(t,t)/t$ | 1 | 0 | 0 |
| `lambda_u` | upper tail dependence | `lambda_U` | $\lim_{t\to 1^-} (1 - 2t + C(t,t))/(1-t)$ | 1 | 0 | 0 |

`xi_2` is Chatterjee's $\xi$ conditioning on the second variable
(`chatterjees_xi(condition_on_y=True)`).

## Copula families

| category | classes |
|---|---|
| Archimedean | `Clayton` (`Nelsen1`), `Nelsen2`, `AliMikhailHaq` (`Nelsen3`), `GumbelHougaard` (`Nelsen4`), `Frank` (`Nelsen5`), `Joe` (`Nelsen6`), `Nelsen7`, `Nelsen8`, `GumbelBarnett` (`Nelsen9`), `Nelsen10`–`Nelsen14`, `GenestGhoudi` (`Nelsen15`), `Nelsen16`–`Nelsen22`; custom ones via `cp.from_generator` |
| Two-parameter Archimedean (Joe) | `BB1`, `BB2`, `BB3`, `BB6`, `BB7`, `BB8`, `BB9`, `BB10` |
| Extreme value | `BB5`, `CuadrasAuge`, `Galambos`, `GumbelHougaardEV`, `HueslerReiss`, `JoeEV`, `MarshallOlkin`, `Tawn`, `tEV`; custom ones via `cp.from_pickands` |
| Elliptical | `Gaussian`, `StudentT`, `Laplace` |
| Other families | `FarlieGumbelMorgenstern`, `Frechet`, `Mardia`, `Plackett`, `Raftery`, `B11`, `DiagonalBandCopula` |
| Special families from exact-region research | `XiRhoBoundaryCopula` ($\xi$–$\rho$), `XiNuBoundaryCopula` ($\xi$–$\nu$), `VThresholdCopula` ($\rho$–$\nu$), `XiPsiApproxLowerBoundaryCopula`, `XiBetaBoundaryCopula`, `MedianSwapCopula`, `EndSwapCopula` |
| Special copulas | `UpperFrechet` ($M$), `LowerFrechet` ($W$), `BivIndependenceCopula` / `IndependenceCopula` ($\Pi$) |
| Approximations | `BivCheckPi`, `BivCheckMin`, `BivCheckW`, `BivCheckMixed`, `BivBlockDiagMixed`, `CheckPi`, `CheckMin` ($d$-dimensional), `ShuffleOfMin`, `BivBernstein`, `Bernstein` |

New copulas arise from `cp.rotate`, `cp.reflect`, `cp.transpose`,
`cp.survival`, `cp.mixture`, `cp.khoudraji`, `cp.ordinal_sum`, `cp.gluing` and
`cp.markov_product`. Copulas can also be built from a cdf, density or conditional distribution
(`cp.from_cdf`, `cp.from_pdf`, `cp.from_cond_distr_1`, `cp.from_cond_distr_2`)
or from a mass matrix (`cp.from_matrix`). `cp.Families` lists everything
programmatically.

## Development

```bash
pip install -e ".[dev]"
make test        # fast test suite
make lint        # ruff
make docs        # Sphinx documentation
```

## Citation

If you use copul in your research, please cite it:

```bibtex
@software{copul,
  author  = {Rockel, Marcus},
  title   = {copul: copulas and dependence measures in Python},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/Corrram/copul},
  note    = {TODO: replace with the reference of the copul paper once available}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
