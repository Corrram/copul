# Changelog

All notable changes to `copul` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/) (pre-1.0: minor versions may break
the API).

## [0.4.0] - unreleased

A large release: a numerical measures engine behind every measure method,
three research subpackages (`copul.regions`, `copul.optim`, `copul.search`),
an exact engine for bivariate checkerboard copulas and many correctness
fixes. See the [migration guide](#migration-guide-03x--040) below.

### Added

- **`copul.measures`**: registry of dependence measures (canonical keys
  `xi`, `xi_2`, `rho`, `tau`, `footrule`, `gamma`, `beta`, `nu`,
  `hoeffdings_d`, `sigma`, `kappa`, `lp`, `bkr`, `mutual_information`,
  `lambda_l`, `lambda_u`, with aliases, formulas, ranges and values at
  $M$, $W$, $\Pi$); vectorised adaptive Gauss–Kronrod quadrature on $[0,1]$ and
  $[0,1]^2$; `compute(copula, keys, method=...)`; `measure_curve` and
  `from_measure` (also available as copula methods, e.g.
  `Clayton.from_measure("tau", 0.5)`).
- **`method=` keyword on every measure method** of bivariate copulas:
  `"auto"` (default), `"closed"`, `"numeric"`, `"symbolic"`, `"mc"`.
- Fast one-dimensional formulas for Kendall's $\tau$ of Archimedean copulas
  and for $\rho$, $\tau$ of extreme-value copulas (valid for kinked Pickands
  functions); tail coefficients from $h$-functions with Aitken extrapolation.
- New measure methods `uniform_distance` ($\kappa$), `measure(key)` and
  `measures(keys)`; `lp_distance`, `blum_kiefer_rosenblatt` and
  `mutual_information` are evaluated by the numerical engine when no closed
  form is known.
- **`copul.regions`**: registry of validated exact regions
  (`(xi, rho)`, `(xi, nu)`, `(rho, nu)`, `(xi, footrule)` for SI copulas)
  with boundaries, key points, boundary copulas, references and plotting
  (`copul.get_region("xi", "rho").plot()`).
- **`copul.optim`**: exact LP/QP optimisation over checkerboard copulas
  (`CheckerboardProblem`, `trace_boundary`, exact shape constraints such as
  SI/SD/LTD/PQD/symmetry, penalty CCP for Kendall's $\tau$). `cvxpy` is an
  optional dependency: `pip install "copul[optim]"`.
- **`copul.search`**: `random_checkerboards`, `find_counterexample`,
  `check_inequality` for stress-testing conjectured inequalities.
- Exact vectorised engine for `BivCheckPi`, `BivCheckMin`, `BivCheckW`,
  `BivCheckMixed`: $O(1)$ cdf per point, exact conditional distributions,
  vectorised sampling with `random_state`, exact footrule/Gini for
  non-square grids, exact SI/LTD/RTI/PQD checks.
- `markov_product` rewritten: exact closed form for two checkerboards,
  quadrature for any other pair.
- Chatterjee's $\xi_n$: tie handling (Chatterjee 2021), `xi_null_variance`,
  tie-corrected independence test, input validation.
- Top-level exports: `copul.measures`, `copul.optim`, `copul.regions`,
  `copul.search`, `compute_measures`, `get_region`, `measure_curve`,
  `from_measure`, `find_counterexample`, `check_inequality`,
  `random_checkerboards`, `from_samples`, `BivCheckMixed`,
  `BivBlockDiagMixed`, `XiBetaBoundaryCopula`, `MedianSwapCopula`,
  `VThresholdCopula`, `EndSwapCopula`, `B11`, `IndependenceCopula`;
  `copul.__version__`.
- Boundary copula families `XiBetaBoundaryCopula`, `MedianSwapCopula`,
  `VThresholdCopula`, `EndSwapCopula`.
- **Uniform numerical evaluation API** on every bivariate copula (families,
  checkerboards, Bernstein, shuffles, boundary families):
  `cdf`, `pdf`, `logpdf`, `cond_distr_1`, `cond_distr_2`,
  `cond_distr_1_inv(u, w)`, `cond_distr_2_inv(v, w)` (conditional quantiles)
  and `survival_function` accept scalars (→ `float`), broadcastable arrays,
  an `(N, 2)` array or `u=`, `v=` keywords, clip arguments to $[0,1]$ and
  never touch SymPy in the hot path (`copul.family.core.numeric_api`,
  vectorised backend in `copul.measures.backend`). Closed-form
  log-densities and conditional quantiles for Clayton, Frank,
  Gumbel–Hougaard, Joe, AMH, FGM, Plackett, Gaussian and Student-t; a
  vectorised safeguarded Newton / Illinois solver otherwise.
- **Fast exact sampling**: `rvs(n, random_state)` of bivariate copulas uses
  Marshall–Olkin frailty algorithms (Clayton: Gamma, Gumbel–Hougaard:
  positive stable, Frank: logarithmic, Joe: Sibuya, AMH: geometric;
  `copul.family.archimedean._frailty`), multivariate normal / t sampling,
  the Marshall–Olkin shock model, Fréchet-bound mixtures, and vectorised
  conditional inversion for all other copulas ($10^5$ samples in
  milliseconds to about one second).
- **`copul.stats`**: `pseudo_obs`, `EmpiricalCopula` (vectorised cdf,
  empirical checkerboard and Bernstein copulas, sample versions of every
  registry measure incl. tail-dependence and mutual-information estimators),
  `estimate` (tidy table with asymptotic or bootstrap confidence intervals),
  `independence_test` (tau, rho, xi, Hoeffding, Cramér–von Mises), `fit`
  (maximum likelihood with standard errors, inversion of tau/rho/xi/beta,
  partly fixed parameters), `select` (AIC/BIC ranking) and `gof_test`
  (Genest–Rémillard–Beaudoin parametric bootstrap). Top-level shortcuts
  `cp.pseudo_obs`, `cp.EmpiricalCopula`, `cp.estimate`, `cp.fit`,
  `cp.select`, `cp.gof_test`.
- **Constructions** (`copul.family.constructions`, also top level):
  `rotate`, `reflect`, `transpose`, `survival`, `mixture`, `khoudraji`,
  `ordinal_sum`, `gluing` with vectorised evaluation, exact sampling and
  classical closed-form measure relations.
- **BB families** `BB1`, `BB2`, `BB3`, `BB6`, `BB7`, `BB8`, `BB9`, `BB10`
  (Joe 1997, 2014) on a shared Laplace-transform Archimedean base with
  Marshall–Olkin frailty samplers, symbolic cdfs for free parameters and
  closed-form Kendall's tau / tail coefficients where known.
- Student-t copula CDF via the Dunnett–Sobel / Genz (BVTL) formula for
  integer degrees of freedom; analytic Pickands derivatives for `tEV`.
- `tests/properties`: universal property tests (margins, 2-increasingness,
  Fréchet bounds, conditional distributions and their inverses, densities,
  sampling, symbolic/numeric agreement) for all 80+ bivariate copulas.
- Packaging and tooling: `optim`, `docs` and `dev` extras; ruff and pytest
  configuration; GitHub Actions CI (lint, tests on Python 3.10–3.13, optional
  cvxpy job, build check); `CITATION.cff`; tested README examples; Sphinx
  pages for the new subpackages, the measures and research workflows.

### Changed

- **Measure methods return Python floats for fully specified copulas**
  (closed form when available, otherwise numerical quadrature). Copulas with
  free parameters still return SymPy expressions.
- Renamed `gini_gamma` → `ginis_gamma`, `spearman_footrule` →
  `spearmans_footrule`, `lp_concordance` → `lp_distance` (old names are
  deprecated aliases).
- `ShuffleOfMin` uses the standard names `kendalls_tau`, `spearmans_rho`,
  `chatterjees_xi`, `lambda_L`, `lambda_U` (+ `blests_nu`, footrule, Gini,
  Blomqvist); cdf/cond_distr accept `u=`, `v=`; `rvs(n, random_state)`.
- `is_cis()` / `is_si()` (checkerboards and `CISVerifier`) return a `bool`;
  the directions are available via `cis_direction()` → `(SI, SD)`.
  `CISVerifier` aggregates over all parameter values.
- `condition_on_y` is keyword-only everywhere.
- **Numerical evaluations return `float` / `ndarray`** instead of SymPy
  wrappers (`Clayton(2).cdf(0.3, 0.7)` is a `float`; no `.evalf()`), are
  evaluated vectorised, and clip arguments outside $[0,1]$ (densities
  vanish there). Calls without arguments or with symbolic / partial
  arguments (`C.cdf()`, `C.cdf(v=0.5)`) still return SymPy wrappers; array
  calls on copulas with free parameters raise a `ValueError` (pass the
  parameters as keywords, e.g. `Clayton().cdf(P, theta=2)`). For copulas
  with a singular component `pdf(u, v)` is the density of the absolutely
  continuous part (families without one still raise
  `PropertyUnavailableException`).
- `rvs` of the Gaussian, Student-t and `IndependenceCopula` honour
  `random_state` and never seed NumPy's global generator.
- `mutual_information` follows the standard sign convention
  $I(C)=\iint c\log c \ge 0$ (0.3.x returned the negative), and is `+inf`
  for copulas with a singular component.
- `CopulaSampler` and the checkerboard samplers no longer seed global RNGs;
  integer `random_state`s give reproducible calls.
- `import copul` no longer configures logging (no `logging.basicConfig`
  side effect, no `sys.path` manipulation) and is about a third faster:
  matplotlib, pandas and statsmodels are imported on first use. Library
  logging is at DEBUG level; stray `print` output was removed.
- `CISRearranger.rearrange_checkerboard` returns a numpy array with total
  mass one.
- Minimum Python is 3.10; runtime dependencies are numpy, scipy, sympy,
  pandas, matplotlib and statsmodels.

### Fixed

- `BivCheckMin`: Blest's $\nu$ missed the in-cell add-on; `lambda_L` /
  `lambda_U` crashed.
- `BivCheckW`: Spearman's footrule, Gini's $\gamma$ and Blest's $\nu$ were
  wrong.
- `CheckPi`: the vectorised pdf returned zeros.
- Bernstein copulas: `chatterjees_xi(condition_on_y=True)` was ignored;
  nan-free basis derivatives (finite pdf on the boundary, correct
  conditional distributions at 0 and (1, 1)); the constructor no longer
  normalises the caller's array in place.
- `ShuffleOfMin`: $n=1$ gives $\tau=\rho=1$ instead of nan; the constructor
  no longer modifies the caller's array.
- `markov_product` failed on every code path (NumPy 2 `trapz` removal,
  signature mismatches, infinite recursion).
- `xi_n_with_ci` used the wrong $\sqrt{n}$ scaling (the CI was always
  $(0,1)$); `test_independence` uses $\sqrt{n}\,\xi_n \to N(0, 2/5)$;
  `xi_ncalculate` no longer returns 0.5 on NaN input or silently truncates
  inputs of different lengths.
- FOCI/CODEC nearest neighbours could select the point itself when points
  are repeated.
- `generate_randomly`: inclusive per-sample grid sizes and proper
  `random_state` handling.
- `CornerSetVerifier` works for checkerboard copulas.
- `CISRearranger` normalisation (mass summed to `1/n_cols`).
- `bounds_from_xi`: cancellation-free formulas for $b>1$ (monotone results
  in $[0,1]$, $\nu$ bounds accurate up to $\xi\to1$).
- `StudentT` conditional distributions used the wrong degrees of freedom
  and scale; the inaccurate Student-t $\rho$ quadrature and the incorrect
  $\sigma=|\rho|$ override were removed.
- The generator-based Blomqvist $\beta$ of Archimedean copulas was wrong for
  some families (e.g. Nelsen 18) and was removed.
- `Checkerboarder.approximate_shuffle_of_min` read a non-existing attribute.
- Class-level state leaks: `CopulaBuilder._from_string` mutated the shared
  `_free_symbols` dict, and fixing a parameter of `Frechet` / `MVFrechet`
  (or restricting an interval) tightened the *class-level* `intervals` of
  all later instances.
- A duplicate, shadowed `blomqvists_beta` in `BivExtremeValueCopula` was
  removed; `Checkerboarder` no longer silently skips failing cdf
  evaluations.
- Numerical `cdf(P)`/`pdf(P)` failed for almost all parametric families
  (Archimedean, elliptical, extreme-value, Plackett, FGM, …) and `rvs` failed
  for extreme-value copulas and some boundary families.
- `Joe.rvs` sampled from a wrong distribution (non-uniform margins).
- `PiOverSigmaMinusPi`: `cond_distr_1`, `cond_distr_2` and `pdf` formulas
  were wrong.
- `B11`: Kendall's $\tau$ is $\delta(2+\delta)/3$ (was $\delta(3-2\delta)/3$),
  $\lambda_L=\lambda_U=\delta$ (were 0 and properties instead of methods),
  not absolutely continuous for $\delta>0$.
- `Mardia`: $\theta=-1$ is the lower Fréchet bound (the cdf returned
  $(uv+W)/2$); `is_absolutely_continuous` only for $\theta=0$.
- `Clayton` is absolutely continuous for $-1<\theta<0$; `EndSwapCopula`,
  `MedianSwapCopula`, `VThresholdCopula` and `ShuffleOfMin` declare their
  singular components.
- `DiagonalBandCopula`: closed-form cdf / conditional distribution (the
  symbolic integration did not terminate).
- `Plackett` numerics at $\theta=0$ (lower bound) and without cancellation;
  `Galambos`, `HueslerReiss`, `JoeEV`, `BB5`, `tEV`: `cdf()` and
  `cond_distr_*()` without arguments return symbolic expressions instead of
  raising.

### Deprecated

- `gini_gamma()`, `spearman_footrule()`, `lp_concordance()` → use
  `ginis_gamma()`, `spearmans_footrule()`, `lp_distance()`.
- `ShuffleOfMin.kendall_tau`, `spearman_rho`, `chatterjee_xi`,
  `tail_lower`, `tail_upper` → standard names.
- `markov_product(checkerboard=...)` (ignored; the result is always a
  `BivCheckPi`).

### Removed

- The `logging.basicConfig(level="INFO")` call and the
  `sys.path.append(<copul dir>)` hack in `copul/__init__.py`, and the
  module attribute `copul.log`.
- Unused runtime dependencies `sympy-plot-backends` and
  `typing-extensions`; `pandas-stubs` moved to the `dev` extra.

### Migration guide (0.3.x → 0.4.0)

| 0.3.x | 0.4.0 |
|---|---|
| `C.gini_gamma()` | `C.ginis_gamma()` |
| `C.spearman_footrule()` | `C.spearmans_footrule()` |
| `C.lp_concordance(p)` | `C.lp_distance(p)` |
| `cis, cds = C.is_cis()` | `cis, cds = C.cis_direction()` (`C.is_cis()` is now a `bool`) |
| `ShuffleOfMin(...).kendall_tau()` etc. | `.kendalls_tau()`, `.spearmans_rho()`, `.chatterjees_xi()`, `.lambda_L()`, `.lambda_U()` |
| `Clayton(2).spearmans_rho()` returned a SymPy number/integral | returns a `float`; use `method="symbolic"` for the SymPy route |
| symbolic formulas via fixed parameters | leave the parameter free (`Clayton().kendalls_tau()` → `theta/(theta + 2)`) or pass `method="symbolic"` |
| `mutual_information()` ≤ 0 | now ≥ 0 (standard sign) |
| relied on `import copul` printing INFO logs | configure logging in your application, e.g. `logging.basicConfig(level=logging.DEBUG)` |
| `sympy-plot-backends` installed with copul | install it yourself if you use it |
| `C.cdf(0.3, 0.7).evalf()` | `C.cdf(0.3, 0.7)` is a `float` |
| `C.cdf(u=u, v=v)` in loops | `C.cdf(U, V)` / `C.cdf(P)` with arrays |
| `C.rvs(n)` relying on `np.random.seed` for Gaussian/t | pass `random_state=` |

Other scripts calling the renamed measures keep working through the
deprecated aliases.

## [0.3.8] and earlier

See the git history.
