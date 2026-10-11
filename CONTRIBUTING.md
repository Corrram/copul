# Contributing to copul

Thank you for your interest in `copul`! Contributions of all kinds are
welcome: bug reports, questions, documentation fixes, new copula families,
new dependence measures, closed-form formulas and performance improvements.

By participating in this project you agree to abide by its
[Code of Conduct](CODE_OF_CONDUCT.md).

## Getting help

- **Documentation:** <https://copul.readthedocs.io> (user guide and API
  reference).
- **Questions:** open an issue on
  [GitHub](https://github.com/Corrram/copul/issues/new/choose) using the
  *Question* template. Please include a short code snippet showing what you
  are trying to do.
- For matters that should not be discussed publicly, write to the maintainer
  at <marcus.rockel@finance.uni-freiburg.de>.

## Reporting bugs

Please open an issue using the *Bug report* template and include

1. a minimal, self-contained code example that reproduces the problem,
2. what you expected to happen and what happened instead (full traceback,
   wrong numerical value together with the value you expected and where it
   comes from, e.g. a closed form from the literature),
3. the output of

   ```bash
   python -c "import copul, sympy, numpy, scipy, sys; print(copul.__version__, sympy.__version__, numpy.__version__, scipy.__version__, sys.version)"
   ```

Wrong mathematical results (an incorrect closed form, a measure outside its
range, a copula that is not 2-increasing, ...) are bugs of the highest
priority. Please report them even if you are not sure.

## Suggesting features

Open an issue using the *Feature request* template. For a new copula family
or dependence measure, please give a reference (paper or book) for its
definition and, if known, for closed-form expressions of its measures.

## Development setup

```bash
git clone https://github.com/Corrram/copul.git
cd copul
python -m venv .venv && source .venv/bin/activate   # or use uv
pip install -e ".[dev,docs]"                        # dev includes cvxpy
```

With [uv](https://docs.astral.sh/uv/) the equivalent is `make install`.

Run `make all` for the complete development workflow: clean generated
artifacts, format, upgrade dependencies, build the documentation, build and
validate the distributions, then run the fast test suite. The stages run
sequentially even with `make -j`. This command updates `uv.lock` and
`requirements.txt` and applies Ruff's safe fixes and formatting. The upgrade
includes both the `dev` and `docs` extras.

Individual targets also support a chosen interpreter, for example
`make test UV= PYTHON=.venv/Scripts/python.exe` on Windows.

## Running the tests

```bash
make test        # fast test suite in parallel: pytest -m "not slow and not instable" -n auto
make test-all    # everything, including slow and numerically instable tests
make doctest     # doctests of copul.measures, copul.optim and copul.search
make lint        # ruff check + ruff format --check
make docs        # Sphinx documentation, warnings are errors
```

Without `uv`, call the tools directly, e.g.
`pytest -m "not slow and not instable" -n auto`. The same checks run in
continuous integration on every push and pull request (Python 3.10–3.13, see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)); a pull request can be
merged once they pass.

## Making changes

1. Fork the repository and create a branch from `master`.
2. Make your change with tests. Tests live in `tests/`, mirroring the
   package layout (e.g. tests for `copul/family/archimedean/nelsen5.py` go to
   `tests/family/archimedean/`). Mark tests that take more than a few seconds
   with `@pytest.mark.slow`.
3. Format and lint: `ruff format copul tests` and `ruff check copul tests`
   (line length 100).
4. Document public functions and classes with NumPy-style docstrings.
5. Add an entry under the unreleased version in [`CHANGELOG.md`](CHANGELOG.md).
6. Open a pull request against `master` describing what you changed and why.

### Mathematical contributions

`copul` is used for research, so correctness comes first.

- Every closed-form expression (a cdf, a density, a dependence measure, a
  boundary of a region) needs a reference or a derivation in the docstring.
- Every closed form needs a test that compares it with an independent route,
  typically the numerical engine (`C.spearmans_rho(method="numeric")`),
  Monte Carlo (`method="mc"`) or a known special case (independence,
  Fréchet bounds).
- New copula families subclass the appropriate base class (for instance
  `BivArchimedeanCopula`, `BivExtremeValueCopula` or `BivCopula`), declare
  their parameter intervals, and are registered in `copul/family_list.py`
  (`Families`) so that they appear in `cp.families` and in the generic
  property tests in `tests/properties/`.
- Results that are not yet publicly available (for instance in a paper under
  preparation) should not be added before the corresponding preprint is
  online.

## Releases (maintainers)

1. Update the version in `pyproject.toml` and `CITATION.cff`, and the date of
   the release in `CHANGELOG.md` and `CITATION.cff`.
2. Commit, then tag and push: `git tag -a vX.Y.Z -m "copul X.Y.Z"` and
   `git push origin vX.Y.Z`.
3. The [release workflow](.github/workflows/release.yml) builds the sdist and
   wheel, checks them, publishes them to PyPI (trusted publishing, configured
   once on PyPI for the `release.yml` workflow and the `pypi` environment)
   and creates a GitHub release.
