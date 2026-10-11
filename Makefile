.PHONY: help all install test test-all test-slow doctest coverage lint format docs build publish clean clean-build upgrade

# Package manager / interpreter (override e.g. with `make test UV=`)
UV := uv
RUN := $(if $(UV),$(UV) run --active,)
PYTHON := python
PY := $(RUN) $(PYTHON)
PYTEST := $(PY) -m pytest
RUFF := $(PY) -m ruff
TWINE := $(PY) -m twine

SRC_DIR := copul
TEST_DIR := tests
DOCS_SRC := docs/source
DOCS_OUT := docs/build/html
FAST := -m "not slow and not instable"

help:
	@echo "all       clean, format, upgrade, docs, build and fast tests (in order)"
	@echo "install   editable install with dev, docs and optim extras"
	@echo "test      fast test suite (parallel)"
	@echo "test-all  all tests including slow and instable ones"
	@echo "test-slow only the slow tests"
	@echo "doctest   doctests of copul.measures, copul.optim, copul.search"
	@echo "coverage  fast suite with coverage report"
	@echo "lint      ruff check + format check"
	@echo "format    ruff fix + format"
	@echo "docs      build the Sphinx HTML docs (warnings are errors)"
	@echo "build     build sdist and wheel, twine check"
	@echo "publish   upload dist/* to PyPI"
	@echo "clean     remove build artifacts and caches"
	@echo "upgrade   upgrade the uv lock file and requirements.txt"

# Recursive invocations keep the stages ordered, including with `make -j`.
all:
	$(MAKE) clean
	$(MAKE) format
	$(MAKE) upgrade
	$(MAKE) docs
	$(MAKE) build
	$(MAKE) test

install:
	$(UV) pip install -e ".[dev,docs,optim]"

test:
	$(PYTEST) $(TEST_DIR) $(FAST) -n auto

test-all:
	$(PYTEST) $(TEST_DIR) -n auto

test-slow:
	$(PYTEST) $(TEST_DIR) -m slow -n auto

doctest:
	$(PYTEST) --doctest-modules $(SRC_DIR)/measures $(SRC_DIR)/optim $(SRC_DIR)/search

coverage:
	$(PYTEST) $(TEST_DIR) $(FAST) -n auto --cov=$(SRC_DIR) --cov-report=term-missing

lint:
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)
	$(RUFF) format --check $(SRC_DIR) $(TEST_DIR)

format:
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR)
	$(RUFF) format $(SRC_DIR) $(TEST_DIR)

docs:
	$(PY) -m sphinx -W --keep-going -b html $(DOCS_SRC) $(DOCS_OUT)

build: clean-build
	$(PY) -m build
	$(TWINE) check dist/*

publish: build
	$(TWINE) upload dist/*

clean-build:
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist']]"

clean: clean-build
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'htmlcov', 'docs/build']]; pathlib.Path('.coverage').unlink(missing_ok=True)"
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('$(SRC_DIR)').rglob('__pycache__')]"
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('$(TEST_DIR)').rglob('__pycache__')]"

upgrade:
	$(UV) sync --active --upgrade --extra dev --extra docs
	$(UV) export --format requirements-txt --extra dev --extra docs --no-hashes --output-file requirements.txt
