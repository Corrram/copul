.PHONY: clean test coverage format build publish upgrade docs

# Package manager
UV := uv

# Python interpreter
PYTHON := $(UV) run python

# Directories
SRC_DIR := copul
TEST_DIR := tests
BUILD_DIR := build
DIST_DIR := dist

# Commands
PYTEST := $(UV) run pytest
PYLINT := $(UV) run pylint
BLACK := $(UV) run black
ISORT := $(UV) run isort
MYPY := $(UV) run mypy
SPHINX_APIDOC := $(UV) run sphinx-apidoc
TWINE := $(UV) run twine

all: upgrade clean format docs build test

# Clean build artifacts and cache files
clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist', '$(PACKAGE).egg-info', '.pytest_cache', 'htmlcov', 'docs/_build', '.coverage']]"
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

docs:
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('docs/source').rglob('*.rst') if p.name == 'modules.rst' or p.name.startswith('copul.') or p.read_text(encoding='utf-8').startswith('.. automodule::')]"
	cd docs && $(SPHINX_APIDOC) --force --automodule-options members,undoc-members -o ./source ../copul __init__.py
	$(PYTHON) -c "import pathlib; [p.write_text(p.read_text(encoding='utf-8').replace('   :show-inheritance:\n', ''), encoding='utf-8') for p in pathlib.Path('docs/source').glob('copul*.rst')]"
	cd docs && $(PYTHON) -m sphinx -E -b html ./source build/html

test:
	$(PYTEST) $(TEST_DIR) -m "not slow and not instable" -v -n 4

# Run tests with coverage
coverage:
	$(PYTEST) --cov=$(SRC_DIR) $(TEST_DIR) --cov-report=term-missing

# Format code with ruff
format:
	$(UV) run ruff check --fix --unsafe-fixes .
	$(UV) run ruff format .

build: clean
	$(UV) build
	$(TWINE) check dist/*
	uv pip install -e .

publish: clean build
	$(TWINE) upload dist/*

upgrade:
	@echo "Upgrading dev dependencies in root package..."
	$(UV) sync --upgrade --extra dev
	$(UV) export --format requirements-txt --extra dev --no-hashes --output-file requirements.txt > $(if $(filter $(OS),Windows_NT),NUL,/dev/null) 2>&1

