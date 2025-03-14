.PHONY: clean test coverage format build publish upgrade docs

# Python interpreter
PYTHON := python

# Package manager
UV := uv

# Directories
SRC_DIR := copul
TEST_DIR := tests
BUILD_DIR := build
DIST_DIR := dist

# Commands
PYTEST := pytest
PYLINT := pylint
BLACK := black
ISORT := isort
MYPY := mypy

all: clean format upgrade docs build test

# Clean build artifacts and cache files
clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist', '$(PACKAGE).egg-info', '.pytest_cache', 'htmlcov', 'docs/_build', '.coverage']]"
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

docs:
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('docs/source').rglob('*.rst') if p.name == 'modules.rst' or p.name.startswith('copul.') or p.read_text(encoding='utf-8').startswith('.. automodule::')]"
	cd docs && sphinx-apidoc -o ./source ../copul __init__.py
	cd docs && $(PYTHON) -m sphinx -b html ./source build/html

test:
	$(PYTEST) $(TEST_DIR) -v -n 4

# Run tests with coverage
coverage:
	$(PYTEST) --cov=$(SRC_DIR) $(TEST_DIR) --cov-report=term-missing

# Format code with ruff
format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

# Build package
build:
	$(UV) build

publish: clean build
	twine upload dist/*

upgrade:
	@echo "Upgrading dev dependencies in root package..."
	$(UV) sync --upgrade --extra dev
	$(UV) export --format requirements-txt --extra dev --no-hashes --output-file requirements.txt > $(if $(filter $(OS),Windows_NT),NUL,/dev/null) 2>&1

