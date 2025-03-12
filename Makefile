# Makefile for copul package using uv as package manager

.PHONY: all clean test lint format install dev-install upgrade upgrade-dev

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

all: clean install test

# Clean build artifacts
clean:
	if exist $(BUILD_DIR) rd /s /q $(BUILD_DIR)
	if exist $(DIST_DIR) rd /s /q $(DIST_DIR)
	if exist *.egg-info rd /s /q *.egg-info
	powershell -Command "Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force"
	powershell -Command "Get-ChildItem -Path . -Include *.pyc,*.pyo,*.pyd -Recurse -File | Remove-Item -Force"

# Install package in development mode
dev-install:
	$(UV) pip install -e ".[dev]"

# Install package
install:
	$(UV) pip install .

# Run tests with pytest
test:
	$(PYTEST) $(TEST_DIR) -v -n 4

# Run tests with coverage
coverage:
	$(PYTEST) --cov=$(SRC_DIR) $(TEST_DIR) --cov-report=term-missing

# Format code with ruff
format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

# Run static type checking
typecheck:
	$(MYPY) $(SRC_DIR)

# Run linting
lint:
	$(PYLINT) $(SRC_DIR)

# Build package
build:
	$(UV) pip build

# Create virtual environment
venv:
	$(UV) venv

# Install development dependencies
dev-deps:
	$(UV) pip install pytest pytest-cov black isort mypy pylint

# Run all quality checks
quality: format typecheck lint test

upgrade:
	@echo "Upgrading dev dependencies in root package..."
	$(UV) sync --active --upgrade --extra dev
	$(UV) export --format requirements-txt --extra dev --no-hashes --output-file requirements.txt > $(if $(filter $(OS),Windows_NT),NUL,/dev/null) 2>&1

upgrade-dev:
	@echo "Upgrading dev dependencies in root package..."
	$(UV) sync --upgrade --extra dev
