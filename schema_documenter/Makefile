.PHONY: install install-dev test lint format clean build publish

# Check if uv is installed, if not install it
UV := $(shell command -v uv 2> /dev/null)
ifndef UV
    $(warning "uv is not installed. Installing uv...")
    UV := $(shell curl -LsSf https://astral.sh/uv/install.sh | sh)
endif

install:
	uv pip install -r requirements.txt

install-dev:
	uv pip install -r requirements-dev.txt
	uv pip install -e .

test:
	pytest

lint:
	flake8 schema_documenter tests
	isort --check-only schema_documenter tests
	black --check schema_documenter tests

format:
	isort schema_documenter tests
	black schema_documenter tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

build:
	python -m build

publish:
	python -m twine upload dist/* 