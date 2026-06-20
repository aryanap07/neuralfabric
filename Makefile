.PHONY: install dev test lint format build publish clean

install:
	pip install .

dev:
	pip install -e ".[dev]"

test:
	pytest --cov=neuralfabric --cov-report=term-missing

lint:
	ruff check src tests
	mypy src

format:
	black src tests

build:
	python -m build

publish: build
	twine upload dist/*

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
