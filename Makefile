export PYTHONHASHSEED := 0

.PHONY: setup lint format typecheck test cov

setup:
	uv sync --extra dev

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy

test:
	uv run pytest -m "not ml"

cov:
	uv run pytest -m "not ml" --cov --cov-report=term-missing
