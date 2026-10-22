export PYTHONHASHSEED := 0

.PHONY: setup setup-all lint format typecheck test test-ml cov data export serve ui

setup:
	uv sync --extra dev

setup-all:
	uv sync --extra dev --extra ml --extra deploy --extra serve --extra ui

lint:
	uv run ruff check src tests benchmarks
	uv run ruff format --check src tests benchmarks

format:
	uv run ruff format src tests benchmarks
	uv run ruff check --fix src tests benchmarks

typecheck:
	uv run mypy

test:
	uv run pytest -m "not ml"

test-ml:
	uv run pytest -m ml

cov:
	uv run pytest -m "not ml" --cov --cov-report=term-missing

data:
	uv run castinspect acquire
	uv run castinspect data

export:
	uv run castinspect export

serve:
	uv run uvicorn --factory castinspect.service.main:app_from_env --host 0.0.0.0 --port 8000

ui:
	uv run streamlit run src/castinspect/ui/app.py
