.PHONY: sync test lint fmt api dev-db

sync:
	uv sync --all-packages

test:
	uv run pytest

lint:
	uv run ruff check pipeline api

fmt:
	uv run ruff check --fix pipeline api && uv run ruff format pipeline api

api:
	uv run uvicorn synchro_api.main:app --reload --port 8000
