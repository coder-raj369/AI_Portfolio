.PHONY: help setup lint format test smoke results clean

help:
	@echo "setup    - create the dev environment with uv"
	@echo "lint     - ruff check + format check"
	@echo "format   - ruff format (writes)"
	@echo "test     - run the full pytest suite (CPU only)"
	@echo "smoke    - import + tiny-run every shipped module (catches rot)"
	@echo "results  - re-run every experiment that produces numbers in a README"

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

smoke:
	uv run python scripts/smoke_test.py

results:
	uv run python 00-foundations/micrograd-engine/experiments/train_spirals.py
	uv run python 00-foundations/optimizers-from-scratch/experiments/benchmark_convex.py
	uv run python 00-foundations/optimizers-from-scratch/experiments/benchmark_spirals.py
	uv run python 00-foundations/nn-primitives/experiments/stability_report.py
	uv run python 04-llms-and-genai/tokenizer-from-scratch/experiments/train_and_benchmark.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
