.PHONY: install test lint typecheck format demo eval clean docker

# ── Setup ─────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# ── Quality ───────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=agent --cov=tools --cov=evaluation --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow" --tb=short

lint:
	ruff check agent/ tools/ evaluation/ tests/

format:
	ruff format agent/ tools/ evaluation/ tests/

typecheck:
	mypy agent/ tools/ evaluation/ --ignore-missing-imports

check: lint typecheck test

# ── Run ───────────────────────────────────────────────────────────────
demo:
	python main.py demo

run:
	@echo "Usage: make run TASK='your task here'"
	python main.py run "$(TASK)"

eval-mock:
	python main.py eval --mock --trials 1

eval:
	python main.py eval --trials 3

# ── Docker ────────────────────────────────────────────────────────────
docker:
	docker build -t archon .

docker-run:
	docker run --rm -it -e OPENAI_API_KEY archon demo

docker-eval:
	docker run --rm -it -e HF_API_TOKEN archon eval --mock

# ── Clean ─────────────────────────────────────────────────────────────
clean:
	rm -rf traces/ evaluation/results/ __pycache__ .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
