.PHONY: install test test-fast test-agents lint format smoke clean help

# ── Installation ──────────────────────────────────────────────────────────── #

install:
	uv pip install -r requirements.txt

install-dev: install
	uv pip install ruff pytest

# ── Testing ───────────────────────────────────────────────────────────────── #

## Run the full test suite
test:
	pytest tests/ -v

## Run fast tests only (skips OGBench data I/O and env tests)
test-fast:
	pytest tests/ -v -x \
	  --ignore=tests/test_envs.py \
	  -k "not test_load_ogbench_data"

## Run only the agent smoke tests (no env or data deps)
test-agents:
	pytest tests/test_agents.py tests/test_nn_models.py tests/test_buffers.py -v

# ── Code quality ──────────────────────────────────────────────────────────── #

lint:
	ruff check .

format:
	ruff format .

# ── Training ──────────────────────────────────────────────────────────────── #

## Fast pipeline smoke test (~30s, verifies everything is wired)
smoke:
	python train.py +experiment=smoke_test domain=antmaze-medium-navigate-v0

## Full antmaze medium run
train-ant-medium:
	python train.py domain=antmaze-medium-navigate-v0

## Full antmaze large run
train-ant-large:
	python train.py domain=antmaze-large-navigate-v0

## Cube manipulation with FlowBC actor
train-cube:
	python train.py domain=cube_single

## Seeded sweep over antmaze medium (seeds 1-3)
sweep-ant-medium:
	python train.py --multirun seed=1,2,3 domain=antmaze-medium-navigate-v0

# ── Housekeeping ──────────────────────────────────────────────────────────── #

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache outputs multirun

help:
	@echo ""
	@echo "Factored-FB — available targets"
	@echo "────────────────────────────────────────────────────────"
	@echo "  install          Install dependencies with uv"
	@echo "  install-dev      Install deps + ruff + pytest"
	@echo ""
	@echo "  test             Full test suite"
	@echo "  test-fast        Skip env/data tests (no OGBench needed)"
	@echo "  test-agents      Agent + nn_models + buffer tests only"
	@echo ""
	@echo "  lint             ruff check"
	@echo "  format           ruff format"
	@echo ""
	@echo "  smoke            30-second pipeline smoke test"
	@echo "  train-ant-medium Antmaze medium full run"
	@echo "  train-ant-large  Antmaze large full run"
	@echo "  train-cube       Cube manipulation (FlowBC)"
	@echo "  sweep-ant-medium Seeds 1-3 multirun"
	@echo ""
	@echo "  clean            Remove pycache, outputs, pytest cache"
	@echo "────────────────────────────────────────────────────────"