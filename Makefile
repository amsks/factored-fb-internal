.PHONY: install install-pip smoke smoke-hecrl train-ant-medium train-ant-large train-cube train-ppp-cube sweep-ant-medium data data-hecrl clean help

# ── Installation ──────────────────────────────────────────────────────────── #

## Install dependencies with uv (recommended; uses .venv/)
install:
	uv pip install -r requirements.txt

## Install dependencies with plain pip
install-pip:
	python -m pip install -r requirements.txt

# ── OGBench dataset ──────────────────────────────────────────────────────── #

## Download + extract all state-based OGBench domains into ./datasets/
data:
	python scripts/extract_ogbench.py --output_folder datasets

## Download HECRL ppp-cube dataset from HuggingFace into ./datasets/hecrl/
data-hecrl:
	python scripts/download_hecrl.py ppp-cube

# ── Pipeline check ────────────────────────────────────────────────────────── #

## Synthetic-data pipeline smoke (~20s, NO OGBench dataset needed)
smoke:
	python scripts/smoke.py

## Synthetic pipeline smoke for HECRL (~5s, NO dataset needed)
smoke-hecrl:
	python scripts/smoke.py --benchmark hecrl

# ── Training (requires extracted OGBench data under ./datasets/) ─────────── #

## Antmaze medium (standard FB)
train-ant-medium:
	python train.py domain=antmaze_medium

## Antmaze large (standard FB)
train-ant-large:
	python train.py domain=antmaze_large

## Cube single (FlowBC)
train-cube:
	python train.py domain=cube_single

## HECRL ppp-cube (factored state, FlowBC)
train-ppp-cube:
	python train.py domain=ppp_cube

## Antmaze medium with seeds 1-3
sweep-ant-medium:
	python train.py --multirun seed=1,2,3 domain=antmaze_medium

# ── Housekeeping ──────────────────────────────────────────────────────────── #

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf outputs multirun

help:
	@echo ""
	@echo "Factored-FB — available targets"
	@echo "────────────────────────────────────────────────────────"
	@echo "  install          Install deps with uv into .venv/"
	@echo "  install-pip      Install deps with plain pip"
	@echo "  data             Download + extract OGBench to ./datasets/"
	@echo "  data-hecrl       Download HECRL ppp-cube dataset to ./datasets/hecrl/"
	@echo ""
	@echo "  smoke            Synthetic pipeline check (no data needed)"
	@echo "  smoke-hecrl      Synthetic pipeline check for HECRL (no data)"
	@echo "  train-ant-medium Antmaze medium run"
	@echo "  train-ant-large  Antmaze large run"
	@echo "  train-cube       Cube single run (FlowBC)"
	@echo "  train-ppp-cube   HECRL ppp-cube run (FlowBC)"
	@echo "  sweep-ant-medium Antmaze medium, seeds 1-3"
	@echo ""
	@echo "  clean            Remove pycache, outputs"
	@echo "────────────────────────────────────────────────────────"
