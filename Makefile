.PHONY: install install-pip smoke train-ant-medium train-ant-large train-cube sweep-ant-medium verify-ortho-fix verify-ortho-broken verify-lr-b-fix verify-lr-b-broken verify-both data clean help

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

# ── Pipeline check ────────────────────────────────────────────────────────── #

## Synthetic-data pipeline smoke (~20s, NO OGBench dataset needed)
smoke:
	python scripts/smoke.py

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

## Antmaze medium with seeds 1-3
sweep-ant-medium:
	python train.py --multirun seed=1,2,3 domain=antmaze_medium

# ── FB collapse diagnostic (50k steps, asserts metrics stay bounded) ─────── #

## Verify ortho_coef=100 prevents FB collapse on antmaze (~10 min on GPU)
verify-ortho-fix:
	python scripts/verify_ortho_fix.py

## Reproduce the collapse with the broken config (ortho_coef=1.0, no clip)
verify-ortho-broken:
	python scripts/verify_ortho_fix.py --ortho-coef 1.0 --clip-grad-norm 0

## Verify lr_b=1e-5 prevents FB collapse on antmaze (~10 min on GPU)
verify-lr-b-fix:
	python scripts/verify_lr_b_fix.py

## Reproduce the collapse with the broken lr_b=1e-4 (other knobs identical)
verify-lr-b-broken:
	python scripts/verify_lr_b_fix.py --lr-b 1.0e-4

## Run both verifications back-to-back; ortho first, then lr_b
verify-both: verify-ortho-fix verify-lr-b-fix

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
	@echo ""
	@echo "  smoke            Synthetic pipeline check (no data needed)"
	@echo "  train-ant-medium Antmaze medium run"
	@echo "  train-ant-large  Antmaze large run"
	@echo "  train-cube       Cube single run (FlowBC)"
	@echo "  sweep-ant-medium Antmaze medium, seeds 1-3"
	@echo ""
	@echo "  verify-ortho-fix    50k-step antmaze run with ortho_coef=100; asserts no collapse"
	@echo "  verify-ortho-broken Same but with ortho_coef=1.0 to reproduce collapse"
	@echo "  verify-lr-b-fix     50k-step antmaze run with lr_b=1e-5;     asserts no collapse"
	@echo "  verify-lr-b-broken  Same but with lr_b=1e-4 to reproduce collapse"
	@echo "  verify-both         Run ortho-fix then lr-b-fix (~20 min on GPU)"
	@echo ""
	@echo "  clean            Remove pycache, outputs"
	@echo "────────────────────────────────────────────────────────"
