.PHONY: install test test-fast run smoke clean help

VENV ?= /dev/shm/.venv-jax
PY   := $(VENV)/bin/python
# JAX on CPU by default for tests; drop these to use the GPU.
JAXENV := XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu

# ── Installation ──────────────────────────────────────────────────────────── #

## Create an isolated JAX venv in tmpfs and install deps (ephemeral on reboot)
install:
	uv venv --python 3.11 $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	@echo "[install] done. Verify: $(PY) -c 'import jax; print(jax.devices())'"

# ── Tests ─────────────────────────────────────────────────────────────────── #

## Full impls/ parity + seam suite (CPU)
test:
	$(JAXENV) $(PY) -m pytest tests/impls/ -q

## Fast collection-only check (catches import breakage in seconds)
test-fast:
	$(JAXENV) $(PY) -m pytest tests/impls/ -q --collect-only

# ── Training ──────────────────────────────────────────────────────────────── #

CRITIC ?= fb
ACTOR  ?= flowbc
DOMAIN ?= cube_single

## Train one critic x actor pair on cube-single (override CRITIC=/ACTOR=/DOMAIN=)
run:
	$(PY) -m impls.main --critic $(CRITIC) --actor $(ACTOR) --domain $(DOMAIN)

## Short smoke: 5 train steps of the default pair on cube-single (CPU)
smoke:
	$(JAXENV) $(PY) -m impls.main --critic $(CRITIC) --actor $(ACTOR) \
	    --domain $(DOMAIN) --train_steps 5 --eval_interval 1000000 --save_interval 1000000

# ── Housekeeping ──────────────────────────────────────────────────────────── #

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf exp outputs multirun

help:
	@echo ""
	@echo "Factored-FB (JAX impls/) — targets"
	@echo "────────────────────────────────────────────────────────"
	@echo "  install    Create JAX venv ($(VENV)) + install deps"
	@echo "  test       Full impls/ suite (CPU)"
	@echo "  test-fast  Collection-only import check"
	@echo "  run        Train CRITIC x ACTOR on DOMAIN (defaults: fb x flowbc, cube_single)"
	@echo "  smoke      5-step train smoke of the default pair"
	@echo "  clean      Remove pycache / exp / outputs"
	@echo "────────────────────────────────────────────────────────"
	@echo "  Valid critics: fb gciql isfb iql_bilinear simple_sf sf_feature centered_bilinear fb_forward_bilinear"
	@echo "  Valid actors:  flowbc ddpgbc awr fql   (see check_combo for valid pairs)"
