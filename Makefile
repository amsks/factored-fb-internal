.PHONY: install test test-fast format run smoke clean help

VENV ?= .venv
PY   := $(VENV)/bin/python
# JAX on CPU by default for tests; drop these to use the GPU.
JAXENV := XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu PYTHONPATH=$(CURDIR)
# Optional dependency groups (pyproject.toml): exorl metaworld maniskill figures dev
EXTRAS ?= dev

# Installation

## Create a JAX venv and install the project with the chosen extras
install:
	uv venv --python 3.11 $(VENV)
	uv pip install --python $(PY) -e ".[$(EXTRAS)]"
	@echo "[install] done. Verify: $(PY) -c 'import jax; print(jax.devices())'"

# Tests

## Full test suite (CPU)
test:
	$(JAXENV) $(PY) -m pytest tests/ -q

## Fast collection-only check (catches import breakage in seconds)
test-fast:
	$(JAXENV) $(PY) -m pytest tests/ -q --collect-only

## Sort imports and format the code
format:
	$(PY) -m isort impls env scripts analysis tests
	$(PY) -m ruff format impls env scripts analysis tests
	$(PY) -m ruff check --fix impls env scripts analysis tests --exit-zero

# Training

CRITIC ?= density_fb
ACTOR  ?= flowbc
DOMAIN ?= cube_single

## Train one critic x actor pair (override CRITIC=/ACTOR=/DOMAIN=)
run:
	PYTHONPATH=$(CURDIR) $(PY) -m impls.main --critic $(CRITIC) --actor $(ACTOR) --domain $(DOMAIN)

## Short smoke: 5 train steps of the chosen pair (CPU)
smoke:
	$(JAXENV) $(PY) -m impls.main --critic $(CRITIC) --actor $(ACTOR) \
	    --domain $(DOMAIN) --train_steps 5 --eval_interval 1000000 --save_interval 1000000

# Housekeeping

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf exp

help:
	@echo ""
	@echo "Factored-FB (JAX) targets"
	@echo ""
	@echo "  install    Create venv ($(VENV)) + install with EXTRAS=$(EXTRAS)"
	@echo "  test       Full test suite (CPU)"
	@echo "  test-fast  Collection-only import check"
	@echo "  format     isort + ruff format"
	@echo "  run        Train CRITIC x ACTOR on DOMAIN (defaults: density_fb x flowbc, cube_single)"
	@echo "  smoke      5-step train smoke of the chosen pair"
	@echo "  clean      Remove pycache / exp"
	@echo ""
	@echo "  Critics: density_fb density_fb_z density_fb_state fb fb_tdj fb_ref hilp_tdjc rldp"
	@echo "  Actors:  flowbc ddpgbc"
	@echo "  Domains: cube_single cube_double scene_play, dmc-{walker,quadruped,cheetah,point-mass-maze},"
	@echo "           mw-{push,bin-picking,box-close}, ms-{push-cube,pick-cube,stack-cube,poke-cube,lift-peg}"
