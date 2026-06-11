#!/usr/bin/env bash
# scripts/agents/gciql_flow/pixel.sh — GCIQL + FlowBC actor on visual_cube_single (DrQ pixel front-end).
# Actor-only ablation of GCIQL-pixel-drq (bc_coeff=3, the constant across flow methods).
USAGE="usage: bash scripts/agents/gciql_flow/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000
    RUN_GROUP=GCIQL-Pixel-DrQ-Flow
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-GCIQL-Pixel-DrQ-Flow}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
export TRAIN_STEPS
_run_jax_sweep cube_single_visual_gciql_flow_drq "$RUN_GROUP"
