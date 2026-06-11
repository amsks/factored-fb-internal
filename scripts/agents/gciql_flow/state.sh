#!/usr/bin/env bash
# scripts/agents/gciql_flow/state.sh — GCIQL + FlowBC actor on cube_single (state).
# Actor-only ablation of GCIQL: IQL critic unchanged, flow actor (bc_coeff=3, the
# constant across all flow methods).
USAGE="usage: bash scripts/agents/gciql_flow/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=1000000
    RUN_GROUP=GCIQL-State-Flow
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-GCIQL-State-Flow}"
_run_jax_sweep cube_single_state_gciql_flow "$RUN_GROUP"
