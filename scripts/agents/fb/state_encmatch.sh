#!/usr/bin/env bash
# scripts/agents/fb/state_encmatch.sh — FB+FlowBC on cube_single with two
# param-matched MLP encoders (cube_single_encmatch). Same FB-loss knobs as
# scripts/agents/fb/state.sh (ortho_coef/lr_b come from the domain config), but
# the state representation path now carries ~10.3M encoder params to match the
# pixel DrQ run (~15.54M online). Isolates the param-count confound, not pixels.
# See PAPER/specs/2026-06-07-encmatch-state-encoder-ablation.md.
#
# Universal knobs: SEEDS, GPUS, TRAIN_STEPS, RUN_GROUP, DATA_PATH, DRY_RUN.
USAGE="usage: bash scripts/agents/fb/state_encmatch.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'   space-separated seeds
    GPUS='0 1'          space-separated GPU ids (seed i -> GPUS[i % nGPU])
    TRAIN_STEPS=1000000
    RUN_GROUP=fb_state_encmatch  wandb group + log dir name
    DATA_PATH=/dev/shm/factored-fb/datasets
    SAVE_EVAL_VIDEOS=false
    DRY_RUN=1           print commands instead of launching"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-fb_state_encmatch}"

_run_fb_sweep cube_single_encmatch "$RUN_GROUP"
