#!/bin/bash
# BASELINE row of the with/without-privileged-coordinates table: FB, RLDP and HILP on
# cube-single and cube-double, each WITH the object-coordinate B/phi input and WITHOUT it.
#
# Fills the cells the density campaign left empty:
#   fb / fb_oc      -- BOTH, on both envs: plain FB had no cube cell here at all, and
#                      fb.py had no `backward_input` hook until now. This is the first
#                      clean test of privileged coordinates on FB -- distinct from
#                      feat/goal-conditioned-fb's V2 `fixed_b=cube_xyz` (0.06 vs plain
#                      FB's 0.50), which ALSO collapsed z_dim to 3 and so confounded the
#                      input restriction with a fatal latent width.
#   hilp_cube       -- the matched state-input CONTROL for hilp_oc, both envs. Without it
#                      we cannot say whether object coordinates HELP or HURT HILP, only
#                      that hilp_oc floors at 0.09.
#   rldp            -- the state-input control on cube-double (cube-single has n=3).
#   rldp_oc seed 1  -- rldp_oc reached 0.98 at 300k then collapsed to 0.28 by 500k, at
#                      n=1. That collapse carries the 'density buys stability' claim, so
#                      it needs a second seed before it can be quoted.
#
#   bash scripts/launch_cube_oc_density.sh
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/launch_cube_oc_density.sh
#
# THE CONFIG RULE (the user's, and it is what the yamls implement):
#   * SUBSTRATE HPs come from the reference side -- official HILP / td_jepa. For HILP that
#     means critic/hilp.yaml's values PLUS the four td_jepa knobs that are about the
#     objective being DEGENERATE rather than about DMC tuning (hilp_q_loss=false, both head
#     reductions mean, sf_loss_scale 0.5; we measured q_loss=true collapsing psi to 2.5-2.8
#     effective dimensions of 50). For RLDP, critic/rldp.yaml ALREADY matches td_jepa
#     structurally -- single joint objective, ortho_coef 0.01, live regularizer -- so
#     nothing is carried across.
#   * DENSITY + ENV HPs come from OUR cube runs: discount 0.99 (density_fb.yaml, NOT
#     td_jepa's DMC 0.98), temp 1.0, actor_q_center true, the hard immediate target, and
#     `backward_input: hit_spec` at the env's own object slice and success radius.
#
# `backward_input: hit_spec` IS the transferable insight. On the 500k seed-0 oc screen it
# separated rldp_density_oc 1.00 from rldp_density 0.15-0.25, and psm_density_oc 0.90 from
# psm_density 0.15-0.20 -- while doing nothing for the non-density arm (rldp_oc 0.05). The
# win is density x object-coordinates JOINTLY. HILP had no such hook; `_phi_in` was added
# to impls/critics/hilp.py mirroring rldp.py's `_b_in`, default "state" so every published
# HILP cell is byte-identical (tests/impls/test_hilp_oc.py).
#
# HIT-SPEC LANDMINE, hit during the smoke test of this very campaign: OmegaConf reads an
# unquoted single-block override `measure_hit_spec=19:22:0.4` as the SEXAGESIMAL literal
# 69720.4. cube-single's spec is the yaml default, so it is NOT passed at all; cube-double's
# has a comma, which breaks the pattern and survives unquoted.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/cubebase_logs
SESSION=cubebase

STEPS=500000
EVAL_EVERY=50000
SAVE_EVERY=250000

read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"

# Exact-name guard: tmux has-session PREFIX-matches, so '=' pins the exact name.
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session '$SESSION' exists"; exit 1; }
[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"

CD_HIT='measure_hit_spec=19:22:0.4,28:31:0.4'

# name          critic            domain       seed  extra-overrides
ARMS=(
  "cs-fb_oc-s0    fb_oc      cube_single  0     -"
  "cd-fb_oc-s0    fb_oc      cube_double  0     $CD_HIT"
  "cs-fb-s0       fb         cube_single  0     -"
  "cd-fb-s0       fb         cube_double  0     -"
  "cs-hilp-s0     hilp_cube  cube_single  0     -"
  "cd-hilp-s0     hilp_cube  cube_double  0     -"
  "cd-rldp-s0     rldp       cube_double  0     -"
  "cs-rldp_oc-s1  rldp_oc    cube_single  1     -"
)
[ "${#GPU_LIST[@]}" -eq "${#ARMS[@]}" ] || {
  echo "GPUS must list exactly ${#ARMS[@]} GPUs, got ${#GPU_LIST[@]}"; exit 1; }

first=1
for i in "${!ARMS[@]}"; do
  read -r name critic dom seed ov <<< "${ARMS[$i]}"
  gpu="${GPU_LIST[$i]}"
  ovflag=""
  [ "$ov" = "-" ] || ovflag="--override $ov"
  cmd="cd $REPO && \
CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_MODE=offline \
$PY -m impls.main \
  --critic $critic \
  --actor flowbc \
  --domain $dom \
  --seed $seed \
  --train_steps $STEPS \
  --eval_interval $EVAL_EVERY \
  --save_interval $SAVE_EVERY \
  $ovflag \
  2>&1 | tee $LOGDIR/$name.log"
  if [ "$first" -eq 1 ]; then
    tmux new-session -d -s "$SESSION" -n "$name" "$cmd"; first=0
  else
    tmux new-window -t "=${SESSION}" -n "$name" "$cmd"
  fi
  echo "  $name -> GPU $gpu   ($critic / $dom / seed $seed)"
done

echo "launched tmux session '$SESSION', ${#ARMS[@]} windows on GPUs ${GPU_LIST[*]}"
echo "logs: $LOGDIR/<arm>.log   attach: tmux attach -t =$SESSION"
