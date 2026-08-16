#!/usr/bin/env bash
# Spec 30 P1 / spec 30a Item 1: the CENTERED index, cube-single 500k seed-0 screen.
#
#   ra2_cs_centpoint  z = sqrt(d)*norm(B̄(phi(g)) - mu_hat)                (arm A)
#                     "does mean subtraction ALONE repair goal prompting?"
#   ra2_cs_centdual   zhat = sqrt(d)*norm((C_hat+lam I)^-1 m_c), M=256 at eval  (arm B)
#                     the general reward-sample interface, exercised for the first time
#
# Both with the KERNEL readout (measure_density_z_actor_reward=kernel), matched to the
# spec-29 R-A recipe in everything else (rand_frac .5, enc_tau 5e-4, lambda .01).
#
# WHY THIS WAVE. Spec 30 re-measured the geometry: spec 28's "rank-1 collapse" was largely
# a SOFTMAX-GAUGE MEAN OFFSET (||mu_B|| 33-103x the centered signal; centered covariance
# eff rank 6-21 of 50; centering ALONE lifts point-index reconstruction AUC .43-.86 ->
# .99+), and the trained uncentered encoder is SHIFT-UNSTABLE (cos(enc(r), enc(r+1)) ~ 0)
# where the centered one is exactly invariant. Preflight evidence: results/spec30_geom/
# (cite; do NOT rerun).
#
# ⚠ THE EVAL RULE (hard requirement). ra2_cs_centdual's native goal eval goes through the
# REWARD-SAMPLE API over a frozen M=256 kernel pool -- sample_actions builds kg(goal, pool)
# and encodes that; it never silently substitutes the point prompt. Spec 29's .812 came
# from exactly that substitution (spec 30 sec.4.8), which is why it could not be read as
# general-encoder evidence. The point shortcut is a SEPARATE diagnostic column only
# (agent.sample_actions_point / density_z_point_prompt).
#
# MID-RUN INSTRUMENTS, ARMED FROM STEP 0 (spec 30a): scripts/rescuing_fb/
# eval_z_action_ranking.py at eval intervals (actor goal-sensitivity must lift off 0.000;
# noise floor .19) + measure/z_cos_spread + the centered-geometry keys
# measure/density_zcov_{eff_rank,cond,min_eig}, density_zc_{mc_norm,c_norm,r_var,
# fallback_frac,sync_mu,sync_cov}. KILL an arm whose actor sensitivity is still ~0 at
# 250k -- do not run it to 500k. NEVER quote mid-run SUCCESS numbers (spec-18 rule: a cell
# read .984 at 450k and .876 at 500k); at 500k quote 500k AND the mean over >= 350k.
# Anchors: cube-single g .93 / BC .111 / R-A-uncentered .812.
#
# Modeled on launch_spec29_ra.sh. active() counts EVERY main.py job on the box, so this
# dispatcher cooperates with the other track's jobs. Do NOT kill this dispatcher to change
# MAX: it is the pipeline parent of its children and SIGTERM propagates to them.
#
# Run in tmux:
#   tmux new -d -s s30p1 'scripts/rescuing_fb/launch_spec30_p1.sh 2>&1 | tee -a /dev/shm/spec30_p1_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec30_p1.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec30_p1}
LOG=/dev/shm/spec30_p1_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-8}
GPUS=(${GPUS:-4 5})
FP=$OUTROOT/factored-fb
export GCIQL_WANDB_ENTITY=${GCIQL_WANDB_ENTITY:-amsks}

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

JOBS=(
  "ra2_cs_centpoint|cube_single_state_measure_density_z_centpoint_d50|0"
  "ra2_cs_centdual|cube_single_state_measure_density_z_centdual_d50|0"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r _t cfg _s <<<"$j"
  [ -f "configs/gciql/$cfg.yaml" ] || { echo "[s30p1] FATAL missing config configs/gciql/$cfg.yaml"; exit 1; }
done
[ -f "$HOME/.ogbench/data/cube-single-play-v0.npz" ] || {
  echo "[s30p1] FATAL missing dataset ~/.ogbench/data/cube-single-play-v0.npz (/dev/shm gets wiped)"; exit 1; }
[ -x /dev/shm/.venv-jax/bin/python ] || {
  echo "[s30p1] FATAL missing JAX venv /dev/shm/.venv-jax (volatile -- recreate with make install-jax)"; exit 1; }

echo "[s30p1] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX (shared with any other wave), entity=$GCIQL_WANDB_ENTITY, GPUS=${GPUS[*]}"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[s30p1] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[s30p1] DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[s30p1] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 120
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[s30p1] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[s30p1] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
