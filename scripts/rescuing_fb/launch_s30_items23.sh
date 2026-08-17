#!/usr/bin/env bash
# Spec 30a Items 2 & 3 -- the two ISOLATION BASELINES on cube-double, 500k, seed 0.
#
#   tdi_cd_exact_s0   paper-exact TD InfoNCE (their Eq. 10 critic + their Eq. 11 /
#                     App. D.3 actor, their supports, their l2-normalized towers, their
#                     goal relabeling)  -- "how does the PUBLISHED method do here?"
#   lsif_cd_phi_s0    goal-conditioned outcome-space LSIF (m = F^T B(phi(x)), FB's own
#                     objective, external kernel reward readout) -- the
#                     PRIVILEGED-COORDINATES confound control.
#
# Named launch_s30_items23 (not launch_spec30_*) so it cannot be confused with, or
# accidentally overwrite, the item-1 dispatcher already running the two centered arms.
#
# WHY CUBE-DOUBLE. It is the load-bearing separation (spec 30, Priority): cube-single does
# not discriminate. Anchors to read against, native GC success at 500k:
#   density soft .736 | density hard .680 | plain FB .00 | BC floor .019.
#
# READING RULES (binding, spec 30a):
#   * NO result claims before 500k; then quote 500k AND the mean over >= 350k.
#   * tdi_cd_exact_s0 is a CONTROL. A faithful arm that scores well is a real finding, not
#     a failure -- do not tune it toward or away from any number. Promotion gate: it does
#     NOT trivially reproduce the soft result.
#   * lsif_cd_phi_s0 is expected to stay substantially below Density-FB; if it closes most
#     of the gap, the normalization story weakens (spec 30's decision tree).
#   * the MATCHED-EXTRACTOR TD InfoNCE arm (cube_double_state_tdinfonce_matched_d50.yaml)
#     is deliberately NOT in this wave -- it is the promotion wave.
#
# Instruments to watch mid-run (never as success claims): measure/tdi_{top1,pos_prob,
# w_ess,ce_immediate,ce_bootstrap} on the TD InfoNCE arm -- top1 must lift off the 1/(1+P)
# chance line; measure/lsif_{m_msq,diag_mean,neg_frac,offdiag} on the LSIF arm -- m_msq
# blowing up is the known unbounded-linear-attraction failure of the FB/LSIF objective and
# is the thing this arm is allowed to exhibit (we do NOT add a scale normalizer: that would
# be a third unmatched difference from Density-FB).
#
# Modeled on launch_spec30_p1.sh. active() counts EVERY main.py job on the box, so this
# dispatcher cooperates with the other track's jobs. Do NOT kill this dispatcher to change
# MAX: it is the pipeline parent of its children and SIGTERM propagates to them.
#
# Run in tmux:
#   tmux new -d -s s30i23 'scripts/rescuing_fb/launch_s30_items23.sh 2>&1 | tee -a /dev/shm/spec30_i23_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_s30_items23.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec30_i23}
LOG=/dev/shm/spec30_i23_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-8}
GPUS=(${GPUS:-0 1})          # 4 and 5 belong to the item-1 centered arms
FP=$OUTROOT/factored-fb
export GCIQL_WANDB_ENTITY=${GCIQL_WANDB_ENTITY:-amsks}

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

JOBS=(
  "tdi_cd_exact_s0|cube_double_state_tdinfonce_exact_d50|0"
  "lsif_cd_phi_s0|cube_double_state_lsif_phi_d50|0"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r _t cfg _s <<<"$j"
  [ -f "configs/gciql/$cfg.yaml" ] || { echo "[s30i23] FATAL missing config configs/gciql/$cfg.yaml"; exit 1; }
done
[ -f "$HOME/.ogbench/data/cube-double-play-v0.npz" ] || {
  echo "[s30i23] FATAL missing dataset ~/.ogbench/data/cube-double-play-v0.npz (/dev/shm gets wiped)"; exit 1; }
[ -x /dev/shm/.venv-jax/bin/python ] || {
  echo "[s30i23] FATAL missing JAX venv /dev/shm/.venv-jax (volatile -- recreate with make install-jax)"; exit 1; }

echo "[s30i23] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX (shared with any other wave), entity=$GCIQL_WANDB_ENTITY, GPUS=${GPUS[*]}"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[s30i23] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[s30i23] DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[s30i23] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 120
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[s30i23] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[s30i23] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
