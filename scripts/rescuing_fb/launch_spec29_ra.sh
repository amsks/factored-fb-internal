#!/usr/bin/env bash
# Spec 29 R-A (PAPER/Multistep-FB/specs/29-ra-dual-index.md section 1): the dual-corrected
# index, cube-single 500k seed 0 screen of the two arms.
#
#   ra/kernel  zhat = sqrt(d)*norm((Sigma_hat+lam I)^-1 m_hat) + actor integrates the ACTUAL
#              reward samples (q = sum_j w_j(s,a,zhat) r(u_j))      THE MAIN ARM
#   ra/latent  same index + the closed-form readout B̃^T zhat/sqrt(d)  (secondary, cheap)
#
# ---------------------------------------------------------------------------------
# GATE -- G29-0 (pre-flight, eval-only) MUST HAVE READ OUT BEFORE THIS SCRIPT RUNS.
# G29-0 computes dual indices zhat(g) over the 4096-goal set on the existing z24/z25/g
# checkpoints and measures mean pairwise cos. PASS = de-collapsed (<< .99; sphere ref
# .002). If the dual indices are ALSO collapsed the arm is DEAD and spec 29 pre-registers
# that these cells are not launched at all -- the compute would buy a foregone conclusion.
# Until the gate is recorded: `DRY_RUN=1` only.
#
# STATUS 2026-08-14: G29-0 PASSED, verified from the artifacts, and the arm is cleared.
#   dual ridge lambda=.01   cos -.0003..+.0075   PR2 9-23
#   shipped sphere z(g)     cos .998-1.000       PR2 1.0
#   primal moment (control) cos .998-1.000       PR2 1.0   <- reward weighting alone does
#                                                             NOT de-collapse; the RIDGE
#                                                             SOLVE does
#   sphere reference        cos -.0007           PR2 49.4
# cos and PR2 agree, so this is not the saturation artifact that fooled z_cos_spread.
# lambda stays .01: {.001,.1} are the ablation ends (.001 inverts ill-conditioned
# directions, .1 drifts back toward a cone on z25_cs_tilt), not default candidates.
# ---------------------------------------------------------------------------------
#
# Single-variable discipline: both configs differ from the spec-28 cell-(ii)/(iv) z lattice
# by measure_density_z_encode=dual (+ measure_density_z_enc_tau=0.0005, which is NOT a free
# second variable -- create() refuses the dual encode without a slow-EMA index tower), and
# from each other by measure_density_z_actor_reward alone. measure_density_z_rand_frac stays
# 0.5: the task mixture is explicitly unchanged in this wave.
#
# ⚠ NEW ARM: archived z24/z25 numbers stop being comparable to these cells. Never mix them
# into one table without saying so.
#
# G29-1 (from the FIRST eval, mid-run kill gate): actor goal-sensitivity must lift off
# 0.000 (noise floor .19) via scripts/rescuing_fb/eval_z_action_ranking.py; train-time
# pairwise cos of zhat = `z_cos_spread`; density_w_ztv; never-extracts curve; plus the
# spec-29 geometry instruments measure/density_zsigma_{eff_rank,cond}. An arm flat at
# sensitivity 0 by 250k is killed rather than run to 500k -- but mid-run SUCCESS numbers
# are NEVER quoted (spec-18 rule: a cell read .984 at 450k and .876 at 500k).
#
# Modeled on launch_spec28_p1.sh. Concurrency: active() counts EVERY main.py job on the box
# (not just this wave's), so this dispatcher cooperates with whatever else is running --
# including the Q4 topline fill-in, which yields to R-A on conflict. Do NOT kill this
# dispatcher to change MAX or anything else: it is the pipeline parent of its children and
# SIGTERM/SIGINT propagates to them (that has killed live cells here before). DRY_RUN is
# checked BEFORE the wait-for-a-free-slot loop, so `DRY_RUN=1` on a full box still prints
# and exits promptly instead of blocking.
#
# Run in tmux:
#   tmux new -d -s spec29ra 'scripts/rescuing_fb/launch_spec29_ra.sh 2>&1 | tee -a /dev/shm/spec29_ra_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec29_ra.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec29_ra}
LOG=/dev/shm/spec29_ra_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
FP=$OUTROOT/factored-fb
export GCIQL_WANDB_ENTITY=${GCIQL_WANDB_ENTITY:-amsks}

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

JOBS=(
  "s29_cs_ra_kernel|cube_single_state_measure_density_z_dual_kernelrew_d50|0"
  "s29_cs_ra_latent|cube_single_state_measure_density_z_dual_latentrew_d50|0"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r _t cfg _s <<<"$j"
  [ -f "configs/gciql/$cfg.yaml" ] || { echo "[s29ra] FATAL missing config configs/gciql/$cfg.yaml"; exit 1; }
done

echo "[s29ra] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX (shared with any other wave), entity=$GCIQL_WANDB_ENTITY"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[s29ra] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[s29ra] DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[s29ra] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 120
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[s29ra] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[s29ra] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
