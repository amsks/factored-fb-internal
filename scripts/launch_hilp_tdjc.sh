#!/usr/bin/env bash
# HILP + Density-HILP matched EXACTLY to td_jepa's OGBench config, cube-single + cube-double.
#
#   bash scripts/launch_hilp_tdjc.sh            # all 8 GPUs
#   DRY=1 bash scripts/launch_hilp_tdjc.sh      # print the job list and exit
#
# WHY 1M AND THE RELABEL EVAL. td_jepa's OGBench BASE_CFG is num_train_steps 1_000_000, and
# the number it reports is a reward-INFERENCE zero-shot readout, not a z = B(goal) Dirac.
# Matching the config but scoring it our usual way would leave the comparison exactly as
# broken as it was for FB, where the same checkpoints read 0.197 goal-conditioned against
# 0.378 under their protocol. So these runs log BOTH routes: eval.csv (goal-conditioned, our
# tables) and eval_relabel.csv (td_jepa's protocol, the headline for this arm), the same
# pairing scripts/launch_fb_tdj_parity.sh established.
#
# NOT MATCHED, deliberately, and it should be stated with any number from these runs:
# td_jepa reports the best cell of a 2x2 sweep (lr_features {1e-4, 1e-5} x prob_random_goal
# {0.375, 0.5}) over TEN seeds. These runs pin the BASE_CFG values (1e-4, 0.375) at three
# seeds. That reproduces their CONFIGURATION, not their SELECTION PROCEDURE.
#
# IDEMPOTENT: --auto_resume everywhere, and the job list skips anything that already has a
# params_1000000.pkl. Re-run after any fault.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/hilptdjc_logs
SESSION="${SESSION:-hilptdjc}"

STEPS=1000000
EVAL_EVERY=100000
SAVE_EVERY=250000
CD_HIT='measure_hit_spec=19:22:0.4,28:31:0.4'
RELABEL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"

read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"; cd "$REPO"

JOBS=""
for critic in hilp_tdjc hilp_density_tdjc; do
  for pair in "cube_single:cube-single-play-v0:-" "cube_double:cube-double-play-v0:$CD_HIT"; do
    dom="${pair%%:*}"; rest="${pair#*:}"; envn="${rest%%:*}"; ov="${rest#*:}"
    for seed in 0 1 2; do
      [ -f "exp/${envn}_${critic}_flowbc_seed_${seed}/params_${STEPS}.pkl" ] && continue
      JOBS+="$critic $dom $seed $ov"$'\n'
    done
  done
done
N=$(printf '%s' "$JOBS" | grep -c . || true)
echo "$N jobs:"; printf '%s' "$JOBS" | nl -ba | sed 's/^/   /'
[ "$N" -gt 0 ] || { echo "nothing to run"; exit 0; }
[ "${DRY:-0}" = "1" ] && exit 0
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session '$SESSION' exists"; exit 1; }

NG=${#GPU_LIST[@]}; first=1
for w in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$w]}"
  slice=$(printf '%s' "$JOBS" | awk -v w="$w" -v n="$NG" 'NF && (NR-1)%n==w')
  [ -n "$slice" ] || continue
  body="cd $REPO; "
  while read -r critic dom seed ov; do
    [ -n "${critic:-}" ] || continue
    ovflag=""; [ "$ov" = "-" ] || ovflag="--override $ov"
    tag="${dom}-${critic}-s${seed}"
    body+="echo '>>> $tag'; \
CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl \
$PY -m impls.main --critic $critic --actor flowbc --domain $dom --seed $seed \
  --auto_resume --train_steps $STEPS --eval_interval $EVAL_EVERY \
  --save_interval $SAVE_EVERY $ovflag $RELABEL 2>&1 | tee $LOGDIR/$tag.log; "
  done <<< "$slice"
  body+="echo ALL_DONE_GPU_$gpu; sleep 86400"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "gpu$gpu" "$body"; first=0
  else tmux new-window -d -t "=${SESSION}" -n "gpu$gpu" "$body"; fi
  echo "  GPU $gpu <- $(printf '%s' "$slice" | grep -c .) jobs"
done
echo "launched session '$SESSION'"
