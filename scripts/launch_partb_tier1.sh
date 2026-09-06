#!/usr/bin/env bash
# Part B, Tier 1 -- the runs that complete the td_jepa-style reward-inference table on cube.
#
#   density_fb_z   cube_single x3, cube_double x3   (Density-FB, latent-index variant; the
#                                                    goal-conditioned checkpoints have no
#                                                    reward-inference path)
#   fb_tdj         cube_double x3                   (cube_single x3 already done, 1M)
#
# 1M steps, 3 seeds, every run logs BOTH routes: eval.csv (goal-conditioned) and
# eval_relabel.csv (td_jepa protocol: 10k relabelled samples, shift 1.0, 10 episodes,
# any-step success). Same pattern as scripts/launch_hilp_tdjc.sh. Idempotent: a run whose
# final checkpoint exists is skipped; --auto_resume picks up partial runs.
#
#   DRY=1 bash scripts/launch_partb_tier1.sh        # print the job list only
#   GPUS="0 1 2 3" bash scripts/launch_partb_tier1.sh
#   JOBS_ONLY=1 bash scripts/launch_partb_tier1.sh   # one shell command per line (for slurm)
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"
PY="${PY:-/dev/shm/.venv-jax/bin/python}"
LOGDIR="${LOGDIR:-/dev/shm/partb_logs}"
SESSION="${SESSION:-partb1}"
STEPS=1000000
EVAL_EVERY=100000
SAVE_EVERY=250000
CD_HIT='measure_hit_spec=19:22:0.4,28:31:0.4'
RELABEL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
cd "$REPO"

JOBS=""
add() { # critic dom envname override
  local critic=$1 dom=$2 envn=$3 ov=$4
  for seed in 0 1 2; do
    [ -f "exp/${envn}_${critic}_flowbc_seed_${seed}/params_${STEPS}.pkl" ] && continue
    JOBS+="$critic $dom $seed $ov"$'\n'
  done
}
add density_fb_z cube_single cube-single-play-v0 -
add density_fb_z cube_double cube-double-play-v0 "$CD_HIT"
add fb_tdj       cube_double cube-double-play-v0 -

N=$(printf '%s' "$JOBS" | grep -c . || true)
if [ "${JOBS_ONLY:-0}" = "1" ]; then
  printf '%s' "$JOBS" | while read -r critic dom seed ov; do
    [ -n "${critic:-}" ] || continue
    ovflag=""; [ "$ov" = "-" ] || ovflag="--override $ov"
    echo "$PY -m impls.main --critic $critic --actor flowbc --domain $dom --seed $seed --auto_resume --train_steps $STEPS --eval_interval $EVAL_EVERY --save_interval $SAVE_EVERY $ovflag $RELABEL"
  done
  exit 0
fi
echo "$N jobs:"; printf '%s' "$JOBS" | nl -ba | sed 's/^/   /'
[ "$N" -gt 0 ] || { echo "nothing to run"; exit 0; }
[ "${DRY:-0}" = "1" ] && exit 0
[ -x "$PY" ] || { echo "missing python at $PY"; exit 1; }
mkdir -p "$LOGDIR"
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
echo "launched in tmux session '$SESSION'; logs in $LOGDIR"
