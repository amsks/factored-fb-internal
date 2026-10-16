#!/usr/bin/env bash
# HILP cube-double: td_jepa's full 2x2 sweep, evaluated in single-task (no cube swap) envs.
#
#   bash scripts/launch_hilp_cd_grid.sh            # GPUs 5 6 7, 20 jobs, all concurrent
#   DRY=1 bash scripts/launch_hilp_cd_grid.sh      # print the job list and exit
#
# WHY. td_jepa reports the best cell of a 2x2 sweep per domain (feature learning rate
# {1e-4, 1e-5} x random-goal probability {0.375, 0.5}). Our cube-double HILP cell (9.2 vs
# their 20.0) only ran the base cell (1e-4, 0.375; scripts/launch_hilp_tdjc.sh, seeds 0-2,
# plus exp/tdjtab seeds 3-4). This runs all four cells, five seeds each, with the
# same training length, evaluation route (reward inference, eval_relabel.csv) and eval
# schedule as the base cell, so the four cells are directly comparable.
#
# 2026-10-15: the reward-inference eval now rolls out with the cube swap OFF
# (eval_relabel_singletask, impls/main.py), as td_jepa's single-task envs do. With the swap on,
# half the episodes ask for a cube-goal pairing the inferred task vector does not describe;
# rescoring the base cell's 1M checkpoints moved 5.7 -> 10.0 (3 seeds, 100 episodes each).
# The base cell is rerun here so all four cells share that eval. Output: exp/hilp_cd_grid_st/.
#
# IDEMPOTENT: --auto_resume, and jobs with a params_1000000.pkl are skipped.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/hilpcdgrid_st_logs
SESSION="${SESSION:-hilpcdgrid_st}"
OUT=exp/hilp_cd_grid_st

STEPS=1000000
EVAL_EVERY=100000
SAVE_EVERY=250000
CD_HIT='measure_hit_spec=19:22:0.4,28:31:0.4'
RELABEL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10 --override eval_relabel_singletask=true"

read -r -a GPU_LIST <<< "${GPUS:-5 6 7}"
[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"; cd "$REPO"

JOBS=""
for cell in "lf1e-4_pr0.375:1e-4:0.375" "lf1e-4_pr0.5:1e-4:0.5" "lf1e-5_pr0.375:1e-5:0.375" "lf1e-5_pr0.5:1e-5:0.5"; do
  name="${cell%%:*}"; rest="${cell#*:}"; lf="${rest%%:*}"; pr="${rest#*:}"
  for seed in 0 1 2 3 4; do
    [ -f "$OUT/$name/cube-double-play-v0_hilp_tdjc_flowbc_seed_${seed}/params_${STEPS}.pkl" ] && continue
    JOBS+="$name $lf $pr $seed"$'\n'
  done
done
N=$(printf '%s' "$JOBS" | grep -c . || true)
echo "$N jobs:"; printf '%s' "$JOBS" | nl -ba | sed 's/^/   /'
[ "$N" -gt 0 ] || { echo "nothing to run"; exit 0; }
[ "${DRY:-0}" = "1" ] && exit 0
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session '$SESSION' exists"; exit 1; }

NG=${#GPU_LIST[@]}; i=0; first=1
while read -r name lf pr seed; do
  [ -n "${name:-}" ] || continue
  gpu="${GPU_LIST[$((i % NG))]}"; i=$((i + 1))
  tag="${name}-s${seed}"
  body="cd $REPO; CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl \
$PY -m impls.main --critic hilp_tdjc --actor flowbc --domain cube_double --seed $seed \
  --save_dir $OUT/$name --auto_resume --train_steps $STEPS --eval_interval $EVAL_EVERY \
  --save_interval $SAVE_EVERY --override $CD_HIT \
  --override group_lr.hilp_phi1=$lf --override group_lr.hilp_phi2=$lf \
  --override hilp_p_randomgoal=$pr $RELABEL 2>&1 | tee $LOGDIR/$tag.log; \
echo DONE_$tag; sleep 86400"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "$tag" "$body"; first=0
  else tmux new-window -d -t "=${SESSION}" -n "$tag" "$body"; fi
  echo "  GPU $gpu <- $tag"
done <<< "$JOBS"
echo "launched session '$SESSION'"
