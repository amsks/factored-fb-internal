#!/bin/bash
# The minimal fb_tdj parity arm (docs/notes/2026-09-03-fb-vs-tdjepa.md #7).
#
#   bash scripts/launch_fb_tdj_parity.sh
#
# WHY THIS EXISTS. The seam's plain FB on cube-single scores 0.11-0.23 against td_jepa's
# published 0.420 / the bridge's reproduced 0.420-0.532, and the line-by-line audit in that
# doc found the training stack (critic math, actor, networks, update loop) is already a
# faithful port -- the one genuine, UNMEASURED difference is the EVAL PROTOCOL. td_jepa's
# number is a reward-inference zero-shot readout (relabel 10k training transitions by their
# next observation's simulator qpos against the task's goal_xyzs, infer z = project_z(sum
# r*B(next_obs)), roll 10 any-step episodes), not the seam's z = B(goal_obs) Dirac. This arm
# reproduces that protocol -- env/ogbench.py::OGBenchAdapter.relabel_infer_batch, wired into
# impls/main.py as a SECOND eval route (eval_relabel/task<N>) that runs alongside, not
# instead of, the existing goal-conditioned one, so both are reportable per run.
#
# critic/fb_tdj.yaml = fb.yaml at ortho_coef 1000 (the reference stack's value; fb_ref
# already showed ortho alone does not close the gap under the OLD eval, ruling it out as a
# confound here too). Matrix: lr_b in {1e-4, 1e-5} x 3 seeds = 6 runs, cube-single-play-v0,
# flowbc actor (bc_coeff 3.0 is its own default), 1M steps.
#
# CUDA GRAPHS OFF: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED has taken a cell in ~25% of waves
# historically (commit 8010207); it faults inside the logging line's device->host copy.
set -euo pipefail

REPO=/home/adityamohan/git/Factored-FB-seam
PY=/dev/shm/.venv-jax/bin/python
LOGDIR=/dev/shm/fbtdj_logs
SESSION=fbtdj

STEPS=1000000
EVAL_EVERY=100000
SAVE_EVERY=250000

# td_jepa's OGBench zero-shot protocol (docs/notes/2026-09-03-fb-vs-tdjepa.md #5/#7):
# 10k relabelled transitions per task, shift_reward=1 (turns "successes - num_cubes" into a
# 0/1 cube-at-goal indicator), 10 any-step episodes per task. The EXISTING goal-conditioned
# route is untouched (still --eval_episodes default = 20).
RELABEL_OV='eval_relabel_size=10000 eval_relabel_shift=1.0 eval_relabel_episodes=10'

read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5}"
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session '$SESSION' exists"; exit 1; }
[ -x "$PY" ] || { echo "missing jax venv python at $PY"; exit 1; }
mkdir -p "$LOGDIR"

# name          lr_b   seed
ARMS=(
  "tdj-lrb4-s0  1.0e-4 0"
  "tdj-lrb4-s1  1.0e-4 1"
  "tdj-lrb4-s2  1.0e-4 2"
  "tdj-lrb5-s0  1.0e-5 0"
  "tdj-lrb5-s1  1.0e-5 1"
  "tdj-lrb5-s2  1.0e-5 2"
)
[ "${#GPU_LIST[@]}" -eq "${#ARMS[@]}" ] || { echo "need ${#ARMS[@]} GPUs, got ${#GPU_LIST[@]}"; exit 1; }

first=1
for i in "${!ARMS[@]}"; do
  read -r name lrb seed <<< "${ARMS[$i]}"
  gpu="${GPU_LIST[$i]}"
  ovargs=""
  for kv in lr_b=$lrb $RELABEL_OV; do ovargs="$ovargs --override $kv"; done
  cmd="cd $REPO && \
CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
WANDB_MODE=offline \
$PY -m impls.main --critic fb_tdj --actor flowbc --domain cube_single --seed $seed \
  --train_steps $STEPS --eval_interval $EVAL_EVERY --save_interval $SAVE_EVERY \
  --auto_resume $ovargs \
  2>&1 | tee $LOGDIR/$name.log"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "$name" "$cmd"; first=0
  else tmux new-window -t "=${SESSION}" -n "$name" "$cmd"; fi
  echo "  $name -> GPU $gpu   (lr_b=$lrb seed=$seed)"
done
echo "launched '$SESSION', ${#ARMS[@]} windows on GPUs ${GPU_LIST[*]}"
echo "logs: $LOGDIR/*.log   poll: tmux attach -t $SESSION   or: tail -f $LOGDIR/tdj-lrb4-s0.log"
