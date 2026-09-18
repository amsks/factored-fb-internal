#!/usr/bin/env bash
# OGBench density lever: oversample persistent (quasi-static) cube transitions by a factor
# lambda in {1,2,4,8} and ask whether FB degrades while Density-FB holds -- the cube analogue
# of adding ordinary-state mass in the toy MDP (see PAPER/ICLR/content/failure.tex).
#
# A transition is persistent when the cube block (obs cols 19:22) moves less than
# persist_vel_thresh between t and t+1; those rows are drawn lambda times as often, with the
# state-action SUPPORT preserved (transient rows keep weight 1). The reweighting is on the
# BASE dataset's index sampler (impls/utils/datasets.py::Dataset.get_random_idxs), so FB and
# Density-FB are reweighted identically.
#
# Methods = the Figure-5 pair (fb_tdj vs density_fb_z) at the tdjtab protocol, so the lever
# experiment and the failure diagnostic use the SAME configs. 2 critics x 4 lambdas x 3 seeds
# = 24 runs, 1M steps, cube_single. Route-B relabel eval logged (eval_relabel.csv).
#
# CONFOUND NOTE: oversampling reweights the whole batch, so the actor sees fewer manipulation
# demos too. Attribute any FB drop to the SUCCESSOR REPRESENTATION (not fewer demos) with the
# action-gap sign metric (scripts/probes/cube_bottleneck/sign_agreement.py): it is a critic
# diagnostic, so if FB's success falls AND its bottleneck sign-agreement collapses toward 0.5
# while Density-FB's holds, the lever acted through the SM as the lemma predicts.
#
#   DRY=1 bash scripts/launch_persist_lambda.sh          # print the 24-job list
#   JOBS_ONLY=1 bash scripts/launch_persist_lambda.sh    # one shell command per line
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/launch_persist_lambda.sh
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"
PY="${PY:-/dev/shm/.venv-jax/bin/python}"
LOGDIR="${LOGDIR:-/dev/shm/persist_lambda_logs}"
SESSION="${SESSION:-persist_lambda}"
SAVE="${SAVE:-exp/persist_lambda}"
STEPS="${STEPS:-1000000}"
EVAL_EVERY="${EVAL_EVERY:-100000}"
SAVE_EVERY="${SAVE_EVERY:-250000}"
LAMBDAS="${LAMBDAS:-1 2 4 8}"
SEEDS="${SEEDS:-0 1 2}"
THRESH="${THRESH:-0.005}"
BLOCK="${BLOCK:-19:22}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
cd "$REPO"
RELABEL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"

emit() { # critic lambda seed -> full command
  local critic=$1 lam=$2 seed=$3 sd="$SAVE/lam${2}" ov=""
  if [ "$critic" = "density_fb_z" ]; then
    ov="--override measure_density_z_actor_reward=latent_centered --override measure_density_hard=true --override 'measure_hit_spec=\"19:22:0.4\"'"
  fi
  echo "$PY -m impls.main --critic $critic --actor flowbc --domain cube_single --seed $seed" \
       "--auto_resume --train_steps $STEPS --eval_interval $EVAL_EVERY --save_interval $SAVE_EVERY" \
       "--save_dir $sd --override persist_oversample_lambda=$lam --override persist_vel_thresh=$THRESH" \
       "--override 'persist_block=\"$BLOCK\"' $ov $RELABEL"
}

JOBS=""
for critic in fb_tdj density_fb_z; do
  for lam in $LAMBDAS; do
    for seed in $SEEDS; do
      [ -f "$SAVE/lam${lam}/cube-single-play-v0_${critic}_flowbc_seed_${seed}/params_${STEPS}.pkl" ] && continue
      JOBS+="$critic $lam $seed"$'\n'
    done
  done
done

N=$(printf '%s' "$JOBS" | grep -c . || true)
if [ "${JOBS_ONLY:-0}" = "1" ]; then
  printf '%s' "$JOBS" | while read -r c l s; do [ -n "${c:-}" ] && emit "$c" "$l" "$s"; done
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
  while read -r c l s; do
    [ -n "${c:-}" ] || continue
    tag="lam${l}-${c}-s${s}"
    body+="echo '>>> $tag'; CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl $(emit "$c" "$l" "$s") 2>&1 | tee $LOGDIR/$tag.log; "
  done <<< "$slice"
  body+="echo ALL_DONE_GPU_$gpu; sleep 86400"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "gpu$gpu" "$body"; first=0
  else tmux new-window -d -t "=${SESSION}" -n "gpu$gpu" "$body"; fi
  echo "  GPU $gpu <- $(printf '%s' "$slice" | grep -c .) jobs"
done
echo "launched '$SESSION'"
