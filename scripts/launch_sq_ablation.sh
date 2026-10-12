#!/usr/bin/env bash
# Density-FB loss ablation: the same softmax critic trained by squared error on the predicted
# probabilities instead of cross-entropy, everything else identical to the table runs.
# Both task interfaces (goal-conditioned density_fb, reward-inference density_fb_z), on
# cube-single, cube-double and scene.
#
#   bash scripts/launch_sq_ablation.sh <train_steps> "<seeds>"
#   e.g. bash scripts/launch_sq_ablation.sh 500000 "0 1 2"      # phase 1
#        bash scripts/launch_sq_ablation.sh 1000000 "0 1 2 3 4"  # phase 2: resumes 0-2, adds 3-4
#
# Runs land in exp/sq_ablation/<env>_<critic>_flowbc_seed_<s>; --auto_resume continues a run
# from its latest checkpoint. Jobs go round-robin over GPUs 0-7 in tmux session `sqabl`.
# DRY=1 prints the commands only.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:?usage: launch_sq_ablation.sh <train_steps> "<seeds>"}
read -r -a SEEDS <<< "${2:?seeds}"
PY=${PY:-/dev/shm/.venv-jax/bin/python}
SESSION=${SESSION:-sqabl}
declare -A HIT=([cube_single]='"19:22:0.4"' [cube_double]='"19:22:0.4,28:31:0.4"'
                [scene_play]='"19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"')
ENVV="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"

cmds=()
for critic in density_fb density_fb_z; do
  for dom in cube_single cube_double scene_play; do
    for s in "${SEEDS[@]}"; do
      extra="--override measure_density_hard=true"
      [ "$critic" = density_fb_z ] && extra="--override measure_density_z_actor_reward=latent_centered $extra"
      cmds+=("$critic $dom $s|$PY -m impls.main --actor flowbc --auto_resume --train_steps $STEPS \
--eval_interval 100000 --save_interval 250000 --critic $critic --domain $dom --seed $s \
--save_dir exp/sq_ablation $extra --override 'measure_hit_spec=${HIT[$dom]}' \
--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 \
--override eval_relabel_episodes=10 --override measure_density_loss=sq")
    done
  done
done

if [[ -n "${DRY:-}" ]]; then printf '%s\n' "${cmds[@]#*|}"; exit 0; fi
mkdir -p logs/sq_ablation
tmux new-session -d -s "$SESSION" -n init "sleep 1" 2>/dev/null || true
i=0
for c in "${cmds[@]}"; do
  read -r critic dom s <<< "${c%%|*}"
  name="${critic}_${dom}_s${s}"
  log=logs/sq_ablation/${name}.log
  tmux new-window -t "$SESSION" -n "$name" \
    "cd $PWD && CUDA_VISIBLE_DEVICES=$((i % 8)) $ENVV ${c#*|} >> $log 2>&1; echo JOB_DONE rc=\$? >> $log"
  echo "gpu $((i % 8)): $name"
  i=$((i + 1)); sleep 5
done
tmux kill-window -t "$SESSION:init" 2>/dev/null || true
