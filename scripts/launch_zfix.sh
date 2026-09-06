#!/usr/bin/env bash
# z-variant fix screen (2026-09-06): does building the actor's z-reward from the SLOW,
# CENTERED index tower (measure_density_z_actor_reward=latent_centered, now the default in
# density_fb_z.yaml) close the gap to the bridge's .82? 2x2 over {latent, latent_centered}
# x {hard, soft} on cube_single, 500k, plus seeds 1-2 of the new default. The old default
# (latent + hard) at seed 0 is the existing exp/cube-single-play-v0_density_fb_z_flowbc_seed_0
# (.43 own-route @500k) and is not rerun. Each arm gets its own --save_dir so run dirs cannot
# collide. Both eval routes are logged.
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"; PY="${PY:-/dev/shm/.venv-jax/bin/python}"
LOGDIR=/dev/shm/zfix_logs; SESSION=zfix; STEPS=500000
RELABEL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
ARMS=(
  "lc_hard 0 measure_density_z_actor_reward=latent_centered measure_density_hard=true  0"
  "lc_hard 1 measure_density_z_actor_reward=latent_centered measure_density_hard=true  1"
  "lc_hard 2 measure_density_z_actor_reward=latent_centered measure_density_hard=true  2"
  "lc_soft 0 measure_density_z_actor_reward=latent_centered measure_density_hard=false 3"
  "l_soft  0 measure_density_z_actor_reward=latent          measure_density_hard=false 4"
  # full-state backward map + index (pool stores full rows since 2026-09-06), GPUs 5-7
  "lc_hard_state 0 backward_input=state measure_density_hard=true 5"
  "lc_hard_state 1 backward_input=state measure_density_hard=true 6"
  "lc_hard_state 2 backward_input=state measure_density_hard=true 7"
)
mkdir -p "$LOGDIR"; cd "$REPO"
tmux has-session -t "=$SESSION" 2>/dev/null && { echo "session $SESSION exists"; exit 1; }
first=1
for a in "${ARMS[@]}"; do
  read -r tag seed ov1 ov2 gpu <<< "$a"
  cmd="cd $REPO && CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl \
$PY -m impls.main --critic density_fb_z --actor flowbc --domain cube_single --seed $seed \
  --save_dir exp/zfix/$tag --auto_resume --train_steps $STEPS --eval_interval 50000 \
  --save_interval 250000 --override $ov1 --override $ov2 $RELABEL 2>&1 | tee $LOGDIR/${tag}_s${seed}.log; \
echo DONE_${tag}_s${seed}; sleep 86400"
  if [ $first -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "${tag}_s${seed}" "$cmd"; first=0
  else tmux new-window -d -t "=$SESSION" -n "${tag}_s${seed}" "$cmd"; fi
  echo "  ${tag} seed $seed -> GPU $gpu"
done
echo "launched in tmux '$SESSION'; logs in $LOGDIR"
