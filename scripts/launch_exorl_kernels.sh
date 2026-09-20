#!/usr/bin/env bash
# Reward-relevant hit kernels on walker and quadruped (2026-09-20, after the cheetah velocity-block win).
# Walker obs = orientations (0-13) + height (14) + velocity (15-23); rewards use height, uprightness, speed.
# Quadruped obs = egocentric_state (0-43) + torso_velocity (44-46) + torso_upright (47) + imu + force_torque;
# rewards use torso_upright and torso_velocity[0]. Radii = hit-rate calibration (scripts/calibrate_hit_spec.py).
# Base recipe per domain = the table's best: walker = round-3 recipe (1M recipe + sqt), quadruped = 1M recipe.
#   usage: bash scripts/launch_exorl_kernels.sh        # 12 jobs (4 arms x seeds 0-2), tmux exorl_kernels, GPUs 0-5
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
ENV="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
PY=/dev/shm/.venv-jax/bin/python
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
COMMON="--auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --eval_episodes 10 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
DFB="--override 'value_hidden_dims=[512,512,512]' --override tau=0.005 --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4"
SQT="--override measure_density_z_rand_frac=1.0 --override actor.q_normalize=true --override 'value_hidden_dims=[1024]' --override tau=0.001"
DOM_walker="--domain dmc-walker --override discount=0.98 --override lr_actor=1e-4 $SQT"
DOM_quadruped="--domain dmc-quadruped --override discount=0.98 --override lr_actor=1e-4"
ARM_walker_vel="--override 'measure_hit_spec=\"15:24:15.428\"'"
ARM_walker_hv="--override 'measure_hit_spec=\"14:15:0.007,15:24:15.428\"'"
ARM_quadruped_vel="--override 'measure_hit_spec=\"44:47:0.110\"'"
ARM_quadruped_vu="--override 'measure_hit_spec=\"44:47:0.110,47:48:0.001\"'"
JOBS=("walker_vel 0" "walker_vel 1" "walker_vel 2" "walker_hv 0" "walker_hv 1" "walker_hv 2"
      "quadruped_vel 0" "quadruped_vel 1" "quadruped_vel 2" "quadruped_vu 0" "quadruped_vu 1" "quadruped_vu 2")
cmd() { local arm=$1 seed=$2 save=$3 dom=${arm%%_*} domflags armflags
  domflags=$(eval echo "\"\$DOM_$dom\""); armflags=$(eval echo "\"\$ARM_$arm\"")
  echo "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $COMMON --save_dir $save $DFB $ACT $domflags --seed $seed $armflags"; }
mkdir -p logs/exorl_kernels; SESSION=exorl_kernels
tmux new-session -d -s "$SESSION" -n init "sleep 1"; i=0
for J in "${JOBS[@]}"; do read -r arm s <<< "$J"; log=logs/exorl_kernels/${arm}_s${s}.log
  tmux new-window -t "$SESSION" -n "${arm}_s${s}" "cd $HOME/git/Factored-FB-seam && CUDA_VISIBLE_DEVICES=$(( i % 6 )) $(cmd $arm $s exp/exorl_kernels/$arm) > $log 2>&1; echo JOB_DONE rc=\$? >> $log; sleep 86400"
  i=$((i+1)); sleep 12; done
tmux kill-window -t "$SESSION:init" || true; echo "launched $i jobs in tmux session $SESSION"
