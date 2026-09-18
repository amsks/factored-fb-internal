#!/usr/bin/env bash
# Density-FB on ExoRL at the 1M RECIPE (tau 0.005, value tower [512,512,512]; the exp/walker_arch
# recipe that scored walker 681 vs 634 for the 2M reference recipe), so every ExoRL row in
# tab:manipulation sits on ONE recipe at ONE budget. Runs the two domains that never had it
# (quadruped, maze with the reference actor lr 1e-6) at 5 seeds, and tops walker up to 5 seeds.
# Waits for the round-2 tmux session (exorl_disc2) to finish before starting; 2 jobs per GPU.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
while tmux has-session -t exorl_disc2 2>/dev/null; do sleep 120; done
mkdir -p logs/exorl_1mrecipe
ENV="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
PY=/dev/shm/.venv-jax/bin/python
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
COMMON="--auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --eval_episodes 10 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
DFB="--override 'value_hidden_dims=[512,512,512]' --override tau=0.005 --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4"
declare -a JOBS=()
# ALL four domains, 5 seeds, at the references' DMC actor lr 1e-4 (ours has been the ddpgbc yaml
# default 3e-4 everywhere; td_jepa launch_fb_dmc.py / rldp_train_dmc.py / Touati all run 1e-4).
# Maze keeps the references' maze value 1e-6. Walker at 3e-4 (exp/walker_arch, 681) is the control.
for s in 0 1 2 3 4; do
  JOBS+=("--domain dmc-walker --override discount=0.98 --override 'measure_hit_spec=\"0:24:15.9\"' --override lr_actor=1e-4 --seed $s > logs/exorl_1mrecipe/walker_s$s.log")
  JOBS+=("--domain dmc-cheetah --override discount=0.98 --override 'measure_hit_spec=\"0:17:7.5\"' --override lr_actor=1e-4 --seed $s > logs/exorl_1mrecipe/cheetah_s$s.log")
  JOBS+=("--domain dmc-quadruped --override discount=0.98 --override 'measure_hit_spec=\"0:78:21.087\"' --override lr_actor=1e-4 --seed $s > logs/exorl_1mrecipe/quadruped_s$s.log")
  JOBS+=("--domain dmc-point-mass-maze --override discount=0.99 --override 'measure_hit_spec=\"0:2:0.03\"' --override lr_actor=1e-6 --seed $s > logs/exorl_1mrecipe/maze_s$s.log")
done
i=0
for J in "${JOBS[@]}"; do
  g=$(( i % 8 ))   # 20 jobs, up to 3 per GPU (~770 MiB each)
  CUDA_VISIBLE_DEVICES=$g bash -c "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $COMMON --save_dir exp/exorl_dfb_1mrecipe $DFB $ACT $J 2>&1" &
  i=$((i+1)); sleep 20
done
wait
echo EXORL_1MRECIPE_DONE
