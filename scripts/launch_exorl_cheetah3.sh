#!/usr/bin/env bash
# Cheetah round 4 (2026-09-20): three arms on top of the round-3 `sqt` recipe (scripts/launch_exorl_sqt.sh:
# 1M recipe + sphere-only task vectors + actor Q normaliser + value tower [1024] + tau 0.001; cheetah 5-seed
# 431.0 +- 82.0, seed 4 = 286), 8 jobs, ONE per GPU so 1M steps land in ~3.5 h.
#   velblock  hit kernel on the VELOCITY block only (obs cols 8:17, hit-rate radius 7.264 from
#             scripts/calibrate_hit_spec.py; the shipped spec 0:17:7.5 spans position+velocity)   seeds 0 1 2
#   speedcol  hit kernel on the forward-speed column only (obs col 8, radius 0.021, support 29/1024)  seeds 0 1
#   tau0025   round-3 recipe with the target rate at 0.0025 instead of 0.001 (stability of seed 4)  seeds 0 1 2
# Cheetah obs layout (dm_control cheetah.get_observation): position = qpos[1:] (8) then velocity (9) => 17.
#   usage: bash scripts/launch_exorl_cheetah3.sh            # tmux session exorl_cheetah3
# Score:  bash scripts/score_exorl_cheetah3.sh [STEP]       # paper protocol, default params_1000000
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
ENV="FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl"
PY=/dev/shm/.venv-jax/bin/python
ACT="--override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 --override actor.q_normalize=false"
COMMON="--auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --eval_episodes 10 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond"
DFB="--override 'value_hidden_dims=[512,512,512]' --override tau=0.005 --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo --override measure_density_b_norm=true --override lr=1e-4"
DOMAIN="--domain dmc-cheetah --override discount=0.98 --override 'measure_hit_spec=\"0:17:7.5\"' --override lr_actor=1e-4"
SQT="--override measure_density_z_rand_frac=1.0 --override actor.q_normalize=true --override 'value_hidden_dims=[1024]' --override tau=0.001"
# arm overrides go LAST so they win the dotlist merge
ARM_velblock="--override 'measure_hit_spec=\"8:17:7.264\"'"
ARM_speedcol="--override 'measure_hit_spec=\"8:9:0.021\"'"
ARM_tau0025="--override tau=0.0025"
if [ -n "${JOBS_OVERRIDE:-}" ]; then eval "JOBS=($JOBS_OVERRIDE)"; else
  JOBS=("velblock 0" "velblock 1" "velblock 2" "speedcol 0" "speedcol 1" "tau0025 0" "tau0025 1" "tau0025 2"); fi

cmd() {   # cmd <arm> <seed> <save_dir>
  local arm=$1 seed=$2 save=$3 armflags
  armflags=$(eval echo "\"\$ARM_$arm\"")
  echo "$ENV $PY -m impls.main --critic density_fb_z --actor ddpgbc $COMMON --save_dir $save $DFB $ACT $DOMAIN --seed $seed $SQT $armflags"
}

mkdir -p logs/exorl_cheetah3
SESSION=${SESSION_NAME:-exorl_cheetah3}
tmux new-session -d -s "$SESSION" -n init "sleep 1"
i=0
for J in "${JOBS[@]}"; do
  read -r arm s <<< "$J"
  log=logs/exorl_cheetah3/${arm}_s${s}.log
  tmux new-window -t "$SESSION" -n "${arm}_s${s}" \
    "cd $HOME/git/Factored-FB-seam && CUDA_VISIBLE_DEVICES=$(( (i + ${GPU0:-0}) % 8 )) $(cmd $arm $s exp/exorl_cheetah3/$arm) > $log 2>&1; echo JOB_DONE rc=\$? >> $log; sleep 86400"
  i=$((i+1)); sleep 15
done
tmux kill-window -t "$SESSION:init" || true
echo "launched $i jobs in tmux session $SESSION"
