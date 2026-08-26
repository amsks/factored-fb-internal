#!/bin/bash
#SBATCH -J exorl-arch-smoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export EXORL_DATA_ROOT="$SCRATCH/exorl_data/exorl/datasets" EXORL_CACHE_ROOT="$SCRATCH/exorl_data/relabel_cache"
OPT="actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3 actor.alpha=0.0 batch_size=1024 discount=0.98 lr_actor=1.0e-4 actor.hidden_dims=[1024]"
fail=0
run () { # run <critic> <actor> <domain> <extra ov...>
  local C=$1 A=$2 D=$3; shift 3
  echo "---- $C x $A on $D (reference architecture) ----"
  $PY -m impls.main --critic $C --actor $A --domain $D --seed 0 \
     --train_steps 200 --log_interval 100 --eval_interval 200 --save_interval 200 \
     --eval_episodes 2 --num_video 0 --save_dir "$SCRATCH/exorl-arch-smoke/$C-$D" \
     --infer_cond_episodes 10000 --eval_on_inferred_cond \
     $(for kv in $OPT "$@"; do echo --override "$kv"; done) \
     --override exorl_max_episodes=100 \
     > "logs/exorl-arch-$C-${D#dmc-}.log" 2>&1 \
    && echo "PASS $C $D" || { echo "FAIL $C $D"; tail -25 "logs/exorl-arch-$C-${D#dmc-}.log"; fail=$((fail+1)); }
}
FBARCH_C="f_target_tau=0.01 b_target_tau=0.01 z_dim=50 L_dim=50 forward.hidden_dim=1024 forward.hidden_layers=1 left_encoder.hidden_dim=1024 left_encoder.hidden_layers=1 backward.hidden_dim=256 backward.hidden_layers=2"
FBARCH_W="${FBARCH_C/z_dim=50 L_dim=50/z_dim=100 L_dim=100}"
run fb ddpgbc dmc-cheetah $FBARCH_C
run fb ddpgbc dmc-walker  $FBARCH_W
run density_fb ddpgbc_density dmc-cheetah tau=0.01 measure_density_z=true 'measure_hit_spec="0:17:7.5"' measure_density_dim=50 value_hidden_dims=[1024]
run density_fb ddpgbc_density dmc-walker  tau=0.01 measure_density_z=true 'measure_hit_spec="0:24:15.9"' measure_density_dim=100 value_hidden_dims=[1024]
echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail FAILED" ) =="
exit $fail
