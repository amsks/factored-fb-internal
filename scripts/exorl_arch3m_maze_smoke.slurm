#!/bin/bash
# 200-step smoke of the ONLY new code paths in launch_exorl_arch3m.sh: point-mass-maze at
# the motivo arch-shape config (density_fb) and at the fb reference shapes, z_dim 100,
# discount 0.99. Override strings are the launcher's DRYRUN output verbatim. Walker/cheetah
# are not smoked here: their override sets are byte-identical to the completed arch-shape arm.
#SBATCH -J exorl-arch3m-maze-smoke
#SBATCH -A ASC26008
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export EXORL_DATA_ROOT="$SCRATCH/exorl_data/exorl/datasets" EXORL_CACHE_ROOT="$SCRATCH/exorl_data/relabel_cache"
fail=0
run () { # run <critic> <actor> <ov...>
  local C=$1 A=$2; shift 2
  echo "---- $C x $A on dmc-point-mass-maze (arch3m smoke) ----"
  $PY -m impls.main --critic "$C" --actor "$A" --domain dmc-point-mass-maze --seed 0 \
     --train_steps 200 --log_interval 100 --eval_interval 200 --save_interval 200 \
     --eval_episodes 2 --num_video 0 --save_dir "$SCRATCH/exorl-arch3m-maze-smoke/$C" \
     --infer_cond_episodes 10000 --eval_on_inferred_cond \
     $(for kv in "$@"; do echo --override "$kv"; done) \
     --override exorl_max_episodes=100 \
     > "logs/arch3m-smoke-$C.log" 2>&1 \
    && echo "PASS $C" || { echo "FAIL $C"; tail -25 "logs/arch3m-smoke-$C.log"; fail=$((fail+1)); }
}
COMMON="actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3 actor.alpha=0.0 batch_size=1024 \
lr_actor=1.0e-4 discount=0.99 actor.hidden_dims=[1024]"
run density_fb ddpgbc_density $COMMON tau=0.01 measure_density_z=true \
    'measure_hit_spec="0:2:0.03"' measure_density_dim=100 value_hidden_dims=[1024] \
    measure_density_b_hidden_dims=[256,256] measure_density_archi=motivo \
    actor.motivo_archi=true measure_density_b_norm=true lr=1.0e-4 lr_f=1.0e-4 lr_b=1.0e-4
run fb ddpgbc $COMMON f_target_tau=0.01 b_target_tau=0.01 z_dim=100 L_dim=100 \
    forward.hidden_dim=1024 forward.hidden_layers=1 \
    left_encoder.hidden_dim=1024 left_encoder.hidden_layers=1 \
    backward.hidden_dim=256 backward.hidden_layers=2
echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail FAILED" ) =="
exit $fail
