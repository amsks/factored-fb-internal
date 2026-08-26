#!/bin/bash
# Smoke the PAPER-EXACT ExoRL config before spending 12 runs x 1M steps on it.
# Touati et al. Appendix G.5 Table 1 (walker/cheetah column): batch 1024, discount 0.98,
# target momentum 0.99 (tau 0.01), lr 1e-4, policy smoothing stddev 0.2 truncated at 0.3,
# 1e6 gradient steps, 5e6 replay buffer (= our 5000 episodes), representation dim 50.
# The only NEW code is target policy smoothing (impls/actors/ddpgbc.py::_smooth), so this
# also checks it runs and that std=0 still means "no draw".
#SBATCH -J exorl-paper-smoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export EXORL_DATA_ROOT="$SCRATCH/exorl_data/exorl/datasets"
export EXORL_CACHE_ROOT="$SCRATCH/exorl_data/relabel_cache"
fail=0
echo "---- 1. bit-identity: smoothing OFF must not move any existing loss ----"
$PY -m pytest tests/impls/test_boot_contract.py -q -p no:cacheprovider > logs/exorl-paper-boot.log 2>&1 \
  && echo "PASS boot contract (std=0 draws nothing)" \
  || { echo "FAIL boot contract"; tail -25 logs/exorl-paper-boot.log; fail=$((fail+1)); }

run () {  # run <critic> <actor> <domain> <hit>
  local C=$1 A=$2 D=$3 HIT=${4:-}
  local OV="actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3 actor.alpha=0.0 \
            batch_size=1024 discount=0.98 lr_actor=1.0e-4"
  case "$C" in
    fb)         OV="$OV f_target_tau=0.01 b_target_tau=0.01" ;;
    density_fb) OV="$OV tau=0.01 measure_density_z=true measure_hit_spec=\"$HIT\"" ;;
  esac
  echo "---- $C x $A on $D (paper config) ----"
  $PY -m impls.main --critic $C --actor $A --domain $D --seed 0 \
     --train_steps 300 --log_interval 150 --eval_interval 300 --save_interval 300 \
     --eval_episodes 3 --num_video 0 --save_dir "$SCRATCH/exorl-paper-smoke/$C-$D" \
     --infer_cond_episodes 10000 --eval_on_inferred_cond \
     $(for kv in $OV; do echo --override "$kv"; done) \
     --override exorl_max_episodes=100 \
     > "logs/exorl-paper-$C-${D#dmc-}.log" 2>&1 \
    && echo "PASS $C $D" || { echo "FAIL $C $D"; tail -30 "logs/exorl-paper-$C-${D#dmc-}.log"; fail=$((fail+1)); }
}
run fb         ddpgbc         dmc-walker
run fb         ddpgbc         dmc-cheetah
run density_fb ddpgbc_density dmc-walker  "0:24:15.9"
run density_fb ddpgbc_density dmc-cheetah "0:17:7.5"
echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail FAILED" ) =="
exit $fail
