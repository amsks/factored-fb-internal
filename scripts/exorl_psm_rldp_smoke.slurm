#!/bin/bash
# psm / rldp have loss-level parity and had never been RUN on an ExoRL domain. 18 T3 jobs
# ride on that, so smoke both before launching them.
#SBATCH -J exorl-psmrldp
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:50:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export EXORL_DATA_ROOT="${EXORL_DATA_ROOT:-$SCRATCH/exorl_data/exorl/datasets}"
export EXORL_CACHE_ROOT="${EXORL_CACHE_ROOT:-$SCRATCH/exorl_data/relabel_cache}"
fail=0
for CRITIC in psm rldp; do
  for DOM in dmc-point-mass-maze dmc-walker; do
    echo "---- $CRITIC on $DOM ----"
    $PY -m impls.main --critic $CRITIC --actor flowbc --domain $DOM --seed 0 \
       --train_steps 300 --log_interval 150 --eval_interval 300 --save_interval 300 \
       --eval_episodes 3 --num_video 0 --save_dir "$SCRATCH/exorl-psmrldp/$CRITIC-$DOM" \
       --infer_cond_episodes 50000 --eval_on_inferred_cond \
       --override exorl_max_episodes=100 \
       > "logs/exorl-psmrldp-$CRITIC-${DOM#dmc-}.log" 2>&1 \
      && echo "PASS $CRITIC $DOM" \
      || { echo "FAIL $CRITIC $DOM"; tail -30 "logs/exorl-psmrldp-$CRITIC-${DOM#dmc-}.log"; fail=$((fail+1)); }
  done
done
echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail FAILED" ) =="
exit $fail
