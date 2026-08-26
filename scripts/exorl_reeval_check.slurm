#!/bin/bash
# Verify the zero-shot re-eval route: score a T3 checkpoint on a reward task it was NOT
# relabelled for. walker seed 0 trained under `walk`; this scores `run` and `stand` off the
# SAME params, which is the property T3 claims.
#SBATCH -J exorl-reeval
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
export EXORL_DATA_ROOT="$SCRATCH/exorl_data/exorl/datasets"
export EXORL_CACHE_ROOT="$SCRATCH/exorl_data/relabel_cache"
RUN="$SCRATCH/factored-fb-exorl/t3/dmc-walker/density_fb/dmc-walker_density_fb_flowbc_seed_0"
OUT="$SCRATCH/factored-fb-exorl/t3_zeroshot.jsonl"
fail=0
for T in walk run stand flip; do
  echo "---- walker, reward task $T (params trained under walk) ----"
  $PY -m scripts.reeval_checkpoint --critic density_fb --actor flowbc --domain dmc-walker \
     --seed 0 --restore_path "$RUN" --restore_epoch 100000 --eval_episodes 10 \
     --infer_cond_episodes 50000 --eval_on_inferred_cond \
     --override measure_density_z=true --override 'measure_hit_spec="0:24:15.9"' \
     --override exorl_task=$T --override exorl_max_episodes=500 \
     --tag "zeroshot-$T" --out_jsonl "$OUT" \
    && echo "PASS $T" || { echo "FAIL $T"; fail=$((fail+1)); }
done
echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail FAILED" ) =="
echo "---- rows written ----"; tail -4 "$OUT" 2>/dev/null
exit $fail
