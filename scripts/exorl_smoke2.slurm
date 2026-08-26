#!/bin/bash
# ExoRL bring-up smoke, round 2: the settings T3/T4 will actually launch with.
#   - walker / cheetah for the first time, on the DECIDED full-observation hit specs
#   - MetaMotivo's inference-sample count (50k, vs the 256 round 1 used)
#   - 5000-episode (5M transition) datasets, i.e. the real relabel + load cost
#
#   sbatch -A ASC26008 scripts/exorl_smoke2.slurm
#
#SBATCH -J exorl-smoke2
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export EXORL_DATA_ROOT="${EXORL_DATA_ROOT:-$SCRATCH/exorl_data/exorl/datasets}"
export EXORL_CACHE_ROOT="${EXORL_CACHE_ROOT:-$SCRATCH/exorl_data/relabel_cache}"
OUT="$SCRATCH/exorl-smoke2"
M=50000        # MetaMotivo's num_inference_samples (examples/fb_train_dmc.py)
fail=0
echo "== host $(hostname) == $(date -u +%FT%TZ)"

run () {  # run <label> <domain> <hitspec> <extra...>
  local label=$1 dom=$2 hit=$3; shift 3
  echo "---- $label ----"
  $PY -m impls.main --critic density_fb --actor flowbc --domain "$dom" --seed 0 \
     --train_steps 300 --log_interval 150 --eval_interval 300 --save_interval 300 \
     --eval_episodes 3 --save_dir "$OUT/$label" \
     --infer_cond_episodes $M --eval_on_inferred_cond \
     --override "measure_hit_spec=\"$hit\"" --override measure_density_z=true "$@" \
     > "logs/exorl-smoke2-$label.log" 2>&1 \
    && echo "PASS $label" || { echo "FAIL $label -> logs/exorl-smoke2-$label.log"; tail -40 "logs/exorl-smoke2-$label.log"; fail=$((fail+1)); }
}

run pmm     dmc-point-mass-maze "0:2:0.03"
run walker  dmc-walker          "0:24:15.9"
run cheetah dmc-cheetah         "0:17:7.5"

echo "---- online, walker (locomotion + real env + the new hit spec) ----"
$PY -m impls.main --critic density_fb --actor flowbc --domain dmc-walker --seed 0 \
   --online --online_start_steps 2000 --online_buffer_size 20000 \
   --train_steps 300 --log_interval 150 --eval_interval 300 --save_interval 300 \
   --eval_episodes 3 --save_dir "$OUT/online-walker" \
   --override 'measure_hit_spec="0:24:15.9"' --override exorl_max_episodes=20 \
   > logs/exorl-smoke2-online-walker.log 2>&1 \
  && echo "PASS online-walker" || { echo "FAIL online-walker -> logs/exorl-smoke2-online-walker.log"; tail -50 logs/exorl-smoke2-online-walker.log; fail=$((fail+1)); }

echo "== $( [ $fail -eq 0 ] && echo ALL PASS || echo "$fail STAGE(S) FAILED" ) == $(date -u +%FT%TZ)"
exit $fail
