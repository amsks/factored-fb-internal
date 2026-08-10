#!/usr/bin/env bash
# Spec-18 audit follow-ups (2026-08-10):
#   1. Regenerate the four cube-single zero-shot evals at pool_size=16384 -- the shipped
#      dzs_* JSONs were decoded at P=4096, which the G3 sweep showed drifts up to .12/task.
#      The eval is deterministic, so these are exact replacements, suffixed _P16384.
#   2. Produce the missing G3 full-rollout artifact at 32768 (the tex's "converged by
#      16384" claim currently has no rollout artifact behind it; 65536 OOMs, 32768 is
#      inside the ceiling).
#   3. Resume scene d18x_sc_dens_s1 from params_400000.pkl for its final 100k steps
#      (died between 450k and 500k; run_gciql.py restore passthrough appends eval.csv
#      with the correct step offset).
# Run: tmux new -d -s s18rerun 'scripts/rescuing_fb/rerun_spec18_evals_p16384.sh 2>&1 | tee /dev/shm/spec18_logs/rerun_dzs16k.log'
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python              # run_gciql.py orchestrator (re-execs into the jax venv itself)
JPY=/dev/shm/.venv-jax/bin/python       # eval scripts import jax directly -- must use the jax venv
EV=scripts/rescuing_fb/eval_density_zeroshot.py
SKIP_SCENE=${SKIP_SCENE:-0}
ROOT=/home/adityamohan/exp_spec18/factored-fb
OUT=results/spec18_eval
LOG=/dev/shm/spec18_logs; mkdir -p "$LOG"

envrun() { # gpu, then command...
  local g=$1; shift
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g JAX_PLATFORMS=${JAXPLAT:-} "$@"
}

echo "[rerun] $(date +%H:%M:%S) start"

# --- 1. dzs at P=16384 (protocol otherwise identical to the shipped files:
#        ckpt 500000, 25 episodes, seed 0, relabel 50000, retrieval k=1) ---
envrun "" "$JPY" "$EV" --run-dir "$ROOT/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s0" \
  --checkpoint-step 500000 --episodes 25 --seed 0 --retrieval-k 1 --pool_size 16384 \
  --out "$OUT/dzs_d18_cs_dens_s0_RET_k1_P16384.json" > "$LOG/dzs16k_s0.log" 2>&1 &
envrun "" "$JPY" "$EV" --run-dir "$ROOT/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s1" \
  --checkpoint-step 500000 --episodes 25 --seed 0 --retrieval-k 1 --pool_size 16384 \
  --out "$OUT/dzs_d18_cs_dens_s1_RET_k1_P16384.json" > "$LOG/dzs16k_s1.log" 2>&1 &
envrun "" "$JPY" "$EV" --run-dir "$ROOT/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s2" \
  --checkpoint-step 500000 --episodes 25 --seed 0 --retrieval-k 1 --pool_size 16384 \
  --out "$OUT/dzs_d18_cs_dens_s2_RET_k1_P16384.json" > "$LOG/dzs16k_s2.log" 2>&1 &
envrun "" "$JPY" "$EV" --run-dir "$ROOT/GCIQL-CubeSingle-Flow-Measure-DensityHard-d50/d18_cs_hard_s0" \
  --checkpoint-step 500000 --episodes 25 --seed 0 --retrieval-k 1 --pool_size 16384 \
  --out "$OUT/dzs_d18_cs_hard_s0_RET_k1_P16384.json" > "$LOG/dzs16k_hard.log" 2>&1 &

# --- 2. G3 full-rollout artifact incl. 32768 (dens_s2, the seed the G3 study used) ---
envrun "" "$JPY" "$EV" --run-dir "$ROOT/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s2" \
  --checkpoint-step 500000 --episodes 25 --seed 0 --retrieval-k 1 --pool_size 16384 \
  --pool_sweep "4096,16384,32768" --pool_sweep_full_rollout \
  --out "$OUT/G3_dens_s2_P32768_fullrollout.json" > "$LOG/g3_32768.log" 2>&1 &

# --- 3. scene s1: final 100k from the 400k checkpoint ---
[ "$SKIP_SCENE" = "1" ] && SCENE_SKIPPED=1
SC=/home/adityamohan/exp_spec18x/factored-fb/GCIQL-Scene-Flow-Measure-Density-d50/d18x_sc_dens_s1
[ "${SCENE_SKIPPED:-0}" = "1" ] || envrun 6 "$PY" run_gciql.py --config-name scene_state_measure_density_d50 seed=1 \
  output_root=/home/adityamohan/exp_spec18x exp_name=d18x_sc_dens_s1 \
  ++restore_path="$SC" ++restore_epoch=400000 ++train_steps=100000 \
  > "$LOG/sc_s1_resume.log" 2>&1 &

wait
echo "[rerun] $(date +%H:%M:%S) all jobs finished"
ls -la "$OUT" | grep -E 'P16384|P32768'
tail -1 "$SC/eval.csv" | cut -c1-100
