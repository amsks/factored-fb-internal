#!/bin/bash
# Collect the ManiSkill play-resample datasets at cube scale (spec E2,
# docs/specs/2026-08-10-deconfound-size-coverage.md): 1,000,000 transitions per task, against
# the 48,000 the upstream (TommyShen) set has. This is the longest pole in the spec — start it
# before E0 lands, not after.
#
#   sbatch -A ASC26008 scripts/collect_ms_play_resample.slurm
#   TASKS=ms-pick-cube EPISODES=200 sbatch -A ASC26008 -p gh-dev -t 01:00:00 \
#       scripts/collect_ms_play_resample.slurm          # short shakeout first
#
# Both tasks collect in PARALLEL on one node: each is one SAPIEN CPU sim plus a small TD-MPC2
# forward pass, so two fit on a GH200 comfortably and the job finishes in about the time the
# slower one takes rather than the sum.
#
#SBATCH -J fb-collect-ms-pr
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail
mkdir -p "$SLURM_SUBMIT_DIR/logs"
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd)  host: $(hostname)  job ${SLURM_JOB_ID:-none} =="

# The collector needs the GENERATION venv (newt TD-MPC2 + torch + mani_skill), not the
# training venv.
PY="${VENV:-$SCRATCH/.venv-gen}/bin/python"
export MUJOCO_GL=egl
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$SCRATCH/vulkan/nvidia_icd.json}"
export CUDA_VISIBLE_DEVICES=0

OUT="${OUT:-$SCRATCH/fb-data/ms-play-resample}"
TASKS="${TASKS:-ms-pick-cube ms-push-cube}"
SEED="${SEED:-0}"
# Empty => the collector's per-task default, which is the count that reaches 1,000,000
# transitions (pick-cube 8000 x 125, push-cube 20000 x 50 — see TASKS in the collector for
# why the two structures differ).
EPISODES="${EPISODES:-}"

pids=() names=()
for task in $TASKS; do
    args=(--task "$task" --seed "$SEED" --out "$OUT")
    [ -n "$EPISODES" ] && args+=(--episodes "$EPISODES")
    echo "== launching $task =="
    $PY data_collection/collect_ms_play_resample.py "${args[@]}" \
        > "logs/collect-ms-pr-$task.log" 2>&1 &
    pids+=($!); names+=("$task")
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "OK   ${names[$i]}"; tail -3 "logs/collect-ms-pr-${names[$i]}.log"
    else
        echo "FAIL ${names[$i]} -> logs/collect-ms-pr-${names[$i]}.log"
        tail -40 "logs/collect-ms-pr-${names[$i]}.log"; fail=1
    fi
done
[ "$fail" -eq 0 ] || { echo "== a collection failed =="; exit 1; }

ls -la "$OUT/data/maniskill"

# ── Verification gate. The E2 square does NOT launch until this table lines up with cube. ──
echo
echo "== E2 verification gate: structural comparison against cube-single-play =="
"${TRAIN_VENV:-$SCRATCH/.venv-jax-fb}/bin/python" \
    scripts/compare_play_resample_to_ogbench.py \
    --ms-root "$OUT" --mw-root "$SCRATCH/fb-data/play-resample" \
    --json-out "$OUT/structure-vs-cube.json"
echo
echo "== read the gate before launching E2: =="
echo "   * traj_len / eval_horizon        pick-cube must be 5.00 (cube 5.00)"
echo "   * segment length / eval horizon  pick-cube must be ~0.44 (cube 0.435)"
echo "   goal-channel stale transitions   must be 0.000 (upstream was 0.030/0.046)"
echo "   segments reaching the target     must stay high and NOT decay over the trajectory"
echo "   P(satisfied at 1-2 x H)          cube is 0.013; upstream pick-cube was 0.023"
echo "   push-cube is a STRUCTURE-MISMATCHED control by design (1 relocation): its"
echo "   segment/H of 1.00 is expected and must be reported as such, not fixed."
