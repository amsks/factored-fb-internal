#!/bin/bash
# W4 / KT-A of docs/specs/2026-08-17-gciql-edge-attribution.md: is kt1-qc03 seed 2's 0.02 at
# step 350k real per-checkpoint instability, or 50-episode eval variance?
#
#   sbatch -A ASC26008 scripts/reeval_kt1_qc03.slurm
#
# SUBSTITUTION: the spec names params_350000.pkl, which DOES NOT EXIST. That run evaluated
# every 50k but checkpointed every 100k, so the 350k eval (0.02) has no saved params. The
# checkpoints that do exist bracket it: 300k (evaled 0.36) and 400k (evaled 0.80). Those are
# what this job re-scores, and they answer the same question — seed 2's curve reads
# 0.36 / 0.02 / 0.80 at 300k / 350k / 400k, so if 300k's 0.36 moves substantially under a
# fresh episode draw, a 50-episode eval is not a reliable point estimate here and the 0.02
# needs no separate explanation. (New arms launched from scripts/launch_edge_attribution.sh
# use SAVE_INTERVAL=50000 so this gap cannot recur.)
#
# The discriminator is the eval SEED OFFSET. Episode seeds are the loop index, so a plain
# re-eval reproduces the original number exactly and proves nothing; `eval_seed_offset` shifts
# the block onto a disjoint set of initial states. Each checkpoint is scored twice: at offset
# 0 (must reproduce the training-time number — that is the control) and at offset 1000 (a
# fresh draw).
#
#   each checkpoint reproduces its training-time score at BOTH offsets -> the curve is real
#     per-checkpoint instability; R4's peak/final disclosure is mandatory.
#   offset 1000 moves a checkpoint substantially -> 50-episode eval variance dominates; the
#     "seed-2 collapse" is largely a measurement artefact and the arm is steadier than the
#     curve suggests.
#
# Login nodes cannot run JAX (pthread limits), hence a job.
#
#SBATCH -J reeval-kt1-qc03
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs

PY="${VENV:-$SCRATCH/.venv-jax-fb}/bin/python"
RUN="${RUN:-$SCRATCH/factored-fb-density-port/kt1-qc03/mw-pick-place_density_fb_flowbc_seed_2}"
OUT_DIR="${OUT_DIR:-$SCRATCH/factored-fb-edge-attr}"
OUT_JSONL="$OUT_DIR/kt1-qc03-s2-forensics.jsonl"
EPISODES="${EPISODES:-50}"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FB_DATA_ROOT="${FB_DATA_ROOT:-$SCRATCH/fb-data/play-resample}"

echo "== host $(hostname)  run=$RUN =="
ls "$RUN"/params_*.pkl 2>/dev/null | sed 's/^/  /' || { echo "no checkpoints under $RUN"; exit 1; }

fail=0
for EPOCH in 300000 400000; do
  for OFF in 0 1000; do
    tag="qc03-s2-e${EPOCH}-off${OFF}"
    echo "---- $tag ----"
    if $PY scripts/reeval_checkpoint.py \
        --critic density_fb --actor flowbc --domain mw-pick-place --seed 2 \
        --restore_path "$RUN" --restore_epoch "$EPOCH" --eval_episodes "$EPISODES" \
        --tag "$tag" --out_jsonl "$OUT_JSONL" \
        --override measure_density_hard=true \
        --override 'measure_hit_spec="4:7:0.07"' \
        --override actor.q_coeff=0.3 \
        --override obs_mask_dims=[36,37,38] --override goal_mask_dims=[36,37,38] \
        --override value_p_randomgoal=0.0 --override value_p_trajgoal=0.8 \
        --override eval_goal_mode=synth \
        --override eval_seed_offset=$OFF \
        > "logs/reeval-$tag.log" 2>&1; then
      echo "PASS $tag"; grep REEVAL "logs/reeval-$tag.log"
    else
      echo "FAIL $tag -> logs/reeval-$tag.log"; tail -30 "logs/reeval-$tag.log"; fail=$((fail+1))
    fi
  done
done

echo
echo "== forensics done: $fail failure(s).  results -> $OUT_JSONL =="
echo "== read: offset 0 must reproduce the training-time eval; offset 1000 is the fresh draw =="
exit $fail
