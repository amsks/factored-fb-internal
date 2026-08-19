#!/bin/bash
# Re-evaluate the E2 ms-pick-cube checkpoints (BC/GCIQL/biGCIQL masked rows x 3 seeds, plus
# the BC-nomask control x 3 seeds -- 12 checkpoints total) against the FIXED synth-goal-frame
# construction (env/unified.py::synth_goal_frame), with NO retraining: same params_300000.pkl,
# only the eval-time goal frame changes.
#
# Also runs the unit tests (including the new tcp-offset-preservation assertions in
# tests/env/test_e0_infra.py) first, since JAX can't run on the Vista login node.
#
#   sbatch -A ASC26008 scripts/reeval_pick_cube_synth_fix.slurm
#
#SBATCH -J reeval-pick-cube-synth
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail                 # NOT -e: run every checkpoint, then summarise
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

VENV_MS="${VENV_MS:-$SCRATCH/.venv-ms}"
PY="$VENV_MS/bin/python"
E2_ROOT="${E2_ROOT:-$SCRATCH/factored-fb-E2/ms-pick-cube}"
MS_ROOT="${MS_ROOT:-$SCRATCH/fb-data/ms-play-resample}"
OUT_DIR="${OUT_DIR:-$SCRATCH/factored-fb-E2-reeval}"
OUT_JSONL="$OUT_DIR/ms-pick-cube-synth-fix.jsonl"
EPISODES="${EPISODES:-50}"
EPOCH="${EPOCH:-300000}"

mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$SCRATCH/vulkan/nvidia_icd.json}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export WANDB_MODE=disabled
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FB_DATA_ROOT="$MS_ROOT"

echo "== host: $(hostname)  job ${SLURM_JOB_ID:-none}  episodes=$EPISODES  out=$OUT_DIR =="
fail=0

# ── 1. Unit tests -- the tcp-offset-preservation fix, and everything else in tests/env ──────
echo "---- pytest tests/env ----"
if "$PY" -m pytest tests/env -q > logs/reeval-pytest.log 2>&1; then
  echo "PASS pytest tests/env"
else
  echo "FAIL pytest tests/env -> logs/reeval-pytest.log"; tail -40 logs/reeval-pytest.log
  fail=$((fail + 1))
fi

# ── 2. Re-eval every checkpoint with the fixed synth frame ──────────────────────────────────
MASK="obs_mask_dims=[26,27,28,39,40,41] goal_mask_dims=[26,27,28,39,40,41]"
NOGR="value_p_randomgoal=0.0 value_p_trajgoal=0.8"

reeval() {  # reeval <tag> <critic> <actor> <sd_subdir> <seed> <extra_override...>
  local tag=$1 critic=$2 actor=$3 subdir=$4 seed=$5; shift 5
  local sd="$E2_ROOT/$subdir"
  local run_dir="$sd/ms-pick-cube_${critic}_${actor}_seed_${seed}"
  if [ ! -f "$run_dir/params_${EPOCH}.pkl" ]; then
    echo "FAIL $tag-s$seed: no checkpoint at $run_dir/params_${EPOCH}.pkl"
    fail=$((fail + 1)); return
  fi
  local ov=() kv
  for kv in "$@"; do ov+=(--override "$kv"); done
  local log="logs/reeval-$tag-s$seed.log"
  echo "---- $tag seed $seed ----"
  if "$PY" scripts/reeval_checkpoint.py \
      --critic "$critic" --actor "$actor" --domain ms-pick-cube --seed "$seed" \
      --restore_path "$run_dir" --restore_epoch "$EPOCH" --eval_episodes "$EPISODES" \
      --tag "$tag-s$seed" --out_jsonl "$OUT_JSONL" \
      "${ov[@]}" > "$log" 2>&1; then
    echo "PASS $tag-s$seed"; grep REEVAL "$log"
  else
    echo "FAIL $tag-s$seed -> $log"; tail -40 "$log"
    fail=$((fail + 1))
  fi
}

for SEED in 0 1 2; do
  reeval BC       gciql        bcflow BC       "$SEED" $MASK $NOGR eval_goal_mode=synth
  reeval GCIQL    gciql        flowbc GCIQL    "$SEED" $MASK $NOGR eval_goal_mode=synth
  reeval biGCIQL  iql_bilinear flowbc biGCIQL  "$SEED" value_latent_dim=50 $MASK $NOGR eval_goal_mode=synth
  # Unmasked control: no mask, but force the NEW synth frame (originally ran eval_goal_mode=
  # fixed since the override was empty -- see config.json). Sanity: an unmasked policy already
  # sees the true target straight off the observation, so the eval-time goal frame it also
  # eval-conditions on should not move its success meaningfully.
  reeval BCnomask gciql        bcflow BC-nomask "$SEED" eval_goal_mode=synth
done

echo "== reeval done: $fail failure(s).  results -> $OUT_JSONL =="
exit $fail
