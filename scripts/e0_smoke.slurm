#!/bin/bash
# E0 gate for docs/specs/2026-08-10-deconfound-size-coverage.md.
#
# Nothing in E1/E2/E3 may be launched until this passes: it exercises the three pieces of
# shared infrastructure E0 adds (synth per-episode eval goals, per-episode goal-distance
# logging, the cube size-ladder truncation knob) as unit tests AND as 200-step real runs
# through impls.main, on both venvs.
#
#   sbatch -A ASC26008 scripts/e0_smoke.slurm
#   STEPS=400 sbatch -A ASC26008 scripts/e0_smoke.slurm
#
# Login nodes cannot run this (JAX's Eigen pool exceeds Vista's ~100 thread/user cap).
#
#SBATCH -J e0-smoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail                 # NOT -e: attempt every check, then summarise
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

VENV_JAX="${VENV_JAX:-$SCRATCH/.venv-jax-fb}"
VENV_MS="${VENV_MS:-$SCRATCH/.venv-ms}"
STEPS="${STEPS:-200}"
OUT="${OUT:-$SCRATCH/factored-fb-e0-smoke}"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$SCRATCH/vulkan/nvidia_icd.json}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export WANDB_MODE=disabled
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MW_ROOT="${MW_ROOT:-$SCRATCH/fb-data/play-resample}"           # v3, 2000 x 500
MS_ROOT="${MS_ROOT:-$SCRATCH/fb-data/upstream-play-resample-packed}"

echo "== host: $(hostname)  job ${SLURM_JOB_ID:-none}  steps=$STEPS  out=$OUT =="
fail=0
pass() { echo "PASS $*"; }
bad()  { echo "FAIL $*"; fail=$((fail + 1)); }

run() {  # run <tag> <python> <args...>
  local tag=$1; shift
  local log="logs/e0-$tag.log"
  echo "---- $tag ----"
  if "$@" > "$log" 2>&1; then pass "$tag"; else bad "$tag -> $log"; tail -30 "$log"; fi
}

# ── 1. Unit tests: synth frames, goal distance, truncation ───────────────────────────────
run pytest "$VENV_JAX/bin/python" -m pytest tests/env -q

# ── 2. E0.3 — cube truncation on the REAL dataset ────────────────────────────────────────
# The full-size no-op has to be checked against the real 1,001,000-row cube-single split, not
# just the synthetic fixture: a no-op there is what makes the existing full-1M results a
# valid anchor for the ladder.
echo "---- cube truncation (real data) ----"
if "$VENV_JAX/bin/python" - > logs/e0-cube-trunc.log 2>&1 <<'PY'; then
import numpy as np
from env.ogbench import truncate_whole_trajectories
from impls.utils.env_utils import make_env_and_datasets

_, train_ds, val_ds = make_env_and_datasets("cube-single-play-v0")
n = len(train_ds["observations"])
print("cube-single-play-v0 train rows:", n, "val rows:", len(val_ds["observations"]))

full = truncate_whole_trajectories(train_ds, n)
for k in train_ds:
    assert np.array_equal(np.asarray(full[k]), np.asarray(train_ds[k])), f"{k}: not a no-op"
print("full-size no-op: OK (bit-for-bit on every field)")

prev = None
for want in (48_000, 100_000, 250_000):
    sub = truncate_whole_trajectories(train_ds, want)
    got = len(sub["observations"])
    ends = np.nonzero(np.asarray(sub["valids"]) == 0)[0]
    assert got >= want, (got, want)
    assert ends[-1] == got - 1 and np.asarray(sub["terminals"])[-1] > 0
    assert np.array_equal(np.asarray(sub["observations"]),
                          np.asarray(truncate_whole_trajectories(train_ds, want)["observations"])), \
        "not deterministic"
    ids = set(map(tuple, np.asarray(sub["observations"])[ends].tolist()))
    if prev is not None:
        assert prev <= ids, f"{want} does not nest the previous rung"
    prev = ids
    print(f"  {want:>7} -> {got} rows, {len(ends)} trajectories, ends on a boundary")
print("ladder: sizes met, whole trajectories, deterministic, nested")
PY
  pass "cube-truncation-real"; grep -E "rows|OK|ladder" logs/e0-cube-trunc.log
else
  bad "cube-truncation-real -> logs/e0-cube-trunc.log"; tail -30 logs/e0-cube-trunc.log
fi

# ── 3. E0.1/E0.2 — 200-step runs through impls.main ──────────────────────────────────────
train() {  # train <tag> <venv> <domain> <data_root> <override...>
  local tag=$1 venv=$2 domain=$3 root=$4; shift 4
  local ov=() kv
  for kv in "$@"; do ov+=(--override "$kv"); done
  export FB_DATA_ROOT="$root"
  run "$tag" "$venv/bin/python" -m impls.main \
      --critic iql_bilinear --actor flowbc --domain "$domain" --seed 0 \
      --train_steps "$STEPS" --log_interval "$STEPS" --eval_interval "$STEPS" \
      --save_interval "$STEPS" --eval_episodes 4 --num_video 0 \
      "${ov[@]}" --save_dir "$OUT/$tag"
}

# MW: the E3 recipe. obs_mask_dims is the knob that closes the BC shortcut; goal_mask_dims
# alone blanks only the emitted goal.
# p_randomgoal=0 is Arm C's spelling: the value goal probabilities must sum to 1
# (GCDataset asserts it), so dropping randomgoal moves its mass to trajgoal. The actor's
# randomgoal is already 0 in every critic config.
NOGR="value_p_randomgoal=0.0 value_p_trajgoal=0.8"
MWOV=(obs_mask_dims=[36,37,38] goal_mask_dims=[36,37,38] value_latent_dim=50 $NOGR)
train mw-push-synth "$VENV_JAX" mw-push "$MW_ROOT" "${MWOV[@]}" eval_goal_mode=synth
train mw-push-fixed "$VENV_JAX" mw-push "$MW_ROOT" "${MWOV[@]}" eval_goal_mode=fixed

# Cube: the E1 recipe. No masking, no synth (the OGBench eval path draws its own goals).
train cube-48k "$VENV_JAX" cube-single "" value_latent_dim=50 dataset_num_transitions=48000

# ManiSkill: the E2 recipe, on the 48k upstream data purely as a plumbing check. NOTE the
# mask dims -- ms-pick-cube's obs carries obj_to_goal_pos at 39:42, so masking 26:29 alone
# leaves the goal exactly reconstructible; see docs/specs/.../findings.
if [ -x "$VENV_MS/bin/python" ]; then
  train ms-push-synth "$VENV_MS" ms-push-cube "$MS_ROOT" \
      obs_mask_dims=[25,26,27] goal_mask_dims=[25,26,27] value_latent_dim=50 \
      $NOGR eval_goal_mode=synth
  train ms-pick-synth "$VENV_MS" ms-pick-cube "$MS_ROOT" \
      obs_mask_dims=[26,27,28,39,40,41] goal_mask_dims=[26,27,28,39,40,41] \
      value_latent_dim=50 $NOGR eval_goal_mode=synth
else
  echo "SKIP maniskill runs: no $VENV_MS"
fi

# ── 4. Did E0.2 actually write per-episode distances? ────────────────────────────────────
echo "---- per-episode distance logs ----"
for tag in mw-push-synth mw-push-fixed ms-push-synth ms-pick-synth; do
  [ -d "$OUT/$tag" ] || continue
  f=$(find "$OUT/$tag" -name eval_episodes.jsonl | head -1)
  if [ -n "$f" ] && [ -s "$f" ]; then
    pass "eval_episodes.jsonl $tag  ($(wc -l < "$f") rows)"; head -1 "$f" | cut -c1-200
  else
    bad "eval_episodes.jsonl missing/empty for $tag"
  fi
done
# Cube goes through the OGBench adapter, which has no goal-in-obs layout: it must NOT write one.
if [ -d "$OUT/cube-48k" ] && [ -n "$(find "$OUT/cube-48k" -name eval_episodes.jsonl)" ]; then
  bad "cube wrote eval_episodes.jsonl (the ogbench path should not)"
else
  pass "cube wrote no per-episode log (correct: no goal-in-obs layout)"
fi

echo "== e0 smoke done: $fail failure(s) =="
exit $fail
