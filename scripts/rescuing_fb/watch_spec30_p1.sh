#!/usr/bin/env bash
# Spec 30a Item 1 mid-run instrument, ARMED FROM STEP 0: run the action-ranking probe
# (scripts/rescuing_fb/eval_z_action_ranking.py) on every new checkpoint of the two P1
# arms as it lands, and print the one number the KILL GATE reads -- ACTOR SENSITIVITY
# E||pi(s,z) - pi(s,z')|| (noise floor .19). An arm still at ~0 by 250k is killed rather
# than run to 500k. z_cos_spread and the centered-geometry keys are already logged to
# wandb every log interval by the training itself; this covers the actor side, which
# density_w_ztv never sees.
#
# NEVER quote a mid-run SUCCESS number from these files (spec-18 rule: a cell read .984 at
# 450k and .876 at 500k). The probe is CPU-jax only (the 16384-row pool OOMs a shared GPU).
#
# Run in tmux:  tmux new -d -s s30watch1 'scripts/rescuing_fb/watch_spec30_p1.sh'
set -uo pipefail
cd "$(dirname "$0")/../.."

OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec30_p1}
FP=$OUTROOT/factored-fb
RES=${RES:-results/spec30_p1}
PY=/dev/shm/.venv-jax/bin/python
mkdir -p "$RES"
POLL=${POLL:-300}

seen=""
while true; do
  for ck in "$FP"/*/ra2_cs_*/params_*.pkl; do
    [ -e "$ck" ] || continue
    case "$seen" in *"$ck"*) continue;; esac
    rd=$(dirname "$ck"); tag=$(basename "$rd")
    step=$(basename "$ck" .pkl); step=${step#params_}
    out="$RES/p0a_${tag}_${step}.json"
    if [ -f "$out" ]; then seen="$seen $ck"; continue; fi
    echo "[s30watch] $(date +%H:%M:%S) probing $tag @ $step"
    env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        MUJOCO_GL=egl OMP_NUM_THREADS=4 \
      "$PY" scripts/rescuing_fb/eval_z_action_ranking.py \
        --run-dir "$rd" --checkpoint-step "$step" \
        --pool_size 16384 --pairs 64 --actions 64 --out "$out" \
        > "$RES/p0a_${tag}_${step}.log" 2>&1
    if [ -f "$out" ]; then
      "$(pwd)/.venv/bin/python" - "$out" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
def find(o, key, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if key in k and isinstance(v, (int, float)):
                yield (path + "/" + k, v)
            yield from find(v, key, path + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from find(v, key, path + f"/{i}")
for k, v in find(d, "sens"):
    print("   ", k, round(float(v), 4))
PYEOF
    else
      echo "[s30watch] probe FAILED for $tag @ $step (see $RES/p0a_${tag}_${step}.log)"
    fi
    seen="$seen $ck"
  done
  sleep "$POLL"
done
