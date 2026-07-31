#!/bin/bash
# Smoke-test the Metaworld integration on a GH200 node: every (critic, actor) pair of the
# run matrix, on both tasks, for a handful of steps plus one real eval rollout. This is the
# gate before submitting the 36-job matrix — it catches config, dataset-shape, env and
# critic/actor-composition errors in ~20 minutes instead of after a queue wait.
#
#   sbatch -A ASC26008 scripts/vista_smoke.slurm
#   FB_HF_EXPORT=$SCRATCH/fb-data/smoke_export sbatch -A ASC26008 scripts/vista_smoke.slurm
#   CRITICS="iql_bilinear gciql" ACTORS=flowbc DOMAINS="mw-assembly mw-basketball" \
#       sbatch -A ASC26008 scripts/vista_smoke.slurm     # gate one specific grid
#
# Also probes whether the compute node can reach api.wandb.ai, which decides whether the matrix
# can log to wandb live or has to run offline and be synced from a login node.
#
# Note this CANNOT be run on a login node: JAX's Eigen thread pool exceeds Vista's ~100
# thread-per-user login cap and aborts before the first step.
#
#SBATCH -J fb-smoke
#SBATCH -p gh-dev                # dev queue, max 2h
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail                 # NOT -e: we want every combo attempted, then a summary
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

VENV="${VENV:-$SCRATCH/.venv-jax-fb}"
PY="$VENV/bin/python"
export FB_HF_EXPORT="${FB_HF_EXPORT:-$SCRATCH/fb-data/hf_export}"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=disabled

STEPS="${STEPS:-200}"
OUT="${OUT:-$SCRATCH/factored-fb-smoke}"

# Which combos to gate. Defaults are the original 3x2x2; a matrix launch should pass the exact
# grid it is about to submit, so the smoke covers every (critic, actor, domain) that grid uses.
CRITICS="${CRITICS:-fb iql_bilinear gciql}"
ACTORS="${ACTORS:-flowbc ddpgbc}"
DOMAINS="${DOMAINS:-mw-bin-picking mw-button-press}"

echo "== host: $(hostname)  data: $FB_HF_EXPORT =="
$PY -c 'import jax; print("jax", jax.__version__, "devices:", jax.devices())'

# ── Does this compute node have outbound internet for live wandb logging? ────────────────────
# vista_train.slurm defaults to WANDB_MODE=offline on the assumption it does not. That is
# untested, and it can only be tested from inside a job. Reachability first (a blackholed route
# makes wandb.init hang, not fail), then a real authenticated init.
echo "---- wandb online probe ----"
if curl -sS --max-time 15 -o /dev/null -w 'api.wandb.ai HTTP %{http_code}\n' https://api.wandb.ai; then
  WANDB_MODE=online WANDB_DIR="$OUT" WANDB__SERVICE_WAIT=60 timeout 120 $PY - <<'PY' && \
    echo "WANDB_ONLINE=yes  -> matrix can log live" || echo "WANDB_ONLINE=no   -> matrix must run offline + sync"
import wandb
r = wandb.init(project="factored-fb", group="smoke-probe", name="connectivity", mode="online")
r.log({"probe": 1.0})
r.finish()
print("wandb online init OK:", r.url)
PY
else
  echo "WANDB_ONLINE=no   -> api.wandb.ai unreachable; matrix must run offline + sync"
fi

fail=0
for DOMAIN in $DOMAINS; do
  for CRITIC in $CRITICS; do
    for ACTOR in $ACTORS; do
      tag="$CRITIC-$ACTOR-$DOMAIN"
      echo "---- $tag ----"
      t0=$SECONDS
      if $PY -m impls.main --critic "$CRITIC" --actor "$ACTOR" --domain "$DOMAIN" \
             --seed 0 --train_steps "$STEPS" --log_interval "$STEPS" \
             --eval_interval "$STEPS" --save_interval "$STEPS" --eval_episodes 2 \
             --num_video "${NUM_VIDEO:-2}" \
             --save_dir "$OUT" > "logs/smoke-$tag.log" 2>&1; then
        vids=$(find "$OUT" -name '*.mp4' 2>/dev/null | wc -l)
        echo "PASS $tag  $((SECONDS - t0))s  vids=$vids  $(grep -h '^eval' "logs/smoke-$tag.log" | tail -1)"
      else
        echo "FAIL $tag  -> logs/smoke-$tag.log"
        tail -20 "logs/smoke-$tag.log"
        fail=$((fail + 1))
      fi
    done
  done
done

echo "== smoke done: $fail failure(s) =="
exit $fail
