#!/bin/bash
# P2 goal-approach monotonicity on eval-task rollouts (phenomenon.tex Pred. mech.3), over
# the finished cube metric-probe matrix in $SCRATCH/factored-fb-metric-probe.
#
#   sbatch -A ASC26008 scripts/rollout_probe.slurm
#
# Per seed s in {0,1,2}: collect 10 episodes/task_id from the biql-d50 and gciql policies,
# then score the 2x2 (scoring critic x rollout source) — crossing rollout source controls
# the state-distribution confound. Idempotent: skips outputs that already exist.
#
#SBATCH -J rollout-probe
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd) =="

VENV="${VENV:-$SCRATCH/.venv-ms}"
PY="$VENV/bin/python"
RUNS="${RUNS:-$SCRATCH/factored-fb-metric-probe}"
ROLL="$RUNS/rollouts"
OUT="$RUNS/probes"
mkdir -p "$ROLL" "$OUT"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

declare -A CRT=( [biql-d50]="iql_bilinear" [gciql]="gciql" )
declare -A OVR=( [biql-d50]="value_latent_dim=50" [gciql]="" )

rdir () { echo "$RUNS/$1/cube-single-play-v0_${CRT[$1]}_flowbc_seed_$2"; }
ovr_flags () { local o="${OVR[$1]}"; [ -n "$o" ] && echo "--override $o"; }

FAIL=0
for s in 0 1 2; do
  for lab in biql-d50 gciql; do
    out="$ROLL/${lab}_s${s}.npz"
    if [ -f "$out" ]; then echo "== skip collect $out"; else
      echo "== collect $lab s$s =="
      $PY -m eval.rollout_probe --mode collect --critic "${CRT[$lab]}" --actor flowbc \
          --domain cube-single $(ovr_flags "$lab") --seed "$s" \
          --restore_path "$(rdir "$lab" "$s")" --restore_epoch 500000 \
          --episodes 10 --out "$out" \
          || { echo "!! FAILED collect $lab s$s"; FAIL=1; }
    fi
  done
  for scorer in biql-d50 gciql; do
    for src in biql-d50 gciql; do
      out="$OUT/p2_${scorer}-on-${src}_s${s}"
      if [ -f "$out.json" ]; then echo "== skip score $out"; continue; fi
      [ -f "$ROLL/${src}_s${s}.npz" ] || { echo "!! no rollouts for $src s$s"; FAIL=1; continue; }
      echo "== score $scorer on $src s$s =="
      $PY -m eval.rollout_probe --mode score --critic "${CRT[$scorer]}" --actor flowbc \
          --domain cube-single $(ovr_flags "$scorer") --seed "$s" \
          --restore_path "$(rdir "$scorer" "$s")" --restore_epoch 500000 \
          --rollouts "$ROLL/${src}_s${s}.npz" --out "$out" \
          || { echo "!! FAILED score $scorer on $src s$s"; FAIL=1; }
    done
  done
done
echo "== DONE (FAIL=$FAIL) =="
exit $FAIL
