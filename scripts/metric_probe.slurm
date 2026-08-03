#!/bin/bash
# Run the metric-probe battery (docs/specs/2026-07-31-bilinear-gciql-metric-probe.md §4-§7)
# over the finished cube-single run matrix in $SCRATCH/factored-fb-metric-probe.
#
#   sbatch -A ASC26008 scripts/metric_probe.slurm
#
# Probes every condition at its final checkpoint (500k), the subject (biql-d50 seed 0)
# across training time, and the untrained iql_bilinear nets (D9 null). Then aggregates
# seed-0 npzs into the P6 RSA matrix with eval/probe_report.py.
#
#SBATCH -J metric-probe
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j

set -uo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
echo "== code: $(pwd) =="

VENV="${VENV:-$SCRATCH/.venv-ms}"
PY="$VENV/bin/python"
RUNS="${RUNS:-$SCRATCH/factored-fb-metric-probe}"
OUT="$RUNS/probes"
PROBE_SET="$RUNS/probe_set.npz"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export FB_HF_EXPORT="${FB_HF_EXPORT:-$SCRATCH/fb-data/hf_export}"
export FB_DATA_ROOT="${FB_DATA_ROOT:-$SCRATCH/fb-data/expert}"

if [ ! -f "$PROBE_SET" ]; then
  $PY -m eval.probe_set --data ~/.ogbench/data/cube-single-play-v0-val.npz --out "$PROBE_SET"
fi

FAIL=0
probe () {  # label critic actor seed epoch extra_flags...
  local label="$1" critic="$2" actor="$3" seed="$4" epoch="$5"; shift 5
  local out="$OUT/${label}_s${seed}_${epoch}"
  if [ -f "$out.json" ]; then echo "== skip $out (exists)"; return 0; fi
  local rdir="$RUNS/$label/cube-single-play-v0_${critic}_${actor}_seed_${seed}"
  echo "== probe $label seed $seed epoch $epoch =="
  if [ "$epoch" = "init" ]; then
    $PY -m eval.metric_probe --critic "$critic" --actor "$actor" --domain cube-single \
        --seed "$seed" --init_only --probe_set "$PROBE_SET" --out "$OUT/${label}-init_s${seed}_0" "$@" \
        || { echo "!! FAILED $label-init s$seed"; FAIL=1; }
  else
    $PY -m eval.metric_probe --critic "$critic" --actor "$actor" --domain cube-single \
        --seed "$seed" --restore_path "$rdir" --restore_epoch "$epoch" \
        --probe_set "$PROBE_SET" --out "$out" "$@" \
        || { echo "!! FAILED $label s$seed e$epoch"; FAIL=1; }
  fi
}

for s in 0 1 2; do
  probe biql-d50  iql_bilinear flowbc $s 500000 --override value_latent_dim=50
  probe biql-d512 iql_bilinear flowbc $s 500000 --override value_latent_dim=512
  probe gciql     gciql        flowbc $s 500000
  probe crl-d50   crl          flowbc $s 500000 --override latent_dim=50
  probe crl-d512  crl          flowbc $s 500000 --override latent_dim=512
  probe qrl-d64   qrl          bcflow $s 500000 --override qrl_components=8 --override qrl_dim_per_component=8
  probe qrl-d512  qrl          bcflow $s 500000 --override qrl_components=64 --override qrl_dim_per_component=8
  probe vip-d50   vip          bcflow $s 500000 --override latent_dim=50 --override discount=0.98
  probe vip-d512  vip          bcflow $s 500000 --override latent_dim=512 --override discount=0.98
done
probe vip-d50-g99 vip bcflow 0 500000 --override latent_dim=50 --override discount=0.99

# probe-vs-training-time for the subject (spec §5: when does metric structure emerge?)
for e in 100000 200000 300000 400000; do
  probe biql-d50 iql_bilinear flowbc 0 $e --override value_latent_dim=50
done

# D9 untrained-net nulls
probe biql-d50  iql_bilinear flowbc 0 init --override value_latent_dim=50
probe biql-d512 iql_bilinear flowbc 0 init --override value_latent_dim=512

# d-sweep checkpoints (phenomenon.tex Remark 1 follow-up: does the LEARNED eff-rank saturate
# at ~15-20 regardless of container size d, i.e. is solution rank the invariant?)
for s in 0 1 2; do
  for d in 5 10 20 30; do
    probe biql-d$d iql_bilinear flowbc $s 500000 --override value_latent_dim=$d
  done
done

# P6 cross-method RSA over seed-0 final-epoch matrices (+ null)
$PY -m eval.probe_report --out "$OUT/report" \
    "$OUT"/biql-d50_s0_500000.npz "$OUT"/biql-d512_s0_500000.npz \
    "$OUT"/gciql_s0_500000.npz "$OUT"/crl-d50_s0_500000.npz "$OUT"/crl-d512_s0_500000.npz \
    "$OUT"/qrl-d64_s0_500000.npz "$OUT"/qrl-d512_s0_500000.npz \
    "$OUT"/vip-d50_s0_500000.npz "$OUT"/vip-d512_s0_500000.npz \
    "$OUT"/biql-d50-init_s0_0.npz \
    || { echo "!! RSA report failed"; FAIL=1; }

echo "== DONE (FAIL=$FAIL) =="
exit $FAIL
