#!/usr/bin/env bash
# Full paper probe suite on the tuned PSM ortho_coef=1000 1Mx10 run (the new best config).
# CPU-only; mirrors scripts/run_psm_rldp_probe_suite.sh. Label = psm_o1000.
# Aliasing values are scored into the EXISTING analysis/value/aliasing_psm_rldp/sN dirs
# (same shared training_states.npz) so ortho1 (tuned) vs ortho1000 compare on identical states.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"; PY="$REPO/.venv/bin/python"; DATA="/dev/shm/factored-fb/datasets"
PRE="/dev/shm/factored-fb_outputs/psm_state_orthohi__ortho_coef1000__lr_phi1e-5"
M="psm"; LBL="psm_o1000"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
N_PAR="${N_PAR:-24}"; DRY="${DRY:-0}"
SRC_ALIAS="$REPO/analysis/value/aliasing_psm_rldp"          # has shared per-seed training_states.npz
ALIAS_ROOT="$REPO/analysis/value/aliasing_psm_o1000"        # separate root (psm_values here = ortho1000)
LOGDIR="$REPO/logs/probe_o1000_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR" analysis/value/repsep
cfg(){ echo "$1/.hydra/config.yaml"; }; ckpt(){ echo "$1/checkpoints/final.pt"; }
base="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY"

JOBS=()
for s in $SEEDS; do
  d="${PRE}__s${s}"; c=$(cfg "$d"); k=$(ckpt "$d")
  JOBS+=("phase_${LBL}_s${s}	$base scripts/phase_probe.py --config $c --checkpoint $k --out $REPO/analysis/profiles/${LBL}_phase/s${s} --device cpu --data-path $DATA --n-episodes 20 --mujoco-gl egl")
  JOBS+=("factor_${LBL}_s${s}	$base -m scripts.factorization_probe --config $c --checkpoint $k --data-path $DATA --out $REPO/analysis/factorization/${LBL}/s${s} --n-states 6000 --load-episodes 300 --tasks all --device cpu")
  JOBS+=("cf_${LBL}_s${s}	$base scripts/counterfactual_value_probe.py --config $c --checkpoint $k --method $M --data-path $DATA --n-states 200 --max-steps 200 --out-tag _${LBL}_ms${s} --mujoco-gl egl")
  # aliasing: reuse the shared per-seed npz (copy from SRC) so states match the tuned run,
  # then score psm here -> psm_values.parquet IS ortho1000 in this separate root.
  sd="$ALIAS_ROOT/s${s}"
  if [[ -f "$SRC_ALIAS/s${s}/training_states.npz" ]]; then
    mkdir -p "$sd"; cp -n "$SRC_ALIAS/s${s}/training_states.npz" "$sd/"
    JOBS+=("alias_${LBL}_s${s}	$base scripts/training_value_profile.py --method $M --config $c --checkpoint $k --data-path $DATA --mujoco-gl egl --out $sd --device cpu")
  fi
done

printf '[suite] %d jobs, N_PAR=%d, logs -> %s\n' "${#JOBS[@]}" "$N_PAR" "$LOGDIR"
if [[ "$DRY" == "1" ]]; then for j in "${JOBS[@]}"; do echo "  ${j%%	*}"; done; exit 0; fi

active=0
for j in "${JOBS[@]}"; do
  name="${j%%	*}" cmd="${j#*	}"
  ( eval "$cmd" >"$LOGDIR/${name}.log" 2>&1 && echo "[ok] $name" || echo "[FAIL $?] $name" ) &
  active=$((active+1)); while (( active >= N_PAR )); do wait -n 2>/dev/null; active=$((active-1)); done
done
wait
# aggregate aliasing for ortho1000 (psm_values here = ortho1000)
if [[ -d "$ALIAS_ROOT" ]]; then
  eval "$base scripts/aliasing_probe.py --root $ALIAS_ROOT --out $ALIAS_ROOT/aggregate" \
    >"$LOGDIR/alias_aggregate.log" 2>&1 && echo "[ok] alias aggregate" || echo "[FAIL] alias aggregate"
fi
echo "[suite] DONE $(date +%T) -> $LOGDIR"
