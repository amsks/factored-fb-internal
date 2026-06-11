#!/usr/bin/env bash
# scripts/run_psm_rldp_probe_suite.sh
# Full paper probe suite on the tuned PSM + RLDP 1M x10 state runs.
# CPU-only (CUDA_VISIBLE_DEVICES="" + --device cpu) so it never touches the
# training GPUs. Probes:
#   1 phase_probe         -> analysis/profiles/<m>_phase/sN
#   2 factorization_probe -> analysis/factorization/<m>/sN
#   3 counterfactual      -> analysis/value/repsep/cf_*_<m>_sN  (fixed path; --out-tag)
#   4 aliasing chain      -> analysis/value/aliasing_psm_rldp/sN  (npz + {psm,rldp}_values)
# Verified invocations from the probe-suite smoke test. DRY=1 prints the job list.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY="$REPO/.venv/bin/python"
DATA="/dev/shm/factored-fb/datasets"
OUTROOT="$REPO/outputs_root_dummy"   # unused; per-probe dirs are absolute below
PSM_PRE="/dev/shm/factored-fb_outputs/psm_state_tuned__ortho_coef1__lr_phi1e-5"
RLDP_PRE="/dev/shm/factored-fb_outputs/rldp_state_tuned__ortho_coef0p001__lr_b1e-5"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
N_PAR="${N_PAR:-24}"
DRY="${DRY:-0}"
ALIAS_ROOT="$REPO/analysis/value/aliasing_psm_rldp"
LOGDIR="$REPO/logs/probe_suite_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR" analysis/value/repsep "$ALIAS_ROOT"

cfg(){ echo "$1/.hydra/config.yaml"; }
ckpt(){ echo "$1/checkpoints/final.pt"; }
base="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY"

# ---- build job list: "<logname>\t<command>" ----
JOBS=()
for m in psm rldp; do
  pre=$([ "$m" = psm ] && echo "$PSM_PRE" || echo "$RLDP_PRE")
  for s in $SEEDS; do
    d="${pre}__s${s}"; c=$(cfg "$d"); k=$(ckpt "$d")
    # 1 phase
    JOBS+=("phase_${m}_s${s}	$base scripts/phase_probe.py --config $c --checkpoint $k --out $REPO/analysis/profiles/${m}_phase/s${s} --device cpu --data-path $DATA --n-episodes 20 --mujoco-gl egl")
    # 2 factorization
    JOBS+=("factor_${m}_s${s}	$base -m scripts.factorization_probe --config $c --checkpoint $k --data-path $DATA --out $REPO/analysis/factorization/${m}/s${s} --n-states 6000 --load-episodes 300 --tasks all --device cpu")
    # 3 counterfactual (fixed output path; seed via --out-tag)
    JOBS+=("cf_${m}_s${s}	$base scripts/counterfactual_value_probe.py --config $c --checkpoint $k --method $m --data-path $DATA --n-states 200 --max-steps 200 --out-tag _${m}_s${s} --mujoco-gl egl")
  done
done
# 4 aliasing chain (per seed: bootstrap shared npz from RLDP net via --method fb,
#    then score psm + rldp into the SAME seed dir). FB row is spurious -> dropped later.
for s in $SEEDS; do
  pd="${PSM_PRE}__s${s}"; rd="${RLDP_PRE}__s${s}"; sd="$ALIAS_ROOT/s${s}"
  mkdir -p "$sd"
  JOBS+=("alias_npz_s${s}	$base scripts/training_value_profile.py --method fb --config $(cfg "$rd") --checkpoint $(ckpt "$rd") --data-path $DATA --mujoco-gl egl --n-states 20000 --out $sd --device cpu")
done

printf '[suite] %d jobs, N_PAR=%d, logs -> %s\n' "${#JOBS[@]}" "$N_PAR" "$LOGDIR"
if [[ "$DRY" == "1" ]]; then
  for j in "${JOBS[@]}"; do echo "  ${j%%	*}"; done
  echo "  (+ per-seed alias: psm/rldp value passes run after npz; + final aggregate)"
  exit 0
fi

run_pool(){ # args: array of "name\tcmd"
  local -n arr=$1; local active=0
  for j in "${arr[@]}"; do
    local name="${j%%	*}" cmd="${j#*	}"
    ( eval "$cmd" >"$LOGDIR/${name}.log" 2>&1 && echo "[ok] $name" || echo "[FAIL $?] $name" ) &
    active=$((active+1))
    while (( active >= N_PAR )); do wait -n 2>/dev/null; active=$((active-1)); done
  done
  wait
}

echo "[suite] phase 1: phase+factor+cf+npz ($(date +%T))"
run_pool JOBS

# aliasing phase 2: score psm + rldp on each seed's shared npz, then aggregate
ALIAS2=()
for s in $SEEDS; do
  pd="${PSM_PRE}__s${s}"; rd="${RLDP_PRE}__s${s}"; sd="$ALIAS_ROOT/s${s}"
  [[ -f "$sd/training_states.npz" ]] || { echo "[warn] no npz for s$s, skipping alias scoring"; continue; }
  ALIAS2+=("alias_psm_s${s}	$base scripts/training_value_profile.py --method psm --config $(cfg "$pd") --checkpoint $(ckpt "$pd") --data-path $DATA --mujoco-gl egl --out $sd --device cpu")
  ALIAS2+=("alias_rldp_s${s}	$base scripts/training_value_profile.py --method rldp --config $(cfg "$rd") --checkpoint $(ckpt "$rd") --data-path $DATA --mujoco-gl egl --out $sd --device cpu")
done
if (( ${#ALIAS2[@]} )); then
  echo "[suite] phase 2: alias psm+rldp value passes ($(date +%T))"
  run_pool ALIAS2
  # drop spurious FB-on-RLDP-net rows
  find "$ALIAS_ROOT" -name fb_values.parquet -delete
  echo "[suite] aggregate aliasing ($(date +%T))"
  eval "$base scripts/aliasing_probe.py --root $ALIAS_ROOT --out $ALIAS_ROOT/aggregate" \
    >"$LOGDIR/alias_aggregate.log" 2>&1 && echo "[ok] alias aggregate" || echo "[FAIL] alias aggregate"
fi
echo "[suite] DONE $(date +%T) -> logs $LOGDIR ; aliasing $ALIAS_ROOT/aggregate"
