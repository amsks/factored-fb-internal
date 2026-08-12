#!/usr/bin/env bash
# Spec 24: z-indexed Density-FB. Swaps the policy-family index from goals to an FB-style
# latent z and changes nothing else, to test whether the density estimator is a drop-in
# inside FB's full interface.
#
# TWO venues, for a reason worth recording.
#   point_mass_maze  the spec's own venue, chosen because z-machinery is known healthy
#                    there. Its anchors came in at g-density .315, stock GCIQL .242,
#                    BC-only .235 -- so spec 23's gate G0b FAILED (it required both to
#                    beat BC by >= .10; they beat it by .080 and .007). The venue's whole
#                    dynamic range is .080, narrower than the +-.05 band spec 24 G1 uses.
#                    Run for interface coverage, not for resolving power.
#   cube_double      a DEVIATION from spec 24's anti-scope, taken deliberately. That
#                    anti-scope rests on the 2026-08-05 stack-divergence verdict, which
#                    was reached with the bilinear scalar critic rather than this
#                    estimator. Cube-double separates methods over .019 to .736, so a
#                    z-vs-g difference is measurable there. Anchor: g-indexed soft .736.
#
# Read: z near the g-indexed anchor means the estimator survives FB's full interface.
# z at the floor means the stack-divergence verdict extends to this estimator, which is
# the informative negative and should be reported, not patched (spec 24 section 6).
#
# Cooperative concurrency; DRY_RUN short-circuits before the wait.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUT=${OUT:-/home/adityamohan/exp_spec24}; LOG=/dev/shm/spec24_logs
mkdir -p "$LOG" "$OUT"; DRY_RUN=${DRY_RUN:-0}; MAX=${MAX:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); FP=$OUT/factored-fb
ENVS=${ENVS:-"pmm cd"}; SEEDS=${SEEDS:-"0 1 2"}
active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }
cfg_for() { case "$1" in
  pmm) echo point_mass_maze_rnd_seed0_measure_density_z_d50 ;;
  cd)  echo cube_double_state_measure_density_z_d50 ;;
  *)   echo "" ;; esac; }
JOBS=()
for e in $ENVS; do
  c=$(cfg_for "$e"); [ -n "$c" ] || { echo "[z24] FATAL unknown env '$e'"; exit 1; }
  [ -f "configs/gciql/$c.yaml" ] || { echo "[z24] FATAL missing configs/gciql/$c.yaml"; exit 1; }
  for s in $SEEDS; do JOBS+=("z24_${e}_z_s${s}|$c|$s"); done
done
echo "[z24] $(date +%H:%M:%S) ${#JOBS[@]} cells, envs='$ENVS', MAX=$MAX"
n=0; launched=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  compgen -G "$FP/*/$tag" >/dev/null && { echo "[z24] skip $tag (exists)"; continue; }
  if [ "$DRY_RUN" = "1" ]; then echo "      DRY: $tag cfg=$cfg seed=$seed"; continue; fi
  while [ "$(active)" -ge "$MAX" ]; do echo "[z24] $(date +%H:%M:%S) active=$(active) waiting for $tag"; sleep 180; done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[z24] $(date +%H:%M:%S) launch $tag on GPU$g"
  XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" output_root="$OUT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait; echo "[z24] $(date +%H:%M:%S) launched=$launched"
