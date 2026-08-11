#!/usr/bin/env bash
# Spec 21 follow-up: the HARD immediate target on cube-double and scene.
#
# Spec 21 stage 1 seeded the cube-single hard arm; its pre-registered gate read out
#   S1a PASS  zero-shot win survives 3 seeds: dirac .888 / dense .912 / region .647
#             (soft: .816 / .725 / .293 -- the region family MORE THAN DOUBLES)
#   S1b FAIL  no goal-success cost: hard .915 final, .951 last-4 vs soft .929
# which by spec 21 section 1's decision table means: adopt measure_density_hard outright,
# do NOT implement the mixture knob. Two cheap seeds replaced an alpha-knob, a six-test
# suite and a two-arm wave -- and overturned the "goal-success cost" premise the mixture
# was designed around (it was one noisy seed reading .876).
#
# Every density number in the paper is currently the SOFT arm (cs .929, cd .736, sc .628).
# This wave reruns the two multi-object envs under hard so the headline uses the better
# configuration, and so we learn whether hard's advantage is cube-single-specific.
# Cube-single needs no rerun: it already has 3 hard seeds (d18 s0 + d21 s1/s2).
#
# Concurrency: `active()` counts EVERY main.py job on the box, so this dispatcher
# cooperates with any other wave running at the same time -- it will idle rather than
# oversubscribe. Do NOT kill this dispatcher to change MAX: it is the pipeline parent of
# its children and the signal propagates (learned the hard way -- two cells died that way).
#
# Run in tmux:
#   tmux new -d -s hardx 'scripts/rescuing_fb/launch_hard_xenv.sh 2>&1 | tee -a /dev/shm/hardx_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_hard_xenv.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_hardx}
LOG=/dev/shm/hardx_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
FP=$OUTROOT/factored-fb

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

JOBS=(
  "hx_cd_hard_s0|cube_double_state_measure_density_hard_d50|0"
  "hx_cd_hard_s1|cube_double_state_measure_density_hard_d50|1"
  "hx_cd_hard_s2|cube_double_state_measure_density_hard_d50|2"
  "hx_sc_hard_s0|scene_state_measure_density_hard_d50|0"
  "hx_sc_hard_s1|scene_state_measure_density_hard_d50|1"
  "hx_sc_hard_s2|scene_state_measure_density_hard_d50|2"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r _t cfg _s <<<"$j"
  [ -f "configs/gciql/$cfg.yaml" ] || { echo "[hardx] FATAL missing config configs/gciql/$cfg.yaml"; exit 1; }
done

echo "[hardx] $(date +%H:%M:%S) ${#JOBS[@]} cells, MAX=$MAX (shared with any other wave)"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[hardx] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[hardx] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 120
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[hardx] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[hardx] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
