#!/usr/bin/env bash
# Phase-2 occupancy + Cell W centering (2026-07-23). Three arms:
#  B  centered: CellW0-NoOrtho + scale_norm + actor_q_center (the wave-2 forensics fix:
#     the M=1+(1-gamma)Q affine offset diluted the actor's greedy signal ~3.8x). x3 seeds.
#     If this recovers ~0.90 the theorem cell (== bilinear GC-IQL) is finally realized.
#  C1 frozen-psi occupancy: warm-start the FULL net from the FP-NoOrtho 0.967 ckpt (per
#     seed), freeze psi (measure_stopgrad_psi), train phi by the PLAIN FB occupancy TD
#     (measure_first_passage=false). Works => occupancy TD failed by CORRUPTING the
#     representation; floors => the occupancy target itself is unlearnable on play data.
#  C2 train-psi control: same warm-start, psi TRAINABLE. Separates "warm basin survives
#     occupancy" from the frozen-psi mechanism read. x2 seeds.
# Queues behind wave-3 via the same MAX gate. GPUs 1/5/6 = other track (never).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec07}
LOG=/dev/shm/spec07_logs; mkdir -p "$LOG" "$OUTROOT"
MAX=${MAX:-6}
GPUS=(0 2 3 4 7)
FPRUNS=/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50

active() { pgrep -fc 'main.py --env_name' 2>/dev/null || echo 0; }
pick_gpu() {
  local elig; elig=$(printf '%s\n' "${GPUS[@]}" | paste -sd'|' -)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v e="^(${elig})$" '$1 ~ e {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

WS="+agent_overrides.warm_start_epoch=500000"
JOBS=(
  # C1 FIRST (2026-07-23 deadline reorder): the occupancy-on-FP verdict is the
  # time-critical read; w0cent moved behind it (s0/s1 lost ~10k steps in the reorder).
  # C1: frozen-psi occupancy phase 2 (warm-start per matching FP seed)
  "occfz_s0|cube_single_state_measure_firstpassage_noortho_d50|0|agent_overrides.measure_first_passage=false +agent_overrides.measure_stopgrad_psi=true +agent_overrides.warm_start_ckpt=$FPRUNS/m_firstpassage_noortho_s0 $WS run_group=GCIQL-State-Flow-Measure-Occ-Phase2-FrozenPsi-d50"
  "occfz_s1|cube_single_state_measure_firstpassage_noortho_d50|1|agent_overrides.measure_first_passage=false +agent_overrides.measure_stopgrad_psi=true +agent_overrides.warm_start_ckpt=$FPRUNS/m_firstpassage_noortho_s1 $WS run_group=GCIQL-State-Flow-Measure-Occ-Phase2-FrozenPsi-d50"
  "occfz_s2|cube_single_state_measure_firstpassage_noortho_d50|2|agent_overrides.measure_first_passage=false +agent_overrides.measure_stopgrad_psi=true +agent_overrides.warm_start_ckpt=$FPRUNS/m_firstpassage_noortho_s2 $WS run_group=GCIQL-State-Flow-Measure-Occ-Phase2-FrozenPsi-d50"
  # B: the centered Cell W (theorem cell with the dilution fix)
  "w0cent_s0|cube_single_state_measure_cellw0_noortho_d50|0|+agent_overrides.measure_scale_norm=true +agent_overrides.actor_q_center=true run_group=GCIQL-State-Flow-Measure-CellW0-NoOrtho-Centered-d50"
  "w0cent_s1|cube_single_state_measure_cellw0_noortho_d50|1|+agent_overrides.measure_scale_norm=true +agent_overrides.actor_q_center=true run_group=GCIQL-State-Flow-Measure-CellW0-NoOrtho-Centered-d50"
  "w0cent_s2|cube_single_state_measure_cellw0_noortho_d50|2|+agent_overrides.measure_scale_norm=true +agent_overrides.actor_q_center=true run_group=GCIQL-State-Flow-Measure-CellW0-NoOrtho-Centered-d50"
  # C2: train-psi warm-start control
  "occtr_s0|cube_single_state_measure_firstpassage_noortho_d50|0|agent_overrides.measure_first_passage=false +agent_overrides.warm_start_ckpt=$FPRUNS/m_firstpassage_noortho_s0 $WS run_group=GCIQL-State-Flow-Measure-Occ-Phase2-TrainPsi-d50"
  "occtr_s1|cube_single_state_measure_firstpassage_noortho_d50|1|agent_overrides.measure_first_passage=false +agent_overrides.warm_start_ckpt=$FPRUNS/m_firstpassage_noortho_s1 $WS run_group=GCIQL-State-Flow-Measure-Occ-Phase2-TrainPsi-d50"
)

n=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed extra <<<"$j"
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[p2] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=$(pick_gpu); [ -z "$g" ] && g=${GPUS[0]}
  n=$((n+1))
  echo "[p2] $(date +%H:%M:%S) launch #$n/${#JOBS[@]} $tag on GPU$g ${extra:+[$extra]}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" $extra \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[p2] all phase-2 jobs finished"
