#!/usr/bin/env bash
# Spec 17 (PAPER/Multistep-FB/specs/17-delta-measure.md) -- the cube-single arms of the
# joint delta wave, 3 seeds each, 500k, FROM SCRATCH (no warm start on either arm).
#
#   d17_cs_delta   variant A: measure_delta -- the region measure's target reward becomes the
#                  SIGNED event indicator hit(x')-hit(x) (enter +1 / dwell 0 / exit -1),
#                  non-absorbing. Fixed point J = (1-gamma)*M_visitation - h(x0), bounded
#                  in [-1,1], SAME action argmax as the visitation measure. Deletes the
#                  action-independent dwell level from the supervision (ledger 9.16/9.15c).
#                  NOTE measure_stopgrad_psi=false -- the ballvisit d50 base is the phase-2
#                  FrozenPsi recipe and freezing psi from scratch would floor the cell.
#   d17_cs_part    variant B: measure_delta_partition -- one-stage simplex successor measure
#                  over K=256 dataset-drawn prototypes of the cube-xyz block. delta_head
#                  ([s,a,g]) -> K logits, p=softmax; CE fit to sg[(1-g)phi(x') + g*p_bar];
#                  actor maximizes p . phi(goal) at frozen head params. Validity is
#                  structural (no ortho, no hinge, no mass anchor, no scale_norm).
#
# Anchors (spec 17 section 5; do NOT re-run): fp-noortho 0.967 | visitation-from-scratch
# floor ~0.004-0.5 unstable | BC floor ~0.06-0.08 multi-object.
# Gates: G3 (variant B) = success >= 0.5 AND delta_p_entropy < 0.9*log(256) = 4.99 at 500k.
#        G2 (variant A) = no zero-flux collapse: measure_delta_event_rate bounded away
#        from 0, and msq not decaying to 0 with a flat loss.
#
# Prototypes for the partition arm are generated first (idempotent -- skipped if present):
#   results/delta_protos/cube_single_K256_s0.npy   [256, 3] float32, raw obs units
# They must survive for checkpoint restore: NOT /dev/shm (gets wiped).
#
# Run in tmux:
#   tmux new -d -s s17cs 'scripts/rescuing_fb/launch_spec17_cs.sh 2>&1 | tee -a /dev/shm/spec17_logs/dispatch.log'
# Dry run:  DRY_RUN=1 scripts/rescuing_fb/launch_spec17_cs.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=$(pwd)/.venv/bin/python
OUTROOT=${OUTROOT:-/home/adityamohan/exp_spec17}
LOG=/dev/shm/spec17_logs; mkdir -p "$LOG" "$OUTROOT"
DRY_RUN=${DRY_RUN:-0}
MAX=${MAX:-8}
GPUS=(0 1 2 3 4 5 6 7)
FP=$OUTROOT/factored-fb
PROTOS=$(pwd)/results/delta_protos/cube_single_K256_s0.npy
DATASET=${DATASET:-$HOME/.ogbench/data/cube-single-play-v0.npz}

active() { local c; c=$(pgrep -fc 'main.py --env_name' 2>/dev/null); echo "${c:-0}"; }

if [ ! -f "$PROTOS" ]; then
  echo "[s17] generating prototypes -> $PROTOS"
  mkdir -p "$(dirname "$PROTOS")"
  "$PY" scripts/rescuing_fb/make_delta_prototypes.py \
    --dataset "$DATASET" --hit_spec '19:22:0.4' --k 256 --seed 0 --out "$PROTOS" || exit 1
else
  echo "[s17] prototypes present: $PROTOS"
fi
# The prototypes are NOT serialized into the checkpoint (they are a static, non-pytree
# field so that pre-17b checkpoints still restore -- see gciql_flow.py's delta_protos
# comment). The checkpoint therefore records the HEAD but not the partition it was
# trained on: every post-wave eval re-reads this exact file, and a regenerated or
# permuted file would silently change the actor's q-term with no error. Freeze it and
# log the digest next to every run. Expected at launch (2026-08-06):
#   b4e13e9d9e0a0b0b18deedf5fdce66c1d3f9a17e02175608dd1679064494cbf5
# Backup: ~/delta_protos_backup/cube_single_K256_s0.npy (both mode 444)
echo "[s17] protos sha256: $(sha256sum "$PROTOS")"

# tag|config|seed
JOBS=(
  "d17_cs_delta_s0|cube_single_state_measure_delta_d50|0"
  "d17_cs_delta_s1|cube_single_state_measure_delta_d50|1"
  "d17_cs_delta_s2|cube_single_state_measure_delta_d50|2"

  "d17_cs_part_s0|cube_single_state_measure_delta_partition_d50|0"
  "d17_cs_part_s1|cube_single_state_measure_delta_partition_d50|1"
  "d17_cs_part_s2|cube_single_state_measure_delta_partition_d50|2"
)

echo "[s17] $(date +%H:%M:%S) ${#JOBS[@]} cells (2 arms x cube-single x 3 seeds), MAX=$MAX"
n=0; launched=0; done_already=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed <<<"$j"
  if compgen -G "$FP/*/$tag" > /dev/null; then
    echo "[s17] skip $tag - output dir already exists"; done_already=$((done_already+1)); continue
  fi
  while [ "$(active)" -ge "$MAX" ]; do
    echo "[s17] $(date +%H:%M:%S) active=$(active) >= $MAX - waiting to launch $tag"; sleep 60
  done
  g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1)); launched=$((launched+1))
  echo "[s17] $(date +%H:%M:%S) launch $tag cfg=$cfg seed=$seed on GPU$g"
  if [ "$DRY_RUN" = "1" ]; then
    echo "      DRY: --config-name $cfg seed=$seed output_root=$OUTROOT exp_name=$tag"; continue
  fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$g \
    "$PY" run_gciql.py --config-name "$cfg" seed="$seed" \
      output_root="$OUTROOT" exp_name="$tag" \
      > "$LOG/$tag.log" 2>&1 &
  sleep 45
done
wait
echo "[s17] $(date +%H:%M:%S) launched=$launched already_present=$done_already"
