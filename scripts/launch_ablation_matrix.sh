#!/usr/bin/env bash
# Goal-conditioned vs z-conditioned training x object-coordinate vs full-state backward input,
# for HILP, RLDP (vanilla + density) and Density-FB, cube_single + cube_double, 1M, 3 seeds.
# Spec: docs/specs/2026-09-06-goal-vs-z-ablation-matrix.md. Every run logs BOTH eval routes.
#
#   DRY=1        print the pending job list
#   JOBS_ONLY=1  one shell command per line (slurm array input)
#   TIER=1|2|3   restrict to a priority tier (default: all)
#   GPUS="0 1"   local tmux launch on those GPUs (one worker each, jobs round-robin)
set -euo pipefail
REPO="${REPO:-$HOME/git/Factored-FB-seam}"; PY="${PY:-/dev/shm/.venv-jax/bin/python}"
LOGDIR="${LOGDIR:-/dev/shm/matrix_logs}"; SESSION="${SESSION:-matrix}"
STEPS=1000000; EVAL_EVERY=100000; SAVE_EVERY=250000
CD_HIT='measure_hit_spec=19:22:0.4,28:31:0.4'
RELABEL="--override eval_relabel_size=10000 --override eval_relabel_shift=1.0 --override eval_relabel_episodes=10"
cd "$REPO"
# tier  critic               needs_cd_hit   (family / conditioning / B input)
ARMS=(
  "1 density_fb_z          hit"   # Density-FB  z     obj   (new default reward basis)
  "1 density_fb_z_state    hit"   # Density-FB  z     state
  "1 density_fb            hit"   # Density-FB  goal  obj   (paper method; 1M may exist)
  "1 density_fb_state      hit"   # Density-FB  goal  state
  "2 hilp_density_gk       hit"   # HILP-dens   goal  state
  "2 hilp_density_gk_oc    hit"   # HILP-dens   goal  obj
  "2 rldp_density_gk       hit"   # RLDP-dens   goal  state
  "2 rldp_density_gk_oc    hit"   # RLDP-dens   goal  obj
  "2 hilp_density_cube     -"     # HILP-dens   z     state
  "2 hilp_density_oc       hit"   # HILP-dens   z     obj
  "2 rldp_density          -"     # RLDP-dens   z     state
  "2 rldp_density_oc       hit"   # RLDP-dens   z     obj
  "3 hilp_gc               -"     # HILP        goal  state
  "3 hilp_gc_oc            hit"   # HILP        goal  obj
  "3 rldp_gc               -"     # RLDP        goal  state
  "3 rldp_gc_oc            hit"   # RLDP        goal  obj
  "3 hilp_cube             -"     # HILP        z     state
  "3 hilp_oc               hit"   # HILP        z     obj
  "3 rldp                  -"     # RLDP        z     state
  "3 rldp_oc               hit"   # RLDP        z     obj
)
JOBS=""
for a in "${ARMS[@]}"; do
  read -r tier critic hit <<< "$a"
  [ -n "${TIER:-}" ] && [ "$tier" != "$TIER" ] && continue
  for pair in "cube_single:cube-single-play-v0:-" "cube_double:cube-double-play-v0:$CD_HIT"; do
    dom="${pair%%:*}"; rest="${pair#*:}"; envn="${rest%%:*}"; ov="${rest#*:}"
    [ "$hit" = "-" ] && ov="-"
    for seed in 0 1 2; do
      [ -f "exp/${envn}_${critic}_flowbc_seed_${seed}/params_${STEPS}.pkl" ] && continue
      JOBS+="$critic $dom $seed $ov"$'\n'
    done
  done
done
N=$(printf '%s' "$JOBS" | grep -c . || true)
mk() { # critic dom seed ov -> command
  local ovflag=""; [ "$4" = "-" ] || ovflag="--override $4"
  echo "$PY -m impls.main --critic $1 --actor flowbc --domain $2 --seed $3 --auto_resume --train_steps $STEPS --eval_interval $EVAL_EVERY --save_interval $SAVE_EVERY $ovflag $RELABEL"
}
if [ "${JOBS_ONLY:-0}" = "1" ]; then
  printf '%s' "$JOBS" | while read -r c d s o; do [ -n "${c:-}" ] && mk "$c" "$d" "$s" "$o"; done; exit 0
fi
echo "$N pending jobs (tier ${TIER:-all}):"; printf '%s' "$JOBS" | nl -ba | sed 's/^/   /'
[ "$N" -gt 0 ] || { echo "nothing to run"; exit 0; }
[ "${DRY:-0}" = "1" ] && exit 0
[ -n "${GPUS:-}" ] || { echo "set GPUS=\"0 1 ...\" for a local launch, or use scripts/ablation_matrix.slurm"; exit 1; }
read -r -a GPU_LIST <<< "$GPUS"; mkdir -p "$LOGDIR"
tmux has-session -t "=${SESSION}" 2>/dev/null && { echo "session '$SESSION' exists"; exit 1; }
NG=${#GPU_LIST[@]}; first=1
for w in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$w]}"; slice=$(printf '%s' "$JOBS" | awk -v w="$w" -v n="$NG" 'NF && (NR-1)%n==w'); [ -n "$slice" ] || continue
  body="cd $REPO; "
  while read -r c d s o; do [ -n "${c:-}" ] || continue
    body+="echo '>>> $d-$c-s$s'; CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl $(mk "$c" "$d" "$s" "$o") 2>&1 | tee $LOGDIR/$d-$c-s$s.log; "
  done <<< "$slice"
  body+="echo ALL_DONE_GPU_$gpu; sleep 86400"
  if [ "$first" -eq 1 ]; then tmux new-session -d -s "$SESSION" -n "gpu$gpu" "$body"; first=0; else tmux new-window -d -t "=${SESSION}" -n "gpu$gpu" "$body"; fi
  echo "  GPU $gpu <- $(printf '%s' "$slice" | grep -c .) jobs"
done
