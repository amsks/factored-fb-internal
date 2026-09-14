#!/usr/bin/env bash
# Why Density-RLDP loses on walker: is it the ACTOR configuration rather than the objective?
#
# Measured at 500k on exp/walker_20260914: q_mean is 1.13 for Density-RLDP against 661 for
# plain RLDP on the identical task. The density critic's value is an inner product against a
# NORMALIZED measure, so it is ~600x smaller in scale than RLDP's bilinear value -- and the
# walker block runs DDPG+BC as a pure TD3 actor (actor.alpha=0, no BC term; q_normalize=false,
# the value term is not divided by its own mean magnitude). That pairing was chosen for parity
# with RLDP's reference actor, and it is right for plain RLDP. But every cube arm where the
# density objective wins runs the FLOW actor, whose value term is both centered and scale-fixed
# against a BC term. The density critic has never been validated against an unnormalized
# pure-Q actor anywhere.
#
# Two single-factor arms, so the attribution is clean:
#   A  actor.q_normalize=true   -- restores the scale handling the density value has elsewhere
#   B  lr_actor=1e-4            -- cube's value; the walker block runs 3x hot (rescore note F1)
#   C  plain RLDP + q_normalize=true -- the control, so arm A's number is not compared against
#                                       a baseline that never saw the normalizer
#
# Everything else is held at the exp/walker_20260914 settings: rldp_density_tdj_reg (regression
# task index), ortho_coef=1, discount 0.98, 500k steps, motivo actor block, hit spec 0:24:15.9.
# Score with scripts/exorl_paper_eval.py exactly as that block was scored (symlink the nested
# layout exp/<root>/dmc-walker/<critic>/<run_dir> first).
#
# Gotchas encoded here:
#   - FB_MUJOCO_SITE_PACKAGES is REQUIRED for every ExoRL launch or the env build crashes.
#   - measure_hit_spec MUST be quoted (OmegaConf reads a bare a:b:c as sexagesimal).
#   - This queue waits for the dfb_state queue to claim its six GPUs first: those are main-table
#     cells and outrank this diagnostic.
set -u
cd "$(dirname "$0")/.."

ROOT=exp/walker_actorfix
LOGS=logs/walker_actorfix
mkdir -p "$LOGS"
SITE=/home/adityamohan/git/Factored-FB/.venv/lib/python3.11/site-packages
SPEC='0:24:15.9'

# "<tag>|<critic>|<seed>|<extra overrides>"
PENDING=()
for s in 0 1 2; do PENDING+=("qnorm|rldp_density_tdj_reg|$s|--override actor.q_normalize=true"); done
for s in 0 1 2; do PENDING+=("lr1e4|rldp_density_tdj_reg|$s|--override lr_actor=1e-4"); done
for s in 0 1 2; do PENDING+=("qnorm_plain|rldp_tdj|$s|--override actor.q_normalize=true"); done

GPUS=(0 1 2 3 4 5 6 7)

gpu_free () {
  [ -n "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 1
  sleep 30
  [ -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}

launch () {
  local gpu="$1" tag="$2" critic="$3" seed="$4" extra="$5"
  echo "$(date -u +%H:%M:%SZ) launching ${tag}_s${seed} ($critic) on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  FB_MUJOCO_SITE_PACKAGES="$SITE" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  WANDB_MODE=offline MUJOCO_GL=egl \
  /dev/shm/.venv-jax/bin/python -m impls.main \
      --actor ddpgbc --auto_resume \
      --train_steps 500000 --eval_interval 100000 --save_interval 100000 \
      --domain dmc-walker --critic "$critic" --seed "$seed" --save_dir "$ROOT/$tag" \
      --override discount=0.98 \
      --override actor.motivo_archi=true \
      --override 'actor.hidden_dims=[1024]' \
      --override actor.alpha=0.0 \
      --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 \
      --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 \
      --override ortho_coef=1 \
      --override "measure_hit_spec=\"$SPEC\"" \
      $extra \
      > "$LOGS/${tag}_s${seed}.log" 2>&1 &
}

# Let the main-table queue take its six GPUs first.
echo "$(date -u +%H:%M:%SZ) waiting for the dfb_state queue to launch its six runs"
while [ "$(ls -1d exp/dfb_state/*seed_* 2>/dev/null | wc -l)" -lt 6 ]; do sleep 300; done
echo "$(date -u +%H:%M:%SZ) dfb_state is fully launched; ${#PENDING[@]} actor-fix runs pending"

i=0
while [ "$i" -lt "${#PENDING[@]}" ]; do
  for g in "${GPUS[@]}"; do
    [ "$i" -lt "${#PENDING[@]}" ] || break
    if gpu_free "$g"; then
      IFS='|' read -r tag critic seed extra <<< "${PENDING[$i]}"
      launch "$g" "$tag" "$critic" "$seed" "$extra"
      i=$((i+1))
      sleep 120
    fi
  done
  [ "$i" -lt "${#PENDING[@]}" ] && sleep 300
done
echo "$(date -u +%H:%M:%SZ) all ${#PENDING[@]} runs started"
wait
echo "$(date -u +%H:%M:%SZ) ACTORFIX_DONE"
