#!/usr/bin/env bash
# Train one OGBench run (1M steps, evaluation every 100k).
#
#   bash scripts/launch_ogbench.sh <method> <domain> <seed> [extra impls.main flags]
#
#   method  fb | hilp | rldp | density_fb | density_fb_z | density_fb_state | fb_ref
#           | density_fb_linear | density_fb_z_linear | density_fb_500k | density_fb_z_500k
#   domain  cube_single | cube_double | scene_play
#
# fb, hilp, rldp, density_fb_z are scored on eval_relabel.csv (reward inference), density_fb on
# eval.csv (goal-conditioned). PY sets the interpreter, DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/.."
METHOD=${1:?usage: launch_ogbench.sh <method> <domain> <seed>}
DOMAIN=${2:?domain}
SEED=${3:?seed}
shift 3
PY=${PY:-python}

# Goal kernel: task observation columns and success radius
case $DOMAIN in
  cube_single) HIT='"19:22:0.4"' ;;
  cube_double) HIT='"19:22:0.4,28:31:0.4"' ;;
  scene_play)  HIT='"19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"' ;;
  *) echo "unknown domain $DOMAIN" >&2; exit 2 ;;
esac

STEPS=1000000
EXTRA=()
case $METHOD in
  fb)                  CRITIC=fb_tdj ;;
  fb_ref)              CRITIC=fb_ref; STEPS=500000 ;;
  hilp)                CRITIC=hilp_tdjc ;;
  rldp)                CRITIC=rldp; EXTRA=(--save_dir exp/ortho1) ;;
  density_fb)          CRITIC=density_fb ;;
  density_fb_state)    CRITIC=density_fb_state ;;
  density_fb_z)        CRITIC=density_fb_z ;;
  density_fb_linear)   CRITIC=density_fb;   STEPS=500000
                       EXTRA=(--save_dir exp/fz_ablation_goal/linear_fz --override actor_value=linear_fz) ;;
  density_fb_z_linear) CRITIC=density_fb_z; STEPS=500000
                       EXTRA=(--save_dir exp/fz_ablation/linear_fz --override actor_value=linear_fz) ;;
  density_fb_500k)     CRITIC=density_fb;   STEPS=500000; EXTRA=(--save_dir exp/fz_ablation_goal/softmax) ;;
  density_fb_z_500k)   CRITIC=density_fb_z; STEPS=500000; EXTRA=(--save_dir exp/fz_ablation/softmax) ;;
  *) echo "unknown method $METHOD" >&2; exit 2 ;;
esac
case $CRITIC in density_fb*) EXTRA+=(--override "goal_spec=$HIT") ;; esac

# Reward-inference eval; it draws dataset samples, so it stays off for runs trained without it
RELABEL=(--override eval_relabel_size=10000 --override eval_relabel_shift=1.0
         --override eval_relabel_episodes=10)
case $METHOD/$DOMAIN in
  fb_ref/*|density_fb_linear/cube_single|density_fb_500k/cube_single) RELABEL=() ;;
esac

CMD=("$PY" -m impls.main --critic "$CRITIC" --actor flowbc --domain "$DOMAIN" --seed "$SEED"
     --train_steps "$STEPS" --eval_interval 100000 --save_interval 250000 --auto_resume
     "${RELABEL[@]}" "${EXTRA[@]}" "$@")
if [[ -n "${DRY:-}" ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=${WANDB_MODE:-offline} MUJOCO_GL=${MUJOCO_GL:-egl}
exec "${CMD[@]}"
