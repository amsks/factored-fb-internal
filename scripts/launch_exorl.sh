#!/usr/bin/env bash
# Train one ExoRL run of Density-FB with reward inference (1M steps on the RND dataset).
#
#   bash scripts/launch_exorl.sh <domain> <seed> [extra impls.main flags]
#   domain  walker | quadruped | point-mass | cheetah
#
# Each domain's settings are those of the runs reported in the ExoRL table (read from their
# config.json). All four use Meta Motivo's network shapes, B normalised onto the sqrt(d)
# sphere, lr 1e-4, and the DDPG+BC actor with alpha 0 (actor/ddpgbc.yaml). Per domain:
#   hit spec     which observation columns define the training goals' reward, and how close
#                counts as reaching them (radius calibrated by scripts/calibrate_hit_spec.py)
#   walker, cheetah  z always drawn uniformly on the sphere (rand_frac 1), F width 1024,
#                    target rate 0.001, actor Q normalised
#   quadruped, point-mass  half the z inferred from hindsight goals, default F, Q unnormalised
#
# Runs land in exp/exorl/<dmc-domain>/density_fb_z/, the layout scripts/exorl_paper_eval.py
# scores. Evaluation during training infers z from 50k reward samples of the domain's default
# task; the table numbers come from scripts/exorl_paper_eval.py after training.
# PY selects the interpreter; EXORL_DATA_ROOT the dataset root; DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/.."
DOM=${1:?usage: launch_exorl.sh <domain> <seed>}
SEED=${2:?seed}
shift 2
PY=${PY:-python}

COMMON=(--override lr=1e-4 --override 'backward_hidden_dims=[256,256]'
        --override archi=motivo --override norm_backward=true)
SPHERE_Z=(--override z_random_frac=1.0 --override 'value_hidden_dims=[1024]'
          --override tau=0.001 --override actor.q_normalize=true)
case $DOM in
  walker)     DOMAIN=dmc-walker; ARM=("${SPHERE_Z[@]}" --override discount=0.98
                --override 'goal_spec="0:24:15.9"') ;;
  cheetah)    DOMAIN=dmc-cheetah; ARM=("${SPHERE_Z[@]}" --override discount=0.98
                --override 'goal_spec="8:17:7.264"') ;;
  quadruped)  DOMAIN=dmc-quadruped; ARM=(--override discount=0.98 --override actor.q_normalize=false
                --override 'goal_spec="44:47:0.110,47:48:0.001"') ;;
  point-mass) DOMAIN=dmc-point-mass-maze; ARM=(--override discount=0.99 --override lr_actor=1e-6
                --override actor.q_normalize=false --override 'goal_spec="0:2:0.03"') ;;
  *) echo "unknown domain $DOM" >&2; exit 2 ;;
esac

CMD=("$PY" -m impls.main --critic density_fb_z --actor ddpgbc --domain "$DOMAIN" --seed "$SEED"
     --train_steps 1000000 --eval_interval 250000 --save_interval 250000 --eval_episodes 10
     --infer_cond_episodes 50000 --eval_on_inferred_cond --auto_resume
     --save_dir "exp/exorl/$DOMAIN/density_fb_z" "${COMMON[@]}" "${ARM[@]}" "$@")
if [[ -n "${DRY:-}" ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=${WANDB_MODE:-offline} MUJOCO_GL=${MUJOCO_GL:-egl}
exec "${CMD[@]}"
