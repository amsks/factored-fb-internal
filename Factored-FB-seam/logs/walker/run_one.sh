#!/bin/bash
# usage: run_one.sh CRITIC SEED GPU SAVE_DIR TRAIN_STEPS EVAL_INTERVAL SAVE_INTERVAL EVAL_EPISODES INFER LOG_INTERVAL LOGFILE
# Mirrors scripts/launch_exorl_official.sh's td_jepa branches (ACTOR_OV, per-domain discount/hit spec),
# plus ortho_coef=1 on the two RLDP arms (2026-09-14 walker block).
set -uo pipefail
C=$1; S=$2; G=$3; SD=$4; STEPS=$5; EI=$6; SI=$7; EE=$8; INF=$9; LI=${10}; LOG=${11}
cd /home/adityamohan/git/Factored-FB-seam
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages
export EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= MUJOCO_GL=egl WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=$G
ACTOR_OV=( --override "actor.motivo_archi=true" --override "actor.hidden_dims=[1024]" --override "actor.alpha=0.0"
           --override "actor.q_noise_std=0.2" --override "actor.q_noise_clip=0.3"
           --override "actor.boot_noise_std=0.2" --override "actor.boot_noise_clip=0.3" --override "actor.q_normalize=false" )
case "$C" in
  rldp_tdj|rldp_density_tdj_reg)
    OV=( --override "discount=0.98" --override "rldp_phase1_steps=2000000" --override 'measure_hit_spec="0:24:15.9"'
         --override "ortho_coef=1" "${ACTOR_OV[@]}" ) ;;
  hilp_tdj|hilp_density_tdj_reg)
    OV=( --override "discount=0.98" "${ACTOR_OV[@]}" ) ;;
  *) echo "unknown critic $C"; exit 2 ;;
esac
/dev/shm/.venv-jax/bin/python -m impls.main --critic $C --actor ddpgbc --domain dmc-walker --seed $S \
  --train_steps $STEPS --eval_interval $EI --save_interval $SI --log_interval $LI --eval_episodes $EE --num_video 0 \
  --infer_cond_episodes $INF --eval_on_inferred_cond --auto_resume \
  --save_dir $SD "${OV[@]}" > "$LOG" 2>&1
echo "rc=$? $C s$S" >> "$LOG"
