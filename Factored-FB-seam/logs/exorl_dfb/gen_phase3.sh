#!/usr/bin/env bash
# Generate per-job shell scripts for Phase 3 (2026-09-17): Density-FB (density_fb_z) on the four ExoRL
# domains, 1M steps, 5 seeds each (20 jobs), with the hit radius RE-CALIBRATED by the
# actor training-reward SUPPORT (scripts/calibrate_hit_spec.py --target_support 85.6, the
# support cube-single has under 19:22:0.4) instead of the random-pair hit rate. Everything
# else is byte-identical to Phase 1 (docs/notes/2026-09-17-exorl-audit.md). Maze keeps its
# benchmark radius (support 64.5 already cube-like). One script per job so the measure_hit_spec
# override keeps its quotes (OmegaConf sexagesimal gotcha) — the scheduler runs these files
# directly and only sets CUDA_VISIBLE_DEVICES in the environment.
# Config = walker_dfbref reference recipe (post FB-audit: value tower [1024], tau 0.001,
# B [256,256] motivo normalized, lr 1e-4, TD3 motivo actor, alpha 0, q_normalize off) plus
# the paper-exact reward-inference eval (--infer_cond_episodes 50000 --eval_on_inferred_cond).
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
JOBDIR=logs/exorl_dfb/jobs
mkdir -p "$JOBDIR"
MANIFEST=logs/exorl_dfb/manifest.tsv   # APPENDED, not truncated
# (phase 3 rows are appended; the running scheduler re-reads the manifest at each phase boundary)

# domain -> "hit_spec discount"
declare -A SPEC=(
  [dmc-walker]="0:24:9.5603 0.98"
  [dmc-cheetah]="0:17:4.6159 0.98"
  [dmc-quadruped]="0:78:6.2467 0.98"
  [dmc-point-mass-maze]="0:2:0.03 0.99"
)

for DOM in dmc-walker dmc-cheetah dmc-quadruped dmc-point-mass-maze; do
  read -r HIT DISC <<< "${SPEC[$DOM]}"
  for S in 0 1 2 3 4; do
    NAME="${DOM}_dfbz_sup86_s${S}"
    SCRIPT="$JOBDIR/${NAME}.sh"
    SAVE="exp/exorl_dfb_sup86/${DOM}/density_fb_z"
    RUNDIR="$SAVE/${DOM}_density_fb_z_ddpgbc_seed_${S}"
    cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "\$HOME/git/Factored-FB-seam"
export FB_MUJOCO_SITE_PACKAGES=\$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages
export EXORL_DATA_ROOT=\$HOME/exorl_data/exorl/datasets
export EXORL_CACHE_ROOT=\$HOME/exorl_data/relabel_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export WANDB_MODE=offline
export MUJOCO_GL=egl
exec /dev/shm/.venv-jax/bin/python -m impls.main \\
  --critic density_fb_z --actor ddpgbc --domain ${DOM} --seed ${S} \\
  --auto_resume --train_steps 1000000 --eval_interval 250000 --save_interval 250000 \\
  --eval_episodes 10 --num_video 0 --infer_cond_episodes 50000 --eval_on_inferred_cond \\
  --save_dir ${SAVE} \\
  --override discount=${DISC} --override 'measure_hit_spec="${HIT}"' \\
  --override 'value_hidden_dims=[1024]' --override tau=0.001 \\
  --override 'measure_density_b_hidden_dims=[256,256]' --override measure_density_archi=motivo \\
  --override measure_density_b_norm=true --override lr=1e-4 \\
  --override actor.motivo_archi=true --override 'actor.hidden_dims=[1024]' \\
  --override actor.alpha=0.0 --override actor.q_noise_std=0.2 --override actor.q_noise_clip=0.3 \\
  --override actor.boot_noise_std=0.2 --override actor.boot_noise_clip=0.3 \\
  --override actor.q_normalize=false
EOF
    chmod +x "$SCRIPT"
    # phase  name  script  done_marker  logfile
    echo -e "3\t${NAME}\t${SCRIPT}\t${RUNDIR}/params_1000000.pkl\tlogs/exorl_dfb/${NAME}.log" >> "$MANIFEST"
  done
done
echo "manifest now $(wc -l < "$MANIFEST") rows (phase 3 appended) -> $MANIFEST"
