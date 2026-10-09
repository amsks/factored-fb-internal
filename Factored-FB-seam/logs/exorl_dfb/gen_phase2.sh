#!/usr/bin/env bash
# Generate Phase 2: top up the OGBench Density-FB arms to 5 seeds (seeds 3,4). Two arms,
# faithful copies of the established campaign commands (scripts/tdjtab_manifest.sh and
# scripts/launch_dfb_state.sh) with only the seed changed:
#   with object coordinates    -> critic density_fb_z        save_dir exp/tdjtab
#   without object coordinates -> critic density_fb_z_state  save_dir exp/dfb_state  (full state)
# Both: flowbc actor, 1M steps, per-env object-coord hit-spec, measure_density_hard,
# latent_centered actor reward, Route-B relabel eval. NO ExoRL/MuJoCo-overlay env (OGBench).
# Appends phase-2 rows to logs/exorl_dfb/manifest.tsv; the running scheduler picks them up
# once phase 1 (ExoRL) is fully done.
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
JOBDIR=logs/exorl_dfb/jobs
MANIFEST=logs/exorl_dfb/manifest.tsv
mkdir -p "$JOBDIR"

# domain (short) -> env-dir name used in the run directory
declare -A ENVDIR=(
  [cube_single]="cube-single-play-v0" [cube_double]="cube-double-play-v0"
  [scene_play]="scene-play-v0" [puzzle_3x3]="puzzle-3x3-play-v0"
)
# per-env object-coordinate hit spec (same for both arms; for the state arm the B tower
# ignores it but the eval kernel still uses it)
declare -A HIT=(
  [cube_single]="19:22:0.4"
  [cube_double]="19:22:0.4,28:31:0.4"
  [scene_play]="19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6"
  [puzzle_3x3]="19:21:0.5,23:25:0.5,27:29:0.5,31:33:0.5,35:37:0.5,39:41:0.5,43:45:0.5,47:49:0.5,51:53:0.5"
)
# arm -> "critic save_dir"
declare -A ARM=(
  [oc]="density_fb_z exp/tdjtab"
  [state]="density_fb_z_state exp/dfb_state"
)

for ARMK in oc state; do
  read -r CRITIC SAVE <<< "${ARM[$ARMK]}"
  for DOM in cube_single cube_double scene_play puzzle_3x3; do
    HITSPEC="${HIT[$DOM]}"; EDIR="${ENVDIR[$DOM]}"
    for S in 3 4; do
      NAME="ogb_${ARMK}_${DOM}_s${S}"
      SCRIPT="$JOBDIR/${NAME}.sh"
      RUNDIR="${SAVE}/${EDIR}_${CRITIC}_flowbc_seed_${S}"
      cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "\$HOME/git/Factored-FB-seam"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export WANDB_MODE=offline
export MUJOCO_GL=egl
exec /dev/shm/.venv-jax/bin/python -m impls.main \\
  --actor flowbc --auto_resume \\
  --train_steps 1000000 --eval_interval 100000 --save_interval 250000 \\
  --critic ${CRITIC} --domain ${DOM} --seed ${S} --save_dir ${SAVE} \\
  --override measure_density_z_actor_reward=latent_centered \\
  --override measure_density_hard=true \\
  --override 'measure_hit_spec="${HITSPEC}"' \\
  --override eval_relabel_size=10000 \\
  --override eval_relabel_shift=1.0 \\
  --override eval_relabel_episodes=10
EOF
      chmod +x "$SCRIPT"
      echo -e "2\t${NAME}\t${SCRIPT}\t${RUNDIR}/params_1000000.pkl\tlogs/exorl_dfb/${NAME}.log" >> "$MANIFEST"
    done
  done
done
echo "appended phase-2 jobs; manifest now has $(wc -l < "$MANIFEST") rows"
