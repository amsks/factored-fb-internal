#!/usr/bin/env bash
# waits for the b_norm=false probe's walker s0 250k checkpoint, then runs the B-spread diagnostic (CPU)
cd "$HOME/git/Factored-FB-seam"
RUN=exp/exorl_dfb_bnorm0/dmc-walker_density_fb_z_ddpgbc_seed_0
for i in $(seq 1 60); do [ -f $RUN/params_250000.pkl ] && break; sleep 300; done
[ -f $RUN/params_250000.pkl ] || { echo "no 250k checkpoint after 5h"; exit 1; }
sleep 60
JAX_PLATFORMS=cpu MUJOCO_GL=egl FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache \
  /dev/shm/.venv-jax/bin/python docs/notes/2026-09-18-density-fb-exorl-impl-audit/bspread_run.py walk $RUN 250000 2>&1 | grep -E "^DFB|zenc tower|online tower|INIT tower|eval z|R2|^FB" 
echo "--- in-run eval at 250k:"; grep -E "^eval 250000" logs/exorl_bnorm0/walker_s0.log | grep -o "'episode_return': [0-9.]*"
