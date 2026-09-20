#!/usr/bin/env bash
# Score the 1M-recipe + reference-actor-lr Density-FB batch (exp/exorl_dfb_1mrecipe, 4 domains x 5 seeds) under the
# paper protocol (10 eps, 50k reward samples, params_1000000) and print per-domain mean / pop-std over seeds.
# Usage: bash scripts/score_exorl_1mrecipe.sh   (4 scorer jobs on GPUs 0-3, ~40 min; idempotent symlink layout)
set -euo pipefail
cd "$HOME/git/Factored-FB-seam"
export FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages EXORL_DATA_ROOT=$HOME/exorl_data/exorl/datasets EXORL_CACHE_ROOT=$HOME/exorl_data/relabel_cache XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= WANDB_MODE=offline MUJOCO_GL=egl
PY=/dev/shm/.venv-jax/bin/python; R=exp/exorl_disc_score_1m; OUT=exp/paper_eval_exorl_1mrecipe.jsonl; mkdir -p logs/exorl_score
g=0
for dom in dmc-walker dmc-cheetah dmc-quadruped dmc-point-mass-maze; do
  mkdir -p $R/$dom/density_fb_z
  for s in 0 1 2 3 4; do ln -sfn ../../../exorl_dfb_1mrecipe/${dom}_density_fb_z_ddpgbc_seed_$s $R/$dom/density_fb_z/${dom}_density_fb_z_ddpgbc_seed_$s; done
  TS=tdjepa; [ "$dom" = dmc-point-mass-maze ] && TS=paper
  CUDA_VISIBLE_DEVICES=$g $PY scripts/exorl_paper_eval.py --root $R --critic density_fb_z --actor ddpgbc --domains $dom --seeds 0 1 2 3 4 --steps 1000000 --episodes 10 --inference-samples 50000 --task-set $TS --jobs 2 --out $OUT > logs/exorl_score/1mrecipe_$dom.log 2>&1 &
  g=$((g+1))
done
wait
python3 - <<'PY'
import json,collections,statistics as st
rows=collections.defaultdict(lambda: collections.defaultdict(dict))
for line in open("exp/paper_eval_exorl_1mrecipe.jsonl"):
    r=json.loads(line); rows[r["domain"]][r["seed"]][r["exorl_task"]]=r["episode_return"]
for d,seeds in sorted(rows.items()):
    dm=[st.mean(t.values()) for s,t in sorted(seeds.items())]
    tasks=sorted(next(iter(seeds.values())))
    pt={t: round(st.mean(seeds[s][t] for s in seeds if t in seeds[s])) for t in tasks}
    print(f"{d}: n={len(dm)} mean={st.mean(dm):.1f} popstd={st.pstdev(dm):.1f} per-seed={[round(x) for x in dm]} {pt}")
PY
