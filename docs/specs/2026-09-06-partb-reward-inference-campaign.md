# Part B — reward-inference (td_jepa-protocol) evaluation campaign: handoff (2026-09-06)

For an agent starting on a cluster with this repo cloned at branch `density-fb`. Everything
referenced is committed. Read `docs/specs/2026-09-03-cluster-run-matrix.md` §"Cluster ops"
first for env vars, datasets and XLA flags.

## Why this campaign exists (one paragraph)

The paper reports two evaluation routes and must never mix them in one column.
**Route A, goal-conditioned:** the policy is handed the goal observation; success at the final
step; this is OGBench's standard protocol and every baseline table (`PAPER/ICLR/table_*`).
**Route B, reward inference (td_jepa's protocol):** 10,000 random dataset transitions are
relabelled with the task reward from simulator state (cube within 4 cm of goal ⇒ 1, else 0,
i.e. `cube_relabel_reward + shift 1.0`), the method's own `infer_cond` turns the samples into
a task index, the goal frame is never given, success is any-step over 10 episodes per task.
Route B is the zero-shot interface the paper claims. Under it HILP is a strong baseline
(cube-single 0.83) that Route A read as inert (0.14) — same checkpoints. Density-FB has no
Route-B number yet: its goal-conditioned checkpoints have no reward-inference path, so the
latent-index variant `density_fb_z` must be trained and scored.

Runs already done under Route B (1M, 3 seeds, td_jepa cube configs):
`exp/cube-{single,double}-play-v0_hilp{,_density}_tdjc_flowbc_seed_{0,1,2}` and
`exp/cube-single-play-v0_fb_tdj_flowbc_seed_{0,1,2}`. Results: HILP cs 0.83±0.03 / cd 0.08;
Density-HILP cs 0.74±0.08 / cd 0.01; FB cs 0.38±0.07. (`eval_relabel.csv` in each dir.)

## Tier 1 — the 9 runs (do these)

    DRY=1 bash scripts/launch_partb_tier1.sh      # job list
    bash scripts/launch_partb_tier1.sh            # local tmux, one worker per GPU
    sbatch scripts/partb_tier1.slurm              # cluster: array over the same job list

| critic | domain | seeds | override | note |
|---|---|---|---|---|
| `density_fb_z` | cube_single | 0 1 2 | — | seed 0 has a 500k ckpt; `--auto_resume` continues it to 1M |
| `density_fb_z` | cube_double | 0 1 2 | `measure_hit_spec=19:22:0.4,28:31:0.4` | two-cube kernel; MUST be passed (unquoted, comma-joined) |
| `fb_tdj` | cube_double | 0 1 2 | — | cube_single already done |

All at `--train_steps 1000000 --eval_interval 100000 --save_interval 250000` with
`--override eval_relabel_size=10000 eval_relabel_shift=1.0 eval_relabel_episodes=10`.
Each run writes `eval.csv` (Route A; for `density_fb_z` this is the own-pool reward-sample
route) and `eval_relabel.csv` (Route B). ~10 GPU-hours per run on an A100-class GPU.

**Gates before any number is quoted.**
1. `eval_relabel.csv` exists and its `inferred_cond_goals_dropped` column is > 0 for every
   row (the goal frame was actually withheld; see `scripts/relabel_rescore.py` docstring).
2. Final-step rows only; never a mid-run eval (cube evals swing ±0.2 between 800k and 1M).
3. Report per seed. Seed count and episode count in every caption.
4. Density-FB z route on cube-single should land near its own-pool 0.82 (500k, 1 seed). A
   result below ~0.70 is the pool-composition sensitivity recorded in
   `docs/notes/2026-09-04-density-fb-central-perspective.md` §7 and memory; report it, do
   not tune toward the number.

## Tier 1b — eval-only rescoring of existing checkpoints (no training)

    python scripts/relabel_rescore.py --env cube-single-play-v0 \
        --critics density_fb_z rldp psm fb_ref --seeds 0 1 2 --epoch 500000 \
        --out analysis/relabel_rescore_cs500k.jsonl

Gives Route-B numbers for the 500k cube checkpoints of RLDP, PSM, FB(ref) and the one
`density_fb_z` seed. Footnote them as 500k. A first pass was started on this box on
2026-09-06 (`analysis/relabel_rescore_cs500k.jsonl`); check whether it completed.

## Tier 2 — scene (needs code first)

`env/ogbench.py::relabel_infer_batch` is cube-only (`goal_xyzs`). Scene needs a relabel
reward on drawer/window/button state from qpos, matched to OGBench's scene success rule, plus
a test. Then: `density_fb_z`, `fb_tdj`, `hilp_tdjc`, `hilp_density_tdjc` on `scene`, 3 seeds,
same flags. Do not start Tier 2 runs before the reward has a passing test.

## Tier 3 (optional) — RLDP/PSM at 1M in td_jepa configs, cube_single + cube_double,
vanilla + density, 3 seeds = 24 runs. Only if a full method × route grid is wanted.

## Scoring and handback

- Aggregate with a pandas one-liner over `eval_relabel.csv` final rows (see the summary in
  `docs/notes/2026-09-04-exorl-rldp-hilp-results-table.md` for the format), or extend
  `scripts/relabel_rescore.py`.
- Hand back the per-run table (critic, env, seed, final step, Route A success, Route B
  success, goals_dropped) and the checkpoints. Do NOT call the paper readings; the eval box
  does.
- Consolidate checkpoints: `python scripts/hf_upload_checkpoints.py` uploads final params +
  config + eval CSVs of every run under `exp/` to the private hub repo
  `amsks/factored-fb-checkpoints` (manifest at repo root). Incremental; safe to re-run.
- td_jepa's published cube numbers must be copied from their paper's OGBench table for the
  comparison row; note their number is best-of-2×2-sweep over 10 seeds vs our pinned
  BASE_CFG at 3 seeds (`scripts/launch_hilp_tdjc.sh` header).

## Do not
- Mix Route A and Route B numbers in one column, or 500k and 1M in one column.
- Quote `eval.csv` of a `density_fb_z` run as "goal-conditioned": it is the own-pool
  reward-sample route.
- Launch with `&`/`nohup`; use tmux or slurm. Never `XLA_PYTHON_CLIENT_PREALLOCATE=true`.
