# Goal-conditioned vs z-conditioned training, object coords vs full state — the ablation matrix (2026-09-06)

Cluster handoff. Branch `density-fb` at or after the commit that adds this file. Read
`docs/specs/2026-09-03-cluster-run-matrix.md` §"Cluster ops" for env vars, datasets and XLA
flags, and `docs/specs/2026-09-06-partb-reward-inference-campaign.md` for the two evaluation
routes (Route A goal-conditioned, Route B td_jepa reward inference) — every run here logs both.

## Question

Does goal-conditioned training (the task enters as a real hindsight goal, the actor's reward is
that goal's object-coordinate kernel) help HILP and RLDP the way it helps Density-FB, and does
that depend on whether the backward map reads object coordinates or the full observation?

## The 2 x 2 per family, and how each cell is trained

| family | z-conditioned (mixed random/goal index, latent reward) | goal-conditioned (goal index every row, kernel reward) |
|---|---|---|
| Density-FB | `density_fb_z` (obj) / `density_fb_z_state` (state) — actor reward from the SLOW CENTERED index tower (`measure_density_z_actor_reward=latent_centered`, the 2026-09-06 fix) | `density_fb` (obj) / `density_fb_state` (state) — the paper's method |
| HILP-density | `hilp_density_oc` (obj) / `hilp_density_cube` (state) | `hilp_density_gk_oc` / `hilp_density_gk` — `goal_kernel=true` (NEW: whitened goal index + kernel reward) |
| RLDP-density | `rldp_density_oc` / `rldp_density` | `rldp_density_gk_oc` / `rldp_density_gk` — `goal_kernel=true` |
| HILP (vanilla) | `hilp_oc` / `hilp_cube` (`hilp_mix_ratio` 0.5) | `hilp_gc_oc` / `hilp_gc` — `hilp_mix_ratio 1.0`: every row's z is a whitened hindsight-goal index |
| RLDP (vanilla) | `rldp_oc` / `rldp` (`train_goal_ratio` 0.5) | `rldp_gc_oc` / `rldp_gc` — `train_goal_ratio 1.0`: every row's z = B(hindsight goal) |

Vanilla HILP/RLDP have no place for a kernel reward (Q = ψᵀz / Fᵀz), so their goal-conditioned
cell is "all-goal indices", the closest analogue. Every `_oc`/`hit` arm on cube_double takes
`measure_hit_spec=19:22:0.4,28:31:0.4` (the launcher adds it).

## Already done (do not rerun; the launcher skips any run whose 1M checkpoint exists)

At 1M: `rldp`, `rldp_density` cube-single x3; `hilp_tdjc`, `hilp_density_tdjc` both envs x3
(td_jepa configs; a different HILP calibration — keep as their own row, not as `hilp_cube`);
`fb_tdj` cube-single x3. Density-FB goal-conditioned 1M numbers live in `table_main_1m.tex`
(bridge stack); the seam 1M cells are re-run here so the table is one stack.
At 500k, 3 seeds, both envs: every z-conditioned cell above and the Density-FB goal cells.
Locally in progress at 500k: `density_fb_z` and `density_fb_z_state` cube-single
(`exp/zfix/lc_hard*`), the fix screen; the cluster runs them to 1M from scratch.

## Launch

    DRY=1 bash scripts/launch_ablation_matrix.sh            # pending job list (114 as of 09-06)
    TIER=1 sbatch scripts/ablation_matrix.slurm              # Density-FB 2x2, 24 runs  <- first
    TIER=2 sbatch scripts/ablation_matrix.slurm              # density retrofits 2x2, 45 runs
    TIER=3 sbatch scripts/ablation_matrix.slurm              # vanilla HILP/RLDP 2x2, 45 runs

Set `REPO`, `PY` for the cluster layout; the array skips finished runs and `--auto_resume`
continues partial ones, so resubmitting after preemption is safe. ~10 GPU-h per run.

## Gates and handback (per run; no readings)

1. `params_1000000.pkl`, `eval.csv`, `eval_relabel.csv` present; `inferred_cond_goals_dropped > 0`
   on every relabel row.
2. Final-step rows only; per-seed table: family, conditioning, B input, env, seed, Route A
   success, Route B success. Never average across seeds in the handback.
3. `python scripts/hf_upload_checkpoints.py` at the end (final params + config + CSVs to
   `amsks/factored-fb-checkpoints`; incremental).
4. Route B on goal-conditioned arms uses each family's own reward-to-index rule (HILP least
   squares, RLDP moment, Density-FB centered encoder); report it, but the deciding column for
   goal-conditioned arms is Route A.

## Do not
- Change any `measure_density_z_*`, `goal_kernel`, `hilp_mix_ratio`, `train_goal_ratio` values.
- Mix 500k and 1M, or Route A and Route B, in one column.
- Start `hilp_density_gk*` / `rldp_density_gk*` on scene: the kernel is cube-only.
