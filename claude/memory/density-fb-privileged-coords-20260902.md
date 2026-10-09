---
name: density-fb-privileged-coords-20260902
description: "2026-09-02 cube campaign: Density-FB at STATE input gets 0.91 (cube-single) — object coordinates are NOT what its win rests on; they rescue the RLDP/PSM/FB retrofits instead. Plus three implementation findings: our FB ships the ortho_coef the reference never uses, PSM is 4x below the paper, and the paper's own RLDP has an SP off-by-one."
metadata:
  node_type: memory
  type: project
---

Cube-single/double, 500k, goal-conditioned eval (z = project_z(B(goal))), 20 eval
episodes, flow actor. Table: `PAPER/ICLR/table_privcoord.tex` ([[primary-paper-dir-iclr]]).

**THE RESULT.** `backward_input: hit_spec` (B/phi consumes the eps-scaled object-coordinate
block slice instead of the full observation) was added to fb.py / hilp.py / density_fb.py,
mirroring the hook rldp.py and psm.py already had. Default off everywhere; the
boot-contract byte-identity gate passes.

cube-single @500k, state input -> object coords:
- **Density-FB 0.91+-0.02 (n=3) -> 0.94+-0.01 (n=3)** — object coordinates barely matter
- Density-RLDP 0.23 -> 0.82 · Density-PSM 0.17 -> 0.88 · PSM 0.12 -> 0.94 · FB 0.11 -> 0.34
- RLDP 0.11 -> 0.53 (0.28/0.78, unstable) · **HILP 0.08 -> 0.09 (inert to everything)**

cube-double @500k: **Density-FB 0.05 -> 0.64** (needs them); every retrofit <= 0.22.

**READ IT THIS WAY.** Density-FB at FULL-STATE input beats every retrofit at full-state
input by 4x. So transplanting the categorical estimator into RLDP/PSM/HILP does NOT
reproduce Density-FB, and the missing ingredient is NOT the object coordinates. What is
left: end-to-end learned outcome representation, goal-kernel reward, direct goal
conditioning, batch 1024, lr 3e-4, no ortho term. **This overturns the earlier reading
("object coordinates are the load-bearing ingredient"), which came from the retrofit rows
alone.** See [[spec18-density-measure-20260806]] for the estimator's own history.

**HARNESS IS SOUND.** Density-FB reproduces its published cells in this harness at this
protocol: 0.94 vs 0.915+-0.098, 0.64 vs 0.674+-0.051. So low baseline cells are the
baselines' problem, not a shared harness defect.

**THREE IMPLEMENTATION FINDINGS** (all verified against source, not inferred):

1. **Our FB ships the one ortho_coef the reference never uses.** `critic/fb.yaml` has
   `ortho_coef: 1.0`, copied from td_jepa's OGBench `BASE_CFG` — but `sweep_cube()` sweeps
   `{100, 1000}` and never uses 1.0, exactly as `sweep_walker` overrides BASE_CFG's 1 on
   DMC. The bridge launch scripts pass 1000, and the user's own ~0.50 anchor was at 1000.
   A line-by-line audit found the jax FB objective (off-diag, diag x P, ortho on B@B.T),
   z-law (50% gaussian / 50% B(permuted next_obs)) and actor (`loss / Qs.abs().mean()`)
   to be FAITHFUL transcriptions — ortho_coef is the only divergence. `fb_ref` arm running.

2. **PSM is 4x below the paper and has had no audit.** vs `Is Manipulation Hard for
   Successor Measures?` (cube-single, 1M, 10 seeds, IQM, goal-conditioned): PSM 0.087 vs
   0.367. RLDP 0.163 vs 0.227 — CIs overlap, replicates. FB has no 1M run. `psm.yaml` also
   carries `ortho_coef: 1.0`.

3. **The paper's OWN RLDP has an SP off-by-one.** `agents/rldp/agent.py:67` slices
   `batch["observation"][:, 1:]` for the self-predictive rollout targets where td_jepa reads
   the NEXT-observation sequence (`traj_observation` = s_{t+2}..s_{t+H}). From rollout step 2
   the predictor is trained that the action does not move the embedding, and the rollout
   never reaches s_{t+h}. Our port corrected it 2026-08-25 (WindowedDataset column offsets).
   So 0.227 (paper) and 0.163 (ours) are not the same algorithm. `rldp_portalign` reinstates
   the bug behind `rldp_sp_port_alignment` and is running at the paper's 1M budget.

**td_jepa HAS NO GOAL-CONDITIONED EVAL.** `metamotivo/evaluations/` holds exactly
`OGBenchRewardEvaluation` and `DMCRewardEvaluation`, both zero-shot reward inference
(launchers: 10 episodes, 10k inference samples). Our whole cube table is goal-conditioned,
so it has no td_jepa counterpart. Their cube runs are also 1M steps, not 500k.
DMC replication does hold: rldp_tdj 95% / hilp_tdj 102% of their aggregates.

**GOTCHAS THAT COST REAL TIME HERE:**
- **eval.csv is one row PER TASK.** Reading the last row gives one task's value, not the
  arm's score. This produced a 5x-wrong cube-double number twice. Always group by step and
  assert 5 task rows before averaging.
- **A reused run directory interleaves two runs' eval.csv** (10 rows at overlapping steps).
  Prefer the per-run log; `exclude_runs` exists for the CSV pipeline.
- Peaks are NOT quotable (adjacent evals swing .05-.12) — final-step or last-k only.
- CUDA graphs must be OFF (`XLA_FLAGS=--xla_gpu_enable_command_buffer=`) or
  `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` kills ~25% of cells.
