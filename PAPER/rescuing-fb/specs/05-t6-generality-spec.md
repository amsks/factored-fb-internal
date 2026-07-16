# T6 — zero-shot generality of the reward→z→policy API (pre-registered protocol)

*2026-07-16. Written BEFORE any rollout ran. Probe on the G4 z-conditioned checkpoint
(`~/bridge_ckpt_backup/gciql_outputs/factored-fb/gciql_g4_zcond/sd000_20260616_193331/`,
`params_1000000.pkl`, GCBilinearValue d=512, actor_cond='z'). Question (02-rgfb-spec §1.3, §4
item 2, §6): the agent trained z only at ψ(g) for single dataset goals; does moment inference
z = E_ρ[r·ψ(s′)] produce sensible policies for rewards OUTSIDE the single-goal family? Decides
"zero-shot RL" vs "zero-shot goal-reaching" claim scope. Must be re-run on the RG-FB v0
checkpoint @500k as confirmation (this checkpoint is d512/1M, actor_vf UN-conditioned).*

## 0. Pinned facts (verified from source/dataset before writing, no rollouts)

- Obs layout (`third_party/ogbench/ogbench/manipspace/envs/cube_env.py:766-794`): cube xyz at
  `obs[19:22]` = (block_pos_m − [0.425, 0, 0]) × 10 ⇒ **1 obs unit = 10 cm confirmed**;
  runtime check vs `qpos[14:17]` agrees to 1e-17. All geometry below in METERS
  (cube_m = obs[19:22]/10 + [0.425,0,0]).
- Eval goals (meters): t1 (0.425,−0.10,0.02) · t2 (0.500,0.00,0.02) · t3 (0.350,0.00,0.02) ·
  t4 (0.500,0.20,0.02) · t5 (0.500,−0.20,0.02).
- Table height: cube resting z ≈ 0.0200; **p95 of cube_z among r>0 dataset states = 0.031**
  (max over the 5 tasks' relabelings: 0.0316) ⇒ lift threshold z_lift = 0.031 + 0.10 =
  **0.131 m**. Dataset support: 22–24% of states have cube_z > 0.131 (lift is data-supported).
- Dataset = 1M cube-single-play transitions, shared across singletask variants (only the
  relabeled reward differs).

## 1. Inference pipeline (shared, frozen)

- **Relabel sample**: ONE fixed subsample of 50,000 dataset states, `np.random.default_rng(0)`,
  reused for every reward family. Ψ = ψ(s′) via the value net's B (`info=True` path), computed
  once.
- **Moment inference**: z_raw = Σᵢ rᵢ·ψ(sᵢ′) with synthetic r ∈ {0,1} computed from the decoded
  cube xyz of each dataset state.
- **Norms (both reported, as in E1a)**:
  - `match_data` (PRIMARY for all synthetic families): ‖z‖ rescaled to mean ‖ψ(s′)‖ over the
    50k sample — the goal-free hygiene rule of 02-rgfb-spec §1.3.
  - `sqrtd`: ‖z‖ = √512 (secondary).
  - `match_g` (anchors only, E1a-faithful): ‖z‖ = ‖ψ(g_task)‖ — defined only when a goal
    observation exists; used for the harness gate.
- **Rollouts**: 25 episodes per (reward, task-variant env, norm), H=200, reset seeds 0..24,
  actor noise PRNGKey(0) split per step, temperature 0. Batched lockstep across the 25 envs
  (one jitted `sample_actions_with_z` call per step); an env that terminates/truncates is
  frozen and its last cube position padded forward. Success metrics computed from the recorded
  cube-xyz trajectories.
- **Compute**: CPU-only jax (`JAX_PLATFORMS=cpu`, `MUJOCO_GL=egl`), **capped at cores 0-47
  (`taskset -c 0-47`) + OMP/MKL/OPENBLAS_NUM_THREADS=4** (box shared, user directive).
  Long stage in tmux `t6`.
- **Diagnostics logged per reward**: relabel positive count, ‖z_raw‖, cos(z, ψ(g_i)) for all 5
  eval goals + the nearest one, cos(z, mean ψ) (the uniform-z direction).

## 2. Reward families

a. **Two-goal mixture** — r(s) = max(1[‖cube−g₁‖<0.04], 1[‖cube−g₂‖<0.04]) for ALL 10 pairs of
   the 5 eval goals. Task-variants: the two member envs (their start distributions), 25 eps
   each. Metric: success = cube within 0.04 m of EITHER goal at any step; also log which goal
   was hit first per episode and the binary entropy of that choice pooled per pair.
b. **Lift-anywhere** — r(s) = 1[cube_z > 0.131] (= table p95 + 0.10, §0). Task-variants: all 5
   envs. Metric: fraction of episodes with cube_z above threshold for ≥10 CONSECUTIVE steps.
c. **Region-goal** — r(s) = 1[cube ∈ 6 cm axis-aligned box centered at c] for 3 centers that
   are NOT eval goals but ARE inside dataset support. Selection rule (fixed before rollouts):
   grid of (p20..p80) xy dataset quantiles at resting z; keep min-dist-to-any-goal ≥ 0.06 and
   ≥1% of dataset states within 3 cm; greedy max-min-pairwise-distance pick of 3:
   - R1 = (0.3515, 0.1720, 0.02)  (goal-dist 0.151)
   - R2 = (0.4251, −0.1708, 0.02) (goal-dist 0.071)
   - R3 = (0.4980, 0.0848, 0.02)  (goal-dist 0.085)
   Task-variants: all 5 envs. Metric (PRIMARY): cube at FINAL step inside the box (|Δ|≤0.03
   per axis) or within 0.04 m of the center; any-step version logged as secondary.
d. **Negative control** — r(s) = 1 uniformly (no task information; z ∝ mean ψ). Task-variants:
   all 5 envs. The same rollouts are scored against EVERY metric above (5 single goals, lift,
   3 regions). Prediction: degenerate z, low/random success everywhere — if this scores well,
   the metrics are broken.
e. **Positive anchors (harness gate)** — the 5 single-goal tasks through the SAME pipeline.
   For the gate, r = the dataset's own relabeled reward (`rewards`+1) restricted to the shared
   50k sample, norm `match_g` — the exact E1a operating point (E1a: 0.955). The synthetic
   indicator 1[‖cube−g‖<0.04] is also run (agreement with the dataset reward logged) and the
   anchors are additionally reported under `match_data`/`sqrtd` as the within-norm reference
   for the synthetic families.

## 3. Pre-registered read (binding, decided before results)

- **GATE**: (e) under `match_g` must reproduce ≥0.90 mean over the 5 tasks (E1a anchor 0.955).
  If not, STOP and debug the harness; do not run or interpret (a)–(d).
- Read on the PRIMARY norm (`match_data`), mean over task-variants:
  - **CONFIRMED** (paper may say "zero-shot RL" within dataset-supported rewards): mean (a)
    ≥ 0.7 AND mean (c) ≥ 0.5.
  - **PARTIAL** (interpolation within the goal hull only): (a) ≥ 0.7 but (c) < 0.5.
  - **REFUTED** (only "zero-shot goal-reaching" is licensed): only (e) works ((a) < 0.7).
  - (b) is EXPLORATORY either way (reward far off the goal family; honest prior: fails). It
    can upgrade the language ("including a non-goal reward") but not the verdict.
  - (d) must stay low on every metric or the metrics are invalidated.
- Outputs: `analysis/rescuing-fb/t6_generality_g4.json` + `analysis/rescuing-fb/t6_notes.md`
  + dated DECISIONS.md entry (noting the RG-FB v0 @500k re-run obligation).
