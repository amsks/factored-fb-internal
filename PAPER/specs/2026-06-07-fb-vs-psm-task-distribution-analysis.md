# FB vs PSM — training task-distribution analysis (handoff spec)

**Date:** 2026-06-07 · **Status:** designed + approved, NOT yet built. Blocked on
fresh plain-FB checkpoints (`fb_state_plain`, ETA ~13:50 2026-06-07).

## Goal

Characterize and compare the **training task distribution** that state-based **FB**
and **PSM** train on, for the cube-manipulation dataset. Both train the actor /
forward(SF) net on `sample_mixed_z`: a 50/50 mix of random-sphere `z` and a
**goal-relabelled** task vector — `z = B(s')` for FB, `z = phi(s')` for PSM — where
`s'` are next-states drawn uniformly from the offline cube buffer. So the "training
task distribution" is the distribution of those goal-derived `z`.

User chose the **full comparison (all three probes)**.

## Methods / checkpoints

- **FB = `fb_state_plain`** (fresh re-run, canonical plain FB: `domain=cube_single`,
  `ortho_coef=1000`, `lr_b=1e-4`, `L_dim=50`, no goal_cond/fixed_b/onestep). 10 seeds.
  Checkpoints (when training finishes):
  `/dev/shm/factored-fb/runs/<ts>__fb_state_plain__s{0..9}/checkpoints/final.pt`
  with config at `.../​.hydra/config.yaml`.
  - **Dev stand-in while training:** `fb_state_L256` (config-identical plain FB despite
    the name — see [[fb-state-run-name-map]]) at
    `/dev/shm/factored-fb/runs/20260603_064553__fb_state_L256__s{0..9}/`.
- **PSM = `psm_flowbc_state`** (10 seeds, already trained):
  `/dev/shm/factored-fb/runs/20260603_071856__psm_flowbc_state__s{0..9}/`.

DO NOT use `fb_state_v1` (goal_cond), `fb_state_v2` (fixed_b), `*onestep*`, or
`covbal_*` — those are different variants.

## Shared front-end (`evals/task_distribution.py`, testable)

1. Load offline buffer once: `data.ogbench.load_ogbench_dataset(domain="cube_single",
   data_path="/dev/shm/factored-fb/datasets", obs_type="state")`. Sample **M≈50k**
   next-states `s'` = the training-goal pool (training draws goals uniformly from these).
2. **Label each `s'` by interaction mode.** State layout (from grasp_feature_probe.py):
   proprio `0:12` arm joints, `12:17` effector pos+yaw, `17:19` gripper (open+contact);
   cube xyz at `physics[14:17]` (cf. FB `_coverage_weights`). Reuse
   `evals.phase_probe.Thresholds(eps_reach, delta_lift)`. Buckets:
   *no-contact / reaching / grasped / lifted-or-transport*. (VERIFY the buffer stores
   `physics`; if not, derive cube xyz from the obs vector — confirm obs layout, obs is 28-d.)
3. Load each checkpoint's backward map via `evals.analysis.{load_cfg, build_env_and_agent,
   load_checkpoint}`. Embed the pool → training-z cloud: FB `model.backward_map(s')`
   then `model.project_z`; PSM the `phi`-based equivalent (check `agents/psm/model.py` —
   `model.phi(...)` + `project_z`; mirror `PSMAgent.sample_mixed_z`).

## Three probes (per (method, seed) → aggregate across seeds)

1. **Goal states, interaction-resolved** — histogram the goal pool over interaction
   buckets (data-side, shared) + the same weighted into z-space. Headline: fraction of
   training tasks involving cube contact/lift vs arm-only.
2. **Task-embedding geometry** — over the z cloud: norm distribution; PCA participation
   ratio `(Σλ)²/Σλ²` (effective dim); sphere coverage; linear separability (AUC) of
   grasped/lifted vs free in z (does the task embedding resolve the interaction var?).
   FB vs PSM side by side.
3. **Train-vs-eval coverage** — infer each of the 5 cube eval tasks' z via
   `agent.eval_context(env, domain, task)` (see `evals/ogbench.py`); score each against
   the training z cloud (cosine-to-nearest + Mahalanobis / kNN density) → in/out of
   distribution, per method.

## Outputs / layout

- `evals/task_distribution.py` — pure logic (label_interaction, participation_ratio,
  separability_auc, coverage_score, embed_pool). **TDD** these on synthetic data.
- `scripts/task_distribution_probe.py` — driver: loops methods×seeds, writes
  `analysis/task_distribution/<method>/s<seed>.json` + figures, an aggregate, `summary.md`.
- Tests in `tests/test_task_distribution.py`.
- Plotting/IO exempt from TDD.

## Run notes

- Use `.venv/bin/python` (NOT `uv run`). wandb entity `amsks` if anything logs.
- CPU is fine (embedding 50k states). No GPU needed for the analysis.
- Trigger: wait until `fb_state_plain` `final.pt` files exist for all 10 seeds
  (`tmux ls` shows the session; per-seed logs at `/dev/shm/fb_state_plain_<ts>/seed_*.log`
  reaching `[step  1000000]`). Then point the FB input at `fb_state_plain` and run.
