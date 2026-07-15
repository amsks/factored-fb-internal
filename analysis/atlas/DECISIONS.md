# Value-Function Atlas — execution decisions

## Split-oracle route for C1/C2 (spec §4) — RESOLVED: dataset states

Task 3 gate, 2026-07-14. 16 contact-rich dataset rows (obs[18]>0.5, cube lifted,
>=20 in-episode steps remaining), qpos/qvel restore, replay recorded actions for 20 steps,
compare to recorded continuation.

- cube-xyz drift: median 0.0024 m, p90 0.0030 m, max 0.0044 m
- full-obs drift: median 0.068, max 0.175 (this is the ARM, via the unrestorable
  `_target_effector_pose` IK integrator; it does not move the cube)
- 16/16 rows under the 0.04 m success threshold

**Decision: C1/C2 use literal dataset states with qpos/qvel restore.** The unrestorable IK
integrator perturbs the arm observation but the cube (what the 4cm criterion measures) stays
within ~4mm over a 20-step horizon. The μ-rollout fallback is NOT needed.

Caveat carried forward: the arm-obs residual means any *arm-dependent* oracle claim at C1/C2
should be read with the ~0.07-0.18 obs perturbation in mind; cube-placement claims are safe.

## Critic registry + checkpoint step policy (Batch 2) — RESOLVED

Checkpoints live on the PERSISTENT disk at
`~/bridge_ckpt_backup/gciql_outputs/factored-fb/<group>/` (NOT /dev/shm, which is volatile RAM
and had been cleared). Identity confirmed by `agent_name` in flags.json, not the group label.

| critic          | group                              | agent_name  | arch     | actor  | STEP USED |
|-----------------|------------------------------------|-------------|----------|--------|-----------|
| bilinear_flow   | GCIQL-State-Flow-Bilinear-d50      | gciql_flow  | bilinear | flow   | **500k**  |
| bilinear_ddpgbc | GCIQL-State-Flow-Bilinear-d50-ddpgbc| gciql_flow | bilinear | ddpgbc | **500k**  |
| mlp_flow        | GCIQL-State-Flow                   | gciql_flow  | MLP      | flow   | **1M**    |
| mlp_ddpgbc      | factored-fb-gciql                  | gciql (stock)| MLP     | gaussian ddpgbc | **500k** |
| simplesf        | GCIQL-State-Flow-SimpleSF-Arch     | gciql_flow  | SF       | flow   | **500k**  |
| fb              | factored-fb_outputs/*fb_flowbc*    | torch FB    | FB       | flowbc | final     |

Step policy: 500k for all EXCEPT mlp_flow, which saved no 500k intermediate (only 1M, every
seed). Using its 1M is a CONSERVATIVE confound: the MLP side of "the win" comparison gets 2x
the training and still loses to 500k bilinear, so the bilinear advantage is understated, not
inflated. Recorded per user decision 2026-07-15.

Notes: mlp_ddpgbc (stock `gciql`) needs a SEPARATE jax loader path (not gciql_flow); its seeds
start at sd001 (no sd000) -- use the three lowest seeds consistently. simplesf secured to
persistent disk (was only in volatile /dev/shm).

## jax runtime (Batch 2) — VERIFIED via end-to-end smoke test

- Atlas jax code MUST run under `/dev/shm/.venv-jax/bin/python` (jax 0.10.0). The repo `.venv`
  has NO jax. This venv is on VOLATILE RAM -- gone on reboot; rebuild if missing.
- Required sys.path inserts (from oodgrid_run_jax.py:18-21), in this order:
  `R, R/scripts, R/third_party/ogbench/impls, R/tools/wandb_mode_shim`  (R = repo root).
  Without `tools/wandb_mode_shim` on path, `import gciql_flow` fails on `ogbench_flow`.
- Load: `restore_agent(GCIQLFlowAgent.create(0, zeros(1,28), zeros(1,5), cfg), CK_DIR, step)`
  where cfg = get_config() updated with flags.json['agent'] intersecting keys, encoder=None.
- VERIFIED call signatures on a real bilinear_flow @500k checkpoint:
  - `agent.network.select('critic')(obs[B,28], goals[B,28], actions[B,5]) -> (q1[B], q2[B])`
  - `agent.network.select('actor_vf')(obs[B,28], actions[B,5], t[B,1]) -> vel[B,5]`
  - `agent.sample_actions(observations=, goals=, seed=PRNGKey, temperature=0.0) -> [B,5]`
- Run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. CPU-only eval via `CUDA_VISIBLE_DEVICES=""`.
- TEST SPLIT: test_config/prefix_replay/split_oracle run under `.venv`; test_mu_flow/critics/
  build_atlas must run under the jax venv (they import jax). Document per-module.

## Batch 2 built artifacts (2026-07-15)

- `analysis/atlas/banks.npz` (1.8 MB): 4 phases x 32 states, A_mu [4,32,256,5], built from
  bilinear_flow's mu (228s). All phases filled 32/32 — P3 (transport) reached in ~3 episodes,
  so the spec's "mu rarely reaches transport" risk did NOT bite (this mu is a competent grasper).
- End-to-end atlas verified on live checkpoints: ||A||_F=1.0; over the 5 eval goals bilinear
  top-sv 0.969 < mlp 0.980 (bilinear more mode-distributed — a first multi-modality signal,
  n=1 state). rank=5 is the n_g ceiling here; use the 64-goal grid for real effective-rank.
