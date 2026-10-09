---
name: antmaze-gciql-vs-fb-smoke
description: "RUNNING (2026-06-24): GCIQL-flow-bilinear-d50 vs FB-flowbc(d50) on antmaze-medium-navigate, 3 seeds x 500k — does the cube bridge story hold on navigation?"
metadata: 
  node_type: memory
  type: project
  originSessionId: 07642137-20e6-48e2-87d9-fa02d091fc99
---

**Goal:** test whether the FB↔GCIQL bridge result (GCIQL-bilinear ≫ FB on cube) also
holds on **antmaze-medium-navigate-v0** (navigation), or whether FB is actually fine on
navigation (the manipulation-is-hard thesis predicts FB should do *better* here than on cube).
Rank-matched: both are d=50 + flow actor.

**RUNNING (launched 2026-06-24 ~22:50 UTC, tmux `antmaze500`):** 3 seeds × **500k** each.
- GCIQL: `run_gciql.py --config-name cube_single_state_gciql_flow_bilinear_d50 env_name=antmaze-medium-navigate-v0`
  (GPUs 0,1,2; eval_episodes=50 eval_on_cpu=1 eval_interval=100000). Output
  `/dev/shm/gciql_outputs/factored-fb/GCIQL-AntMaze-Medium-Flow-Bilinear-d50/sd00*/eval.csv` (`evaluation/overall_success`).
- FB: `train.py domain=antmaze_medium ortho_coef=100 lr_b=1e-5 data_path=/dev/shm/datasets`
  (GPUs 3,4,5; wandb group `fb_antmaze_medium_500k`, metric `eval/reward/eval/success`).
- Logs `/dev/shm/antmaze_smoke_logs/{gciql500,fb500}_s*.log`. ETA ~3h → ~2026-06-25 ~02:00 UTC.

**50k pilot result (superseded by 500k):** FB ≈ **4.7%** (seeds 6/4/4%), GCIQL didn't finish.
50k is far too short for antmaze (FB needs ~500k–1M) — hence the 500k extension.

**GOTCHAS learned (antmaze GCIQL via cube config):**
- No GCIQL antmaze config exists (cube-only) — override `env_name=antmaze-medium-navigate-v0` on the d50 config.
- ogbench `main.py` **evals at hardcoded `i==1`** over all **5 tasks** × eval_episodes; flow-actor
  rollout is slow. Keep eval cheap.
- **`eval_on_cpu=0` is 90× SLOWER** for single-env flow rollout (100s/ep vs 1.1s/ep) — per-step
  host↔device round-trip ([[ogbench-drq-augment-host-bound]]). ALWAYS keep `eval_on_cpu=1` for flow-actor eval.
- `run_gciql.py` argv builder does NOT pass `eval_tasks`; only eval_episodes/eval_on_cpu are CLI-controllable.
- bc_coeff=3.0 in the cube d50 config is cube-tuned (FB antmaze uses 0.3) — a possible confound, left as-is for the smoke.

Conventions: [[wandb-entity-amsks]], [[jax-gpu-packing-preallocate]], [[always-run-in-tmux]], [[gciql-flow-agent]].
Relates to the cube bridge: [[handoff-bridge-10seed-1M-campaign]].
