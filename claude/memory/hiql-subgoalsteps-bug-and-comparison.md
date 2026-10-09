---
name: hiql-subgoalsteps-bug-and-comparison
description: HIQL cube runs used wrong subgoal_steps (25 vs OGBench-reference 10); flow-vs-actor + MLP-vs-bilinear comparison queued with corrected HPs
metadata: 
  node_type: memory
  type: project
  originSessionId: 93f5c9a5-a5c6-42e4-bb06-d18aac9bae68
---

2026-07-01. Audited our HIQL runs vs OGBench reference (`third_party/ogbench/impls/hyperparameters.sh`).

**HP BUG:** OGBench cube-single-play HIQL reference passes `--agent.subgoal_steps=10`; **our cube
HIQL runs used the default `subgoal_steps=25`** (wandb-confirmed on HIQL-State-Stock/Bilinear/Flow-*).
Everything else matched reference (low/high_alpha=3, discount=0.99, expectile=0.7, rep_dim=10). So the
cube HIQL numbers (stock 24.6, bilinear 15.9, flow-both 22.8) are on a **mis-tuned subgoal horizon**
(25 = the antmaze value; cube needs the shorter 10). The bilinear-vs-MLP comparison is internally
valid (both used 25) but absolute numbers + the bilinear interaction may change at sg=10. For
**antmaze-medium-navigate the reference HP = defaults** (subgoal_steps=25, alpha=3) — so antmaze needs
NO override. Refines [[hiql-flow-agent-and-flow-campaign]], [[hiql-gcivl-bilinear]].

**Architecture audit (why HIQL flow ≠ GCIQL):** HIQL's flow port is bit-identical to GCIVL's
AWR-weighted flow-BC (`exp(alpha*adv)*flow_matching`, V-only value), NOT GCIQL's Q-max flow (FQL
`-q.mean()+distill`). HIQL/GCIVL *reweight* BC (bounded by data); GCIQL *extracts* via Q-max — that's
the 90-vs-47-vs-20 gap, not the value form. Bilinear HURTS HIQL because `_BilinearRepValue`
V=<phi(s),psi(goal_rep([s;g]))>/sqrt(d) puts the rank-d bottleneck on the SAME subgoal rep the
hierarchy uses as its action space (GCIQL has no hierarchy → benign). Best untested version for HIQL =
GCIQL-style Q-max flow at the LOW level (subgoal-conditioned IQL Q) — needs new code.

**QUEUED comparison (4 cfgs x 3 seeds @500k, correct HPs), tmux `hiqlcmpwatch`** running
`/dev/shm/hiql_cmp_watch.sh` — gated behind the whole bridge campaign (waits for ablation "launched"
marker + zero `main.py --env_name=` procs), then launches into tmux `hiqlcmp` throttled to <=8/GPU:
- **cube, FLOW actor** (`hiql_flow`, flow_levels=both, subgoal_steps=10): `..._flow_both_mlp_sg10` +
  `..._flow_both_bilinear_d50_sg10` → groups `HIQL-State-Flow-both-{MLP,Bilinear-d50}-sg10`.
- **antmaze-medium, DEFAULT actor** (native `agents/hiql.py`, defaults): `antmaze_medium_hiql_mlp` +
  `antmaze_medium_hiql_bilinear_d50` → groups `HIQL-AntMaze-Medium-{MLP,Bilinear-d50}`.
All 4 configs CPU-smoke-passed (40 steps + eval, env loads, sg=10 on cube / default on antmaze, 0 err).
Purpose: (cube) valid MLP-vs-bilinear at correct HP; (antmaze) does bilinear hurt a WORKING HIQL —
the clean test of the audit's rank-d-bottleneck hypothesis (antmaze is HIQL's home turf, no cube
mis-tuning confound). Configs UNCOMMITTED. [[always-run-in-tmux]] [[jax-gpu-packing-preallocate]]
