---
name: hiql-flow-agent-and-flow-campaign
description: "New hiql_flow agent (flow actor for HIQL, low/high/both knob) + the GCIVL/HIQL flow-vs-stock @500k campaign state"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c50f5c8-10e2-42be-b25e-9a708249ec30
---

2026-06-30. Building out a flow-actor column for the HIQL/GCIVL bilinear-vs-stock table (cube-single state, @500k, 3 seeds, `evaluation/overall_success`).

**Results so far (means):** HIQL-bil-d50 native-AWR **0.159**; GCIVL-bil-d50 native-AWR **0.675**; GCIVL-bil-d50 **flow 0.792** (best; already-run 2026-06-29, runs `ook68szn/qjrkyeow/kga96805`). Flow actor BEATS native AWR for GCIVL-bilinear.

**New agent `tools/wandb_mode_shim/hiql_flow.py`** (`HIQLFlowAgent`, registered in sitecustomize): HIQL IVL value + subgoal goal_rep inherited; one/both Gaussian-AWR actors → AWR-weighted flow-matching BC (mirrors gcivl_flow, reuses GCVectorField). Knob `flow_levels ∈ {low,high,both}` (default low): `low`=action policy flow cond on concat(obs,goal_reps); `high`=subgoal-proposer flow over rep_dim cond on concat(obs,goal), sample→length-normalize. Inherits value_arch mlp|bilinear. Design spec: `docs/superpowers/specs/2026-06-30-hiql-flow-actor-design.md` (gitignored, local). TDD: `tests/test_hiql_flow_injection.py` (4 tests) + gciql/gcivl regressions all green. **UNCOMMITTED** (along with gciql_flow fb_forward_z + measure_ortho_coef).

**IMPORTANT gotcha:** HIQL goal_rep ends in LengthNormalize → an all-zeros smoke batch gives 0/0 NaN (stock HIQL NaNs too); use a RANDOM batch in hiql smoke tests, not zeros. Isolation timing proved sample_actions compiles once (~1ms/call, no recompile) — production 18s/it was GPU OVER-SUBSCRIPTION (9 extra jobs on 8 already-100%-util GPUs; GPU compute contention is invisible in CPU load average), NOT a code bug.

**RUNNING now (tmux):** `fbz` (FB probe, ~400k); `readybatch` = 9 runs (GCIVL-stock+flow `GCIVL-State-Flow-Stock`, stock-native HIQL `HIQL-State-Stock`, stock-native GCIVL `GCIVL-State-Stock`), logs `/dev/shm/ready_batch_logs/`. **DEFERRED:** `hiqlflow_watch` watcher waits for readybatch "ALL DONE" then runs `/dev/shm/launch_hiql_flow.sh` = 9 HIQL+flow ablations (low/high/both × 3, bilinear-d50, groups `HIQL-State-Flow-{lvl}-Bilinear-d50`). **OPEN:** collect all @500k overall_success → finish the bilinear×{native,flow} × {HIQL,GCIVL} + stock table; commit the flow code. Relates to [[hiql-gcivl-bilinear]], [[gciql-flow-agent]], [[fb-graft-audit-and-fbz-probe]], [[always-run-in-tmux]].
