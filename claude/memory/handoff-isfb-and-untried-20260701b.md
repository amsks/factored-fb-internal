---
name: handoff-isfb-and-untried-20260701b
description: READ FIRST (2026-07-01 late) — ISFB agent built+running (SG-ISFB) + untried B1/C2/exp4 campaign + cube-only paper remodel & audit; all code UNCOMMITTED on gciql-fb-bridge
metadata: 
  node_type: memory
  type: project
  originSessionId: a2505c06-4c7d-4d6c-aaf8-06b650716d8f
---

**READ FIRST. 2026-07-01 (later than [[handoff-session-20260701]], which it supersedes).**
Two live campaigns, a NEW agent (ISFB), the fb-gciql-bridge paper remodeled cube-only +
audited, all code TDD-green but UNCOMMITTED on branch `gciql-fb-bridge`.

## 1. Live campaigns (check `tmux ls`)
- **tmux `isfb`** — the ISFB Exp-1 ladder (PAPER/fb-modified/isfb.tex): 9 runs @500k =
  {scalar_gate_hard, scalar_gate_awr, coord} × 3 seeds. At handover 7 launched (hard s0/s1/s2,
  awr s0/s1/s2, coord s0; ~33–61k, 0 errors); **coord s1/s2 QUEUED** on the free-GPU watcher
  (all 8 GPUs busy: 7 isfb + straggler). Watcher `/dev/shm/isfb_launch.sh`, logs `/dev/shm/isfb_logs/`.
- **tmux `untried`** — B1/C2/exp4 (see §3): C2 ×3 DONE, exp4 s0/s1 DONE (s2 killed-stuck),
  B1 s0/s2 DONE, **B1_s1 straggler ~241k** on gpu1 (was slow-eval under CPU contention).
- Old idle sessions bridgeabl/bridgeends/bridgewatch = leftovers, ignore.

## 2. COLLECT (wandb `amsks/factored-fb`, key `evaluation/overall_success` @500k)
- **ISFB** `ISFB-State-{ScalarGateHard,ScalarGateAWR,Coord}-d50` vs the **C4 16 → B4 87** band.
  Near 87 ⇒ FB's missing ingredient was the in-sample improvement operator. hard/awr **vs coord**
  tests the incoherence critique (coord is now the CLEAN coordinatewise baseline, isfb_alpha_v=0).
- **Untried** `GCIQL-State-Flow-Bilinear-d50-{B1-FBzIQL, C2-MeasureTwoTower, Exp4-ActorZInfer}`.
  B1 vs 90/87/25.5 · C2 vs MeasureOnly 9.9 · Exp4 vs G4 exact-93/inferred-7 (**only 2 seeds**, s2 lost).

## 3. What ISFB is (the new agent)
SG-ISFB = IQL's in-sample improvement inside FB's successor measure, no a+~pi. Shared-B bilinear
critic **`GCISFBValue`** (F, F_V, one B=psi): M=<F(s,a,z),B>/√d, N=<F_V(s,z),B>/√d.
`L_V=expectile(sg q−v)` (q=<F,z>, v=<F_V,z>) → advantage A → weights (hard τ | AWR exp(βA)) →
weighted vector regression N→M → M bootstraps through **γ·mask·N⁻** (masked, GCIQL-consistent).
`measure_insample∈{scalar_gate_hard,scalar_gate_awr,coord}`. B1/C2/exp4 = the 3 untried bridge
cells the audit found empty (B1 = fb_forward_z under IQL; C2 = measure+two-tower; exp4 = actor
trained on inferred z=E[r·psi]).

## 4. UNCOMMITTED (branch `gciql-fb-bridge`; commit next — push blocked on box auth)
- **M `tools/wandb_mode_shim/gciql_flow.py`**: (B1) relaxed `assert measure_only` on fb_forward_z
  + added `critic_loss` override + z-threading in `value_loss` so fb_forward_z runs under IQL;
  (exp4) `actor_z_infer` knob + `_infer_actor_z` helper + actor Q via <phi,z>; **`GCISFBValue`**
  module (~line 150) + `isfb` critic branch (~line 530).
- **NEW `tools/wandb_mode_shim/isfb_flow.py`**: `ISFBAgent`, `isfb_loss` (masks + isfb_alpha_v),
  `get_config`. Re-exports GCISFBValue.
- **M `tools/wandb_mode_shim/sitecustomize.py`**: register `isfb_flow`→ISFBAgent.
- **NEW tests (all green)**: `test_gciql_untried_bridge.py` (B1/C2/exp4), `test_isfb_agent.py`
  (6, incl. mask test). Gotcha found: measure-objective finite tests MUST use a random batch —
  an all-zeros batch is rank-degenerate and NaNs (agent is stable on real data).
- **NEW configs** `configs/gciql/`: B1_fbz_iql, C2_measure_twotower, exp4_actorZinfer, and
  isfb_{scalar_gate_hard,scalar_gate_awr,coord} (coord sets **isfb_alpha_v=0**).
- Plus still-uncommitted prior work (B4/C4/D1 + project_z/actor_z_from_critic knobs & tests).

## 5. PAPER (gitignored, local-only)
- **`PAPER/fb-gciql-bridge/fb-gciql-bridge.tex`** — REMODELED into a coherent table-first note:
  §1 design decisions D1–D8 menu → §2 the morph ladder (master combinations table) → §3 one-knob
  isolations → §4 endpoints → §5 synthesis → §6 [BEING UPDATED] 1M tier. **CUBE-ONLY** (AntMaze+Scene
  removed per user). PDF built (`fb-gciql-bridge.pdf`) via **tectonic musl static binary** at
  `…/scratchpad/tectonic` (session-scratch, volatile — refetch the musl build if gone; box has no LaTeX).
- **Audit verdict**: paper ACCURATE (32/32 numbers match wandb). Fixed 2 real errors I'd introduced:
  B4 mislabeled "z-fwd"+fabricated assert-relaxation footnote (B4 uses `fwd_two_tower` not `fb_forward_z`);
  D8 knob attribution (`train_goal_ratio`/`decouple_actor_z` live in `agents/fb/agent.py`, NOT gciql_flow).
- **`PAPER/fb-modified/isfb.tex`** (USER-authored) = SG-ISFB spec. **`fb-modified.tex`** = the parent
  objective note (IS-FB §4; SG-ISFB is its refinement). Open design point: F_V double-constraint —
  v=<F_V,z> ties the scalar V to the vector F_V (kept as-written); switch to a separate scalar V head
  if L_V and L_N fight (diagnose via advantage histogram).

## 6. NEXT
1. Collect both campaigns @500k. Add isfb.tex §Diagnostics (advantage histogram, coord κ score) before over-interpreting.
2. If hard/awr ≈ 85–90 → operator confirmed; fold SG-ISFB shared-B note into fb-modified §4; run Exp-2 (full-support z) then Exp-3 (=exp4). If it plateaus < B4 → representation-under-resolves-interaction (paper thesis) → try SF+GPI (fb-modified 1B).
3. Commit all §4 code. coord s1/s2 auto-launch when a GPU frees.

## 7. Watch-outs
NEVER co-init 2 jax on one GPU (deadlock at step 0) — the watcher must poll until the launched GPU
shows >1500MB before the next launch (I hit + fixed a double-assign bug). `eval_on_cpu=1` × 8 concurrent
→ slow evals under CPU contention (stalled B1_s1/E4_s2). [[jax-gpu-packing-preallocate]]
[[always-run-in-tmux]] [[wandb-entity-amsks]] [[host-identity-you-are-aws-gpu]] [[commit-no-claude-coauthor]]
