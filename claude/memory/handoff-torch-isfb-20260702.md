---
name: handoff-torch-isfb-20260702
description: "READ FIRST (2026-07-02) — faithful SG-ISFB reimplemented in TORCH on agents/fb/ (only isfb.tex additions), TDD-green + COMMITTED on gciql-fb-bridge; Exp1 3x500k campaign RUNNING (tmux isfbtorch); KEY: torch FB band is INVERTED vs JAX (plain FB 48% > FB-IQL 24%)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 065a9135-b1f9-4b78-ae23-92513741a64c
---

**READ FIRST. 2026-07-02.** Supersedes [[handoff-isfb-and-untried-20260701b]] for the ISFB thread.

## What happened
An audit (2 subagents: canonical torch FB `agents/fb/` vs the JAX `isfb_flow.py`) found the
JAX ISFB was **not faithful FB** — it silently dropped two *matched* load-bearing FB decisions:
(a) the training z-distribution (canonical FB = 50/50 random-sphere-z ⊕ z=B(goal); JAX only ever
uses z=B(value_goals)) and (b) reward-weighted zero-shot z-inference z=E[r·B]. User decided:
**reimplement SG-ISFB in TORCH on top of the canonical FB agent**, adding ONLY the isfb.tex pieces.

## Built + COMMITTED (branch gciql-fb-bridge, 5 commits e1bb7db..9eddeb8)
- **`agents/fb/isfb/model.py`** `ISFBModel(FBFlowBCModel)`: adds `_value_forward_map` (F_V) =
  a 2nd ForwardMap fed a **zeroed action** (arch-identical to F) + `_target_value_forward_map`.
- **`agents/fb/isfb/agent.py`** `ISFBAgent(FBFlowBCAgent)`: flow actor UNCHANGED; overrides
  `update_fb` with SG-ISFB loss (M=⟨F,B⟩, N=⟨F_V,B⟩, q=⟨F,z⟩, v=⟨F_V,z⟩; L_V=expectile(sg q−v);
  A=sg(q−v); hard/awr/coord weights; L_N weighted regression fitting N→sg M; **L_M bootstraps
  through N⁻=⟨target_F_V(s'),B⟩, NO a'~π**; ortho on B coef 1.0). F_V rides forward_optimizer;
  target F_V soft-updated in update_fb. `isfb_hard_weights(A,tau)` helper.
- **`train.py`**: `cfg.agent=="isfb_flowbc"` branch. **`configs/agent/isfb_flowbc.yaml`**: inherits
  `fb` DIRECTLY (NOT fb_flowbc — 2-level defaults mis-orders _self_), flow actor keys inlined,
  measure_insample/alpha_n/alpha_m/awr_beta/awr_wmax + ortho_coef=1.0 + train_goal_ratio=1.0.
- **Tests (22 green)**: `test_torch_isfb_{model,agent,wiring}.py` (incl. "no a'~π in critic",
  coord omits L_V, all 3 variants finite). `test_fb_iql.py` still green (agents/fb/ untouched).
- End-to-end smoke on cube (all 3 variants) PASSED: losses finite+decreasing, eval runs, coord L_V=0.
- Launch invocation: **`domain=cube_single agent=isfb_flowbc`** (bare `agent=` under default
  antmaze domain does NOT swap the group — domain's defaults list selects the agent).
- Spec `docs/superpowers/specs/2026-07-02-torch-isfb-design.md`; plan `docs/superpowers/plans/2026-07-02-torch-isfb.md`.

## RUNNING (v2, CLEAN after code review): tmux `isfbtorch2`, logs /dev/shm/isfbtorch2_logs
9 runs {coord, scalar_gate_hard, scalar_gate_awr}×{0,1,2} @500k cube, wandb amsks group
**ISFB-Torch-Exp1-fixed**, names `isfbt2_<mode>_s<seed>`. Collector bg. COLLECT `success/overall` @500k
vs ~0.65 torch ceiling. (The FIRST campaign `isfbtorch`/group ISFB-Torch-Exp1 was mis-specified —
KILLED at ~65%; its @250k means were coord~0.15 hard~0.19↗ awr~0.07. Ignore those.)

## CODE REVIEW (user, 2026-07-02) — fixes applied, TDD-green, COMMITTED (2f88afc + AWR commit)
Review found the first campaign was NOT the clean exact-goal test. Fixes (all in agents/fb/isfb/):
1. **train_goal_mode: traj** (was default "perm" → z=B(random permuted goal), not exact-goal). Now
   z_i=B(s_i^future). THE decisive fix. (needs future buffer — train.py sets future_gamma when traj.)
2. **isfb_goal_mask** (new knob, ON for Exp1): cut measure-TD bootstrap at goal-success γ(1−success),
   GCIQL-consistent (the JAX ISFB masked; torch port had dropped it). Helper `isfb_goal_success`.
3. **×P** on L_V/L_N (fb_successor_terms sums over P=2 heads; new terms were per-mean → underweighted).
4. **AWR** now on **std-normalized advantage** (`isfb_awr_weights`): the FB scalar <F,z> INFLATES over
   training (adv_std 3→54 by step 14k; q→1455), so any fixed β re-saturates the wmax clip → the first
   campaign's awr collapse (0.07). std-norm makes β scale-invariant; awr_beta=1.0. hard(sign)/coord are
   scale-invariant so unaffected. Added adv_std/min/max logging.
5. asserts on ignored FB opts (w/goal_obs/next_action_override) + clip _left_encoder.
WATCH-OUT: measure/loss scale inflates unbounded over training (q,adv,L_V,L_N,L_M → 1000s). B is
norm-capped (Norm layer) so it's F growing — known FB behavior; benign for hard/coord (scale-inv) and
the flow actor (FQL scale-norm distill). If success collapses, revisit (maybe ortho too weak at 1 vs the
stock-FB 1000, or need an F norm). Watch the 50k eval.

## TARGET (user, 2026-07-02): match/beat the torch bilinear-GCIQL ceiling WITH the measure objective
Goal is NOT "beat plain FB 48%" — it is to reach the bilinear-GCIQL range using SG-ISFB's MEASURE
objective. On the TORCH substrate the reachable ceiling @500k×3 (all already trained, wandb, group
via `fb_bridge_*`, agent=fb_flowbc, critic=iql = the IQL/reward-TD objective) is **~0.65 mean**
(fb_bridge_r2_dnorm/idB_dnorm 0.65, top seeds 0.72–0.74; the "0.74" is one seed of r1_decond,
mean 0.58). So SG-ISFB target = **≥ ~0.65 mean** (measure objective competitive with reward-TD).
No re-run needed — compare ISFB-Torch-Exp1 directly against these fb_bridge cells @500k.

## KEY FINDING — the torch band is INVERTED vs isfb.tex's premise
isfb.tex's **16→87 (C4→B4)** gap is a **JAX/GCIQL two-tower** phenomenon (paper's 87.1 = bilinear
+two-tower on the GCIQL side). On the **faithful torch** substrate:
- plain torch FB (measure, C4-analog) ≈ **48–50%** (`fb_state_plain`/`L256`, wandb mean 0.50 @1M).
- torch FB-IQL critic (learned B, B4-analog) ≈ **23.6%** (PAPER/fb-gciql-bridge line 310; zero-shot
  z 24±7 line 320) — i.e. FB-IQL is BELOW plain FB on torch.
So SG-ISFB's honest torch target is **"beat plain FB's ~48%"**, NOT "reach 87". Existing FB-IQL
runs (`fb_state_iqlcritic_fix_s0..9`, `agent=fb_flowbc critic_mode=iql train_goal_mode=traj`) are
at 1M — if a clean 500k anchor is wanted, run plain-FB + FB-IQL at 3×500k too (deferred; user said
"we have FB-IQL already").

## NEXT
1. Collect ISFB-Torch-Exp1 @500k vs plain-FB 48% / FB-IQL 24%. 2. If SG-ISFB > 48% → operator helps
real FB; else the null holds on the faithful substrate too. 3. Only then Exp2 (full-support z:
train_goal_ratio<1) + Exp3 (inferred-z actor via reward_inference) — both need NO structural change.
[[handoff-isfb-and-untried-20260701b]] [[always-run-in-tmux]] [[wandb-entity-amsks]] [[host-identity-you-are-aws-gpu]] [[commit-no-claude-coauthor]]
