---
name: stack-divergence-audit-20260805
description: "2026-08-05 STACK DIVERGENCE CLOSED: wave T (traj-z) AND noflow control BOTH FLOORED on multi-object (cd <=0.06, scene <=0.08; cs holds 0.96/0.74 tz, 0.86/0.60 nf) => pre-reg branch fired: z-CONDITIONING ITSELF is the obstruction (not perm-z grounding, not the flow actor) => fork (B) decoupled hindsight-grounded actor head / jax no-z structure is load-bearing; audit PAPER/FB-Manipulation-ICLR/CLAIMS-AUDIT.md"
metadata: 
  node_type: memory
  type: project
  originSessionId: ffc6d4fb-3bc7-4cb3-8840-dd6544bf7f15
  modified: 2026-08-05T17:16:11.656Z
---

**Bridge track.** Found while filling tab:main's missing baselines
([[validmeasure-campaign-20260729]], [[p2-multienv-decay-mechanism-20260730]]).

**The divergence (verified from configs + code, NOT memory):**
- jax paper cells (`gciql_flow`, all multi-object numbers): critic
  `M(s,a,g)=φ(s,a)·ψ(g)` — **no task vector exists**; `actor_cond='goal'`
  (default, no measure config overrides it); actor goals = trajectory-future;
  zero-shot = z_r **nearest-neighbour decoded** to a dataset goal.
- torch (`fb_measure_flowbc`, ALL measure-validity + transfer numbers): true
  `M(s,a,z,g)=F(s,a,z)ᵀψ(g)`, actor `π(a|s,z)`, z from **permuted-batch**
  goals (`train_goal_mode='perm'`), z_r consumed directly.
- Success p1: cube-single 0.922(torch)/0.967(jax); cube-double 0.013/0.549
  (**41×**); scene 0.057/0.484 (8.5×). Gap monotone in region mass
  (1.5%→0.37%→0.033%) = §9.16 grounding mechanism at the ACTOR's conditioning
  index (perm-z reachable w.p. ≈ region mass).

**Falsified paper claims (full list: `PAPER/FB-Manipulation-ICLR/CLAIMS-AUDIT.md`):**
- "CRL scores 0.10–0.31" — matched jax CRL+flow scene = **0.384/0.304 (2 seeds,
  mean 0.344)** > GCIQL anchor 0.281 ≈ our p2 0.342, on the MOST concentrated
  env. cube-double 0.044/0.012. The "successor-style baselines degrade where
  occupancy concentrates" ¶ is dead (FB half confounded by stack: our own
  objective floors identically in torch).
- §rq3 "implicating the penalty strength; weaker penalty under evaluation" —
  wrong per §9.15c (anchor causes ordering loss; mcE ran and failed).
- Torch multi-object baselines (2026-08-05 wave: stock/tuned/ref FB ≤0.12,
  FB-on-regions ≤0.08) must NOT enter tab:main against jax ours — stack
  confound. CRL rows (jax) are the only matched new baselines.

**USER DECISION 2026-08-05:** cross-stack baselines acceptable; **our method
must be consistently reproducible in one stack** → make torch work multi-object,
then update the paper's method rows. Torch↔jax baseline reconciliation deferred.

**WAVE T + NOFLOW COLLECTED 2026-08-05 17:15 — BOTH NEGATIVE, question CLOSED.**
All 12 cells ran the full 500k (final.pt in `~/exp_trajz`/`~/exp_noflow`; tmux
sessions gone). Final `success/overall` from wandb (`amsks/factored-fb`):
- **tz (traj-z, flow actor):** cs **0.96/0.74**, cd **0.02/0.06**, scene
  **0.02/0.08**. No recovery toward jax 0.549/0.484 ⇒ the grounded-z bridge does
  NOT rescue torch multi-object; perm-z was not the problem.
- **nf (TD3+BC actor control):** cs 0.86/0.60, cd 0.02/0.02, scene 0.00/0.00.
  Actor class exonerated too — floors identically without the flow actor.
⇒ Pre-registered "still floored" branch FIRED: **z-conditioning of the
critic/actor itself is the obstruction** on conjunctive envs (consistent with
§9.16k "batch columns dead for actor training"). Waves U (phase-2 on traj-z)
and V (torch L2 conjunctive) are MOOT as chained. Next = **fork (B)**: decoupled
hindsight-grounded actor head (keep measure ρ-pure), i.e. move torch toward the
jax no-z structure — or accept jax `gciql_flow` as the method artifact (its
zero-shot harness now exists, spec14 §7, scene hindsight 0.872).

**Original wave-T plan (for reference; tmux `trajz`, GPUs 0–5):** torch p1 +
`train_goal_mode=traj` (zHER: z=B(own trajectory future) — grounded actor z's,
the jax bridge; knob existed already), 3 envs × 2 seeds,
`launch_traj_region.sh`, out `~/exp_trajz`. Pre-registered: cs must hold ~0.92;
cd/scn recovery toward 0.549/0.484 ⇒ torch = the method artifact, chain wave U
(phase 2: visitation+freeze_b+warm-start; ⚠ `reward_z_ratio` asserts perm at
agents/fb/agent.py:433 — revisit before rz-on-traj) and wave V (torch L2
extended to conjunctive regions ⇒ DIRECT-z multi-object zero-shot, stronger
than jax NN-decode). Still floored ⇒ z-conditioning itself is the obstruction
⇒ fork (B).

**Also running:** no-flow wave (tmux `noflow`, `FBMeasureAgent` =
RegionMeasureMixin + FBAgent TD3+BC actor, mixin refactor + 5 tests, 62 green)
— actor-class control, ~145k/500k. All groups in results_map
(nf_*/tz_*/mb_*/fr_*/crl multi-env).

**Gotchas this session:** hydra needs `"hit_spec='...'"` quoted-inside-quotes
or commas parse as a list; undeclared keys fail (declare in agent yaml);
launch-script arithmetic $((G++)) does not survive backgrounded forks — use an
explicit GPU table; drawio renderer = `scripts/render_drawio.py` (no
inkscape/drawio on box).
