# Claims audit — edited_main.tex vs. run provenance (2026-08-05)

Trigger: the two implementations of the region method were found to diverge
**41×** on cube-double and **8.5×** on scene (same objective, same data, same
500k budget), while agreeing on cube-single (0.922 vs 0.967). Every number in
the draft therefore needs a provenance tag before it can be trusted.

## The two agents (verified from configs + code, not memory)

| | **jax** (`gciql_flow.py`) | **torch** (`agents/fb/measure`) |
|---|---|---|
| critic | `M(s,a,g) = φ(s,a)·ψ(g)` — **no z exists** | `M(s,a,z,g) = F(s,a,z)ᵀψ(g)` |
| TD columns | batch next-obs (`tpd='batch'`) | batch next-obs (native n×n) |
| actor | `π(a\|s,g)`, `actor_cond='goal'` (default; no paper cell overrides it) | `π(a\|s,z)`, z from **permuted-batch** goals (`train_goal_mode='perm'`) |
| actor's training goals | trajectory-future (OGBench mix) | permuted batch ⇒ reachable w.p. ≈ region mass |
| "zero-shot" readout | z_r → **nearest-neighbour decode to a dataset goal** → GC actor | z_r fed to the actor **directly** |
| multi-object success (p1) | cd **0.549**, sc **0.484** | cd **0.013**, sc **0.057** |
| cube-single success (p1) | 0.967 | 0.922 |

The gap ordering tracks region mass exactly (cube-single 1.3–1.7% →
scene 0.37% → cube-double 0.033% gives 1.0× → 8.5× → 41×): a permuted-batch z
almost never indexes a reachable conjunctive goal, so the torch actor never
trains where it is evaluated — §9.16's grounding mechanism, relocated from the
critic's columns to the actor's conditioning index. 3 points, 2+ confounded
axes ⇒ consistent-with, not proof. The no-flow wave (running) and a torch
`train_goal_mode='traj'` arm are the discriminating experiments.

## Verdicts, claim by claim

### VALID — survive as stated

1. **§playdata concentration statistics.** Data-side, stack-free.
2. **§frozenrep divergence** (154→5.4e4, 0.98→0.42, ψ-trainable 0.97→0.07).
   One frozen representation, three objectives, one stack, cube-single. The
   closing cross-stack sentence ("tuned FB 0.42 vs GCIQL 0.928") is
   cube-single-only, where the stacks agree; tolerable.
3. **§rq1 headline: region TD > matched GCIQL on all three envs**
   (0.967/0.549/0.484 vs 0.928/0.265/0.281). Both rows are jax,
   goal-conditioned flow actors, same data. This is the paper's strongest
   surviving claim and is genuinely apples-to-apples.
4. **§rq1 2×2 ablation** (0.284 / 0.108 / 0.922 / effector 0.823). All torch,
   all cube-single, same stack throughout.
5. **§rq2 reward transfer** (17.2 → 32.4 = 1.9×; twozone gap_frac > 1;
   readout weight; reward-z coverage). All torch, cube-single, internally
   consistent — and the actor consumes z_r directly, so the compositional
   claim is real *there*.
6. **§rq4 jax-internal results** (batch-p2 regression, hindsight repair,
   scene 0.872 zero-shot). The NN decode and the hindsight-is-not-ρ caveat
   are disclosed in the text.

### FALSIFIED / CONFOUNDED — must change

7. **"Successor-style baselines degrade where occupancy concentrates"** — dead
   on both halves.
   - *"CRL scores 0.10–0.31 across implementations"*: *falsified 2026-08-05.*
     Matched jax CRL+flow scores **0.384 on scene** (single seed; s1 running) —
     the best CRL cell in the project, above the matched GCIQL baseline
     (0.281) and above our phase-2 (0.342), on the environment with the
     **strongest** concentration. (cube-double: 0.044.)
   - *"FB drops to 0.047 on scene … consistent with the proposed mechanism"*:
     *confounded.* The 0.047 is the torch z-conditioned stack — in which **our
     own region objective scores 0.057/0.067 on scene and 0.013–0.022 on
     cube-double**. The number measures the z-conditioned actor's failure on
     sparse conjunctive regions, which hits every objective equally
     (2026-08-05 wave: stock FB 0.00–0.02, ortho1000 FB 0.01/0.12, FB-on-regions
     0.02/0.07, ours 0.01–0.07). It cannot discriminate density vs region.
8. **Table 1 "architecture-matched" block** mixes stacks: GCIQL + CRL are jax
   (valid vs our jax rows); Tuned FB / FB-ref / FB-on-regions are torch. Fine
   on cube-single (stacks agree). The 2026-08-05 multi-object fill-ins
   (all ≤0.12) **must not** be dropped into this block against our jax
   0.549/0.484 — the comparison would attribute a stack/conditioning failure
   to the objective. Honest options: (a) quote only jax baselines (GCIQL, CRL)
   in the matched block and move all torch multi-object rows to a separate
   same-stack block that *includes our own torch numbers* (0.013/0.057); (b)
   wait for the torch `traj` arm and report a reconciled stack.
9. **Method section describes an agent that does not produce the multi-object
   numbers.** Eq. `M(s,a,z,g)=F(s,a,z)ᵀψ(g)`, "samples the task z and the goal
   g independently", and the z_r→actor readout describe the **torch** agent
   (cube-single only). Every multi-object number comes from a **z-free**
   goal-conditioned bilinear critic + GC actor, with z_r retrieval-decoded at
   eval. The Protocol paragraph never states actor conditioning. Required: a
   sentence in Protocol + §rq4 stating the instantiation split, and the
   abstract-level composite ("beats GCIQL on all three" + "1.9× transfer" +
   "calibrated measure") must be attributed to two artifacts sharing the
   objective, not one agent.
10. **§rq3 tail sentence** *"penalty-only runs already showed this degradation,
    implicating the penalty strength; a weaker penalty is under evaluation"* —
    wrong twice as of ledger §9.15c (2026-08-05): the **mass anchor**, not the
    hinge strength, causes the ordering loss (anchor-only Spearman
    0.483/0.468); and the weaker penalty (mcE) ran and **failed** its primary
    gate. Independent of the stack issue; must be edited now.

### Additional notes

- "replicated in a second implementation" (0.922) is true but env-limited: the
  implementations agree **only** on cube-single. Do not let it imply general
  inter-stack agreement.
- The multi-object "zero-shot" label is defensible but weak: reward → z_r →
  nearest dataset goal → GC actor can only *select among visited states*; it
  cannot express the compositional rewards that motivate the measure. The
  cube-single-only scope of the compositional claim should be stated where the
  0.872 is celebrated, not only in the protocol footnote.
- CRL 0.384 also *supports* one part of the story: CRL's bounded (saturating)
  contrastive objective is exactly the contrast the paper draws with FB's
  linear term in §failure. CRL surviving where FB floors is evidence *for*
  "the unbounded linear term is the problem" and *against* "all
  future-reachability objectives fail here". The paper currently claims the
  latter.

## Pending evidence that moves verdicts

| arm | status | resolves |
|---|---|---|
| no-flow region-FB (torch TD3+BC, z-cond), 3 envs × 2 seeds | running, tmux `noflow` | actor-class vs goal-distribution on the torch floor |
| CRL seed 1 (cd + scene) | running, GPUs 6–7 | whether 0.384 replicates |
| torch `train_goal_mode='traj'` (zHER: z = B(trajectory future)) | **not launched** — knob already exists in `agents/fb/agent.py` | the direct jax↔torch bridge: z-conditioning with grounded z's |
