---
name: hiql-bilinear-brainstorm
description: "Why HIQL barely benefits from bilinear — H1 rep-starvation / H2 BVN-form / H3 cube-ceiling hypotheses, published OGBench baselines, antmaze 8-seed discriminator RUNNING (tmux amhiql)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 061c5b6d-fefa-4082-ada9-d56198028c2d
---

2026-07-07 brainstorm (mid-flight; spec doc NOT yet written — task #4 pending).
Question: why does bilinear buy GCIQL +26pts / GCIVL +18-30pts on cube but HIQL only
+4 native (0.17→0.21 sg10) and −12 with flow-both (0.26→0.14)?

**Ranked hypotheses** (grounded in [[sota-bilinear-paper]] spec + audit):
- **H1 (primary, explains flow harm + capped win): bilinear starves the shared goal_rep.**
  In stock HIQL the value loss is the SOLE gradient source for goal_rep (low actor stop-grads
  it, high actor stop-grads its target; hiql.py:96-135). Bilinear option A routes that sole
  gradient through a rank-≤50 dot product φ(s)·ψ(goal_rep([s;g])) → rep can collapse to the
  ~50 value-relevant directions; but goal_rep is ALSO the low actor's conditioning input and
  the high actor's OUTPUT space. Flow high-actor (generates a distribution in rep space +
  √rep_dim renorm at eval) is most rep-geometry-sensitive → worst cell. Free probe: effective
  rank of goal_rep outputs, MLP-run vs bilinear-run checkpoints.
- **H2 (co-primary, explains tiny native win): BVN-form confound.** HIQL's _BilinearRepValue
  = φ(s)·ψ(goal_rep([s;g])) — s on BOTH sides → NOT low-rank over (s,g); the structure that
  wins in GCIQL/GCIVL (Schaul form ψ(g)) was never imposed. GCIVL is the matched control
  (same V-only expectile + AWR readout, Schaul form, +18pts) — kills "AWR readout" explanations.
  Discriminator: Schaul-form HIQL bilinear cell (= spec M3 axis run in reverse).
- **H3 (background, first-order on cube): HIQL's cube ceiling is hierarchical.** Published
  OGBench (1M, 8 seeds, avg last-3 evals): HIQL cube-single = **15±3** (near-worst method)
  but scene = 38 (mid-pack) and antmaze-medium = 96 (owns it). GCIQL 68/51/71, GCIVL 53/42/72,
  CRL 19/19/95, QRL 5/5/88, GCBC 6/5/29. Our sg10 MLP ~0.17 REPRODUCES published 15. Sharper
  RLBrew tie-in: hierarchy fails on FREE-OBJECT transport (cube), not manipulation per se
  (scene articulated objects OK).
- H4 (dismissable free): AWR exp_a saturation — GCIVL argues against; check wandb stats.

**Facts settled this session:** OGBench protocol = avg of last 3 eval epochs (100k apart), 8
seeds, 1M, 50 eval eps (spec's open TODO — resolved). Per-level bilinear (low-only/high-only)
NEVER tried and not expressible now (single shared V queried with different goals; would need
a two-head split — promising H1/H2 discriminator cell: bilinear high + MLP low). HIQL+DDPGBC
is not a defined cell (no Q(s,a); both actors AWR).

**RUNNING: `tmux amhiql`** — antmaze-medium HIQL {MLP, bilinear-d50} × 8 seeds @1M,
OGBench-faithful HPs (= pure defaults, sg=25 correct for antmaze). The cleanest discriminator:
max headroom (96), zero manipulation confound. If bilinear fails to help HERE → H1/H2 (form/rep
pathology); if it helps → cube ceiling (H3) was hiding the win. See [[tmux-sessions-registry]].

**Next:** collect amhiql → goal_rep effective-rank probe on existing cube sg10 checkpoints →
possible cells: Schaul-form HIQL ψ(g), high-level-only bilinear split, low_actor_rep_grad=True
rescue, HIQL-scene (published 38 = headroom on manipulation). Then write the spec addendum
(M6 "HIQL boundary mechanism") to [[sota-bilinear-paper]]'s spec.
