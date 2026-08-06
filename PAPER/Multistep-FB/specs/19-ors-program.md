# Spec 19 — ORS program: occupancy reward shaping from the density measure, plus dwell-balanced FB

Status: WRITTEN 2026-08-06. Parents: spec 18 COLLECTED (density BEATS
partition: .896/.928/.964 vs .884 on cube-single; region-family zero-shot
.26–.28 vs part's .04–.18; K-sweep saturates at K≥256) + external proposal
2026-08-06 #3 (dwell-balanced FB + occupancy reward shaping) + the ORS
paper (ICLR 2026, arXiv 2604.20627: W₂ shaping from a learned occupancy
model; provably optimal-policy-preserving; OGBench cd 45±7 / triple 37±8 /
scene 80±4 vs GCIQL 40/7/51).

Implementation contracts (self-contained, one per implementing agent):
- `specs/19a-behavior-density-impl.md` — behavior-policy density head
  (`measure_density_behavior`) + GCDataset `next_actions`.
- `specs/19b-ors-extractor-impl.md` — r_W shaping reward from the frozen
  behavior density + GCIQL-flow extractor + calibration probe.
- `specs/20-dwell-balance-impl.md` — dwell detector/weights + torch-FB
  row-weighted arm.

## 0. One paragraph

The single load-bearing ingredient the multi-env campaigns isolated is TD
grounding DENSITY at actor-goal pairs (§9.16: hindsight = the only working
arm; batch columns dead; conjunctive hit rate 0.046%). ORS replaces the
sparse conjunctive indicator with a reward that is dense at EVERY (s,a,g):
r_W(s,a,g) = −W₂²(δ_g, d^{π_D}(·|s,a)) — for a Dirac goal this collapses
to an expectation over the behavior policy's future occupancy of a squared
goal distance, i.e. EXACTLY a successor-measure readout of the wide-support
reward −d²(x,g). Classical FB's linear readout is convicted in precisely
this regime (linear diag diverges as reward support widens, ms_abs_mean
→1100; zero-shot fades with reward bandwidth) — so we source the occupancy
from the spec-18 density head instead (normalized, valid by construction,
readout harness already exists), retrained in BEHAVIOR mode: no goal
conditioning, SARSA bootstrap with the dataset's next action. A frozen
behavior-density checkpoint then supplies r_W to a stock g-conditioned
GCIQL-flow extractor — which also respects the stack-divergence verdict
(z-conditioned extraction is the obstruction; extraction here is
g-conditioned TD). Dwell-balancing is the separate, weaker hypothesis
(spec-18's result — same rank, dwell-inclusive targets, .90+ — argues the
binding constraint was the estimator, not dwell's sample weight); it gets
one cheap deconfounded arm on the torch FB stack, sequenced after ORS.

## 1. Priority order (hand each item to an implementing agent in this order)

**P0 — debts that gate all interpretation (eval-only / one cheap run; do
these FIRST, they are prerequisites for reading every later result):**
- P0.1 G4 Spearman calibration read for density (d18 ckpts) AND part K256
  (d17 ckpts): Spearman of MC region masses vs read-out masses ≥ 0.65
  (protocol: spec 17 §6 G4; the valid-measure campaign's Spearman
  machinery is the reference implementation). For ORS this is not a paper
  gate but load-bearing engineering — the shaping signal is only as good
  as the occupancy ordering.
- P0.2 BC-only control: gciql_flow with the actor's q-term zeroed (pure
  BC flow actor, bc_coeff 3.0 as shipped; locate or add a knob — if added,
  default-off bit-identical + lattice), one run per env (cs first). No
  cube-single number in the program is currently attributable to the
  measure over the cloning term.
- P0.3 CSV refresh: add every `d17_*`/`d18_*` run_group to
  `scripts/results_map.py`, run
  `.venv/bin/python scripts/compile_results_csvs.py --refresh`.

**P1 — ORS on the density measure (the main bet):**
- P1.1 spec 19a: behavior-density training runs, cs/cd/scene, seed 0
  first (3 seeds on cs only after the P1.3 gate passes).
- P1.2 spec 19b §3 calibration probe on the P1.1 checkpoints — HARD GATE:
  Spearman(r_W ordering vs MC ground truth) ≥ 0.65 per env, else STOP and
  report (do not launch extractors on an uncalibrated signal).
- P1.3 spec 19b extractor wave: cs (tie gate), cd, scene (the
  discriminators). Anchors: cs density .93 avg / fp-noortho .967; cd
  phase-1 .549; scene hindsight .927 (the honest scene comparison — ORS's
  published 80±4 is BELOW our hindsight number; the question is whether
  ORS matches hindsight WITHOUT hindsight columns).

**P2 — dwell detector + weights (no training; independent of P1, can run
in parallel on a second agent):**
- spec 20 §1–2: `make_dwell_weights.py` + the validation report. HARD
  GATE: the phase-tagged report is reviewed before ANY training consumes
  the weights (the proposal's own §4 warning: pre-grasp stationarity is
  manipulation skill; a detector that downweights it poisons the arm).

**P3 — dwell-balanced FB (one deconfounded arm; after P2's gate):**
- spec 20 §3–4: torch FB `reweight_mode="dwell"` (row weights), cs/cd/
  scene vs plain-FB anchors (~0.42–0.48 cs) at the same commit.
  Pre-registered reading: dwell-FB ≈ FB strengthens the spec-18 estimator
  attribution (a RESULT); dwell-FB → density-level numbers = a genuine
  foundation-model result.

**P4 — deferred, gated on P1 (do NOT start):**
- Classical-FB-derived ORS (r_W ≈ (1−γ)F_Dᵀz_g^dist) as the
  foundation-model comparison arm — only if P1.3 succeeds.
- cube-triple env extension (new to our suite; where published ORS is most
  dramatic).
- Dwell-balanced density (row weights in the CE) — only if P3 wins.
- External flow-model ORS reproduction — quote published numbers instead
  unless a reviewer demands it.

## 2. Learnings applied

1. **Grounding density is the ingredient** (§9.16k) — ORS is a dense-
   grounding route that needs no hindsight columns and no phase switch.
2. **Wide-support rewards break the linear FB readout** (torch-FB
   deconfound; phase-2 audit bandwidth fade) — the shaping value comes
   from the NORMALIZED density readout, never FᵀB·z_r, in round 1.
3. **z-conditioning is the obstruction** (stack-divergence) — extraction
   is g-conditioned GCIQL; the occupancy model is unconditioned
   (behavior policy). No z anywhere in P1.
4. **Occupancy = arrival × dwell, r≥0.99** (spec 15) — dwell is SIGNAL
   for ORS (futures concentrate where objects stay); the occupancy model
   is trained on the UNWEIGHTED objective. Dwell weights touch only the
   representation-fitting arm (P3), never the shaping-signal arm (P1).
5. **Instrument the gate stat, not a proxy** (spec-18 lesson: mean vs
   median ESS) — every gate below names its exact statistic.
6. **Never compare mid-run evals** (spec-18 lesson: hard .984@450k vs
   .876@500k) — quote final-step multi-seed numbers only.
7. **Fake-NULL hazard** (17b measure_coef) — every new knob is either
   load-bearing and tested, or refused by the lattice.
8. **File couplings documented** (17b protos lesson) — the dwell-weights
   .npy and the ORS pool seed are recorded in config; neither is needed
   at checkpoint-restore time (training-time inputs only).

## 3. Pre-registered interpretations (program level)

- P1 wins cd/scene without hindsight ⇒ dense occupancy shaping replaces
  hindsight grounding ⇒ the paper's constructive answer to §9.16.
- P1 ties cs (expected: extraction is not binding there). A cs LOSS is a
  red flag for the shaping signal, not for ORS-the-idea — check the P1.2
  calibration first.
- P1.2 gate fails ⇒ the behavior-density occupancy is miscalibrated in
  ordering ⇒ fix the measure before touching shaping (and G4/P0.1 tells
  us whether the g-conditioned measure has the same defect).
- P3 null + P1 win ⇒ estimator (normalization/CE), not sample weighting,
  was the manipulation bottleneck; dwell hypothesis retired for OGBench
  play.
- Everything fails ⇒ the semantic hypothesis (first-passage + dwell head,
  spec 15 RSM) resumes as the program.

## 4. Wave hygiene (applies to every arm)

Standing recipe: d50, 500k, eval_episodes 50, eval_interval 50000 (cs) /
10000 (cd/sc), wandb entity amsks, tmux, distinct run_group per arm,
XLA_PYTHON_CLIENT_PREALLOCATE=false, XLA_FLAGS=--xla_gpu_enable_command_buffer=
(the d17_cs_part_s1 CUDA stream-capture fix). After every collection:
results_map + CSV refresh. Quote nothing from a mid-run eval.
