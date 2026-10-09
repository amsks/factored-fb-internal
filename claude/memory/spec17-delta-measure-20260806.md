---
name: spec17-delta-measure-20260806
description: "2026-08-06 spec17 IMPLEMENTED, COMMITTED (79a9c6b+370938f) and the cube-single half LAUNCHED (tmux s17cs, 6 jobs, seeds 0/1/2, 500k, /home/adityamohan/exp_spec17): DELTA successor measure J=(1-γ)M−h(x0); 17a measure_delta + 17b measure_delta_partition, 46 tests green. Review caught 2 launch blockers first — measure_coef default 0.0 zeroed the 17b CE (would have logged a FAKE NULL), and delta_protos as a pytree field broke restore for EVERY pre-existing checkpoint"
metadata: 
  node_type: memory
  type: project
  originSessionId: b53886da-0c88-4efb-9557-afa31be3742f
  modified: 2026-08-06T11:04:49.022Z
---

Bridge track, follows [[spec16-multistep-measure-20260806]]. Source: an
external review proposing the "centered/net-flow successor measure"; user
renamed it DELTA (avoid "flow"=FlowBC collision, "centered"=T10 collision)
and delegated coding to another agent — the specs are the handoff artifact.

**Specs (untracked, MOVED 2026-08-06 to `PAPER/Multistep-FB/specs/` — see
[[primary-paper-dir-multistep-fb]]; they were originally in
`PAPER/rescuing-fb/specs/`):**
- `17-delta-measure.md` — science + learnings checklist + pre-registered
  joint wave §6 + gates G1–G4/NULL + anchors. Key identity:
  J(s,a,g)=E[Σγ^t(h(x_{t+1})−h(x_t))] = (1−γ)M − h(x0) ∈ [−1,1], SAME
  action argmax as M; dwell (inside→inside) contributes 0 to every target;
  degenerate constant RELOCATES to 0 (zero-flux, visible) not eliminated.
  Eval conversion M̂=(h+Ĵ)/(1−γ) required before any calibration/L2 quote.
- `17a-delta-measure-impl.md` — variant A `measure_delta`: in measure_loss's
  fp hit path, hit ← hit_n − hit_c (hit_c = _hit_matrix on OBSERVATIONS —
  off-by-one is a silent object change, test pinned), absorbing=False route
  reused; no new modules ⇒ warm_start works; lattice mirrors nstep block;
  ⚠ ballvisit d50 base configs are the FROZEN-PSI phase-2 recipe — the
  from-scratch cs delta config MUST set measure_stopgrad_psi: false.
- `17b-delta-partition-impl.md` — variant B one-stage simplex measure:
  φ=softmax-RBF over K dataset prototypes (per-block ε scaling from
  measure_hit_spec), delta_head MLP([s,a,g])→K logits ×2 heads + target
  head (requires an `update` override adding target_update('delta_head') —
  agent currently inherits base update; train loop = impls/main.py), CE to
  sg[(1−γ)φ(x')+γp̄], actor q = p·φ(goal) at FROZEN head params,
  actor_q_center forced; protos via make_delta_prototypes.py into
  results/delta_protos/ (NOT /dev/shm); agent gains dataclass field
  delta_protos. NO ortho/anchor/hinge/scale_norm anywhere (T10-b/9.15c).

**Why (the critique-integration session, same day):** multistep visitation
targets risk amplifying the dwell-level channel (the §9.15c mass-anchor
ordering mechanism); delta deletes the level from supervision instead.
A vs nstep in IDENTICAL warm-start phase-2 slots = two rival mechanisms for
the §9.16 collapse (starved bootstrap vs level-dominated regression).
B = cleanest estimator-vs-semantics deconfounder (exact occupancy
reparameterization, dense simplex targets, no hindsight, one stage); its
pre-registered failure signatures separate zero-flux collapse (entropy→logK)
from the semantic verdict (calibrated p + dead actor ⇒ occupancy ranking is
the wrong object ⇒ fp+dwell-head becomes the program).

**Prior-art check done:** temporal centering NEVER tried here. Adjacent
records: T10 centered_measure = ACTION centering (aux .823/−8pts; only-mode
runaway→anchored dead band .06–.10); SF-feature-TD vector TD collapse ~0.09
(confounded readout — frozen/identity ψ); SimpleSF representability wall
(TD-shaped base, not fixed partitions); §9.15c hard-simplex ordering risk =
pre-registered gate G4 (Spearman ≥.65), fallback bounded-unnormalized head.

**Wave (§6, NOT launched, 8 GPUs free at spec time):** cd/sc phase-2
warm-start slots (ckpts exp_spec08/factored-fb/GCIQL-{CubeDouble,Scene}-
Flow-Measure-FirstPassage-NoOrtho-d50/{cd,sc}_fp_s0 @500k): delta-batch,
delta-hindsight, nstep64-batch each; cs from scratch: delta, partition
K=256, fp+nstep64 hold gate. eval_interval 10000 on cd/sc (shock window).
Anchors: cs .967 fp / ~.1 visitation floor; cd p1 .549 / p2 batch .358 /
hs .53; sc p1 .484 / p2 batch .342 / hs .927.

**IMPLEMENTED + COMMITTED 2026-08-06** (parallel subagents; 46 tests green:
delta 8, delta_partition 12, nstep 6, hit_spec/axes/scale/zmix 20).
`79a9c6b` lands gciql_flow.py wholesale — HEAD's copy predated the ENTIRE
measure program (no `_parse_hit_spec`, no `measure_first_passage`), so the
already-committed measure test suites could not pass against HEAD; specs
06/08/14/16 necessarily ride along (one file cannot be split into per-spec
commits without committing a version that does not run). `370938f` = review
follow-ups. train.py deliberately EXCLUDED (torch-FB track's reward_z work).

**⚠ TWO LAUNCH BLOCKERS the opus review caught — both would have burned the
wave, neither caught by any test:**
1. `config.measure_coef` defaults to **0.0** and `total_loss`'s `measure_only`
   branch multiplies by it. Spec 17b §6 tells the config to DROP the knob ⇒
   the partition CE was weighted by zero: delta_head frozen at random init,
   yet `delta_ce` and a near-`log K` entropy still logged — indistinguishable
   from the pre-registered scientific NULL on gate G3. FIX: the 17b branch is
   weighted by `measure_delta_coef` only, and `create()` now refuses
   `measure_coef>0` under the knob.
2. `restore_agent` runs `flax.serialization.from_state_dict` over the WHOLE
   agent (`impls/utils/flax_utils.py:198`), so the new `delta_protos` pytree
   data field made EVERY pre-17b checkpoint fail restore ("Missing field") —
   breaking ~14 eval harnesses incl. the spec-14 L2 battery and the §9.16
   anchors. Spec 17b §2.2's "old checkpoints keep loading" is FALSE. FIX:
   `nonpytree_field` holding a hashable tuple (1 XLA compile / 12 steps,
   uint32-equal round-trip). CONSEQUENCE: the partition a 17b ckpt was
   trained on lives ONLY in the .npy, never the checkpoint —
   `results/delta_protos/cube_single_K256_s0.npy` sha256 `b4e13e9d9e0a…`,
   chmod 444, backup `~/delta_protos_backup/`. A regenerated/permuted file
   silently changes the q-term with no error.

**LAUNCHED 2026-08-06 03:50** tmux `s17cs`, `scripts/rescuing_fb/launch_spec17_cs.sh`,
OUTROOT `/home/adityamohan/exp_spec17`, 6 jobs GPUs 0-5, from scratch, 500k,
seeds 0/1/2: `d17_cs_delta_s{0,1,2}` + `d17_cs_part_s{0,1,2}`. cd/scene arms
of §6 NOT launched. First reads @15–35k: partition entropy 5.545(=logK)→3.99,
p_max 0.118 vs uniform 0.0039 ⇒ head genuinely training; delta arm
event_rate ~0.0013 vs hit_level ~0.0148 (events ≈9% of dwell mass, by design),
msq 0.34–1.70 ⇒ gate-G2 instrument live and bounded away from 0.

**Still open:** L1fix/L2 harness `--delta` conversion flag and the partition
`z_r=r(protos)` readout script (both in flight); the cd/scene §6 arms; CSV
refresh (`scripts/results_map.py` + `compile_results_csvs.py --refresh`) after
collection. 17a refuses delta×nstep composition in round 1.

Related: [[spec16-multistep-measure-20260806]] (rival arm, same wave),
[[p2-multienv-decay-mechanism-20260730]] (the collapse + anchors),
[[validmeasure-campaign-20260729]] (G4's provenance),
[[firstpassage-measure-works-20260722]] (no-ortho lesson).
